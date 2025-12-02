import os
import base64
import tempfile
import uuid
import io
import requests
import json
import time
import torch
#from PIL import Image
import fitz
from pdf2image import convert_from_path
#import numpy as np
from tqdm.auto import tqdm
from transformers import CLIPModel, AutoProcessor
from qdrant_client import QdrantClient, models
#import shutil

class QueryEngine:
    def __init__(self, uploaded_file, session_id, progress_callback=None, poppler_path=None):
        self.uploaded_file = uploaded_file
        self.processed_data = []
        self.embedded_data_clip = []
        self.clip_model = None
        self.clip_processor = None
        self.qdrant_client = None
        self.collection_name = f"clip_multimodal_pdf_rag_{session_id}"
        self.embedding_dimension_clip = None
        self.ollama_model_name = 'gemma3:latest'

        # chage api_base from local to production environment
        #self.ollama_api_base = "http://localhost:11434"
        self.ollama_api_base = "http://ollama:11434"

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.progress_callback = progress_callback or (lambda x: None)
        self.POPPLER_PATH = poppler_path
        
        # Cache directory for HuggingFace models
        self.CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
        self.CACHE_DIR = "./hf_cache"

        self._process_pdf()
        self._load_embedding_model()
        self._generate_embeddings()
        self._setup_qdrant()
        self._ingest_data()

    def _process_pdf(self):
        self.progress_callback("Processing PDF...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, self.uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(self.uploaded_file.getvalue())

                pil_images = convert_from_path(file_path, poppler_path=self.POPPLER_PATH)
                doc = fitz.open(file_path)

                for i, page_image in enumerate(tqdm(pil_images, desc="Extracting pages")):
                    page_text = doc[i].get_text("text") if i < len(doc) else ""
                    page_text = ' '.join(page_text.split())

                    buffered = io.BytesIO()
                    if page_image.mode == 'RGBA':
                        page_image = page_image.convert('RGB')
                    page_image.save(buffered, format="JPEG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    self.processed_data.append({
                        "id": str(uuid.uuid4()),
                        "page_num": i + 1,
                        "text": page_text,
                        "image_pil": page_image,
                        "image_b64": img_base64
                    })
                doc.close()
            self.progress_callback(f"Successfully processed {len(self.processed_data)} pages/chunks.")
        except Exception as e:
            self.progress_callback(f"Error processing PDF: {e}")
            raise

    def _load_embedding_model(self):
        self.progress_callback(f"Loading CLIP model: {self.CLIP_MODEL_NAME}")
        try:
            if not os.path.exists(self.CACHE_DIR):
                os.makedirs(self.CACHE_DIR, exist_ok=True)
            
            self.clip_model = CLIPModel.from_pretrained(
                self.CLIP_MODEL_NAME,
                cache_dir=self.CACHE_DIR
            ).to(self.DEVICE).eval()
            
            self.clip_processor = AutoProcessor.from_pretrained(
                self.CLIP_MODEL_NAME,
                cache_dir=self.CACHE_DIR
            )
            
            if hasattr(self.clip_model.config, 'projection_dim'):
                self.embedding_dimension_clip = self.clip_model.config.projection_dim
            elif hasattr(self.clip_model.config, 'hidden_size'):
                self.embedding_dimension_clip = self.clip_model.config.hidden_size
            else:
                self.embedding_dimension_clip = 512

            self.progress_callback(f"CLIP model loaded successfully. Embedding dimension: {self.embedding_dimension_clip}")
        except Exception as e:
            self.progress_callback(f"Error loading CLIP model/processor: {e}")
            raise

    def _generate_embeddings(self):
        if not self.clip_model or not self.clip_processor:
            self.progress_callback("CLIP model or processor not loaded. Skipping embedding generation.")
            return

        self.progress_callback(f"Generating CLIP IMAGE embeddings for {len(self.processed_data)} items...")
        for chunk in tqdm(self.processed_data, desc="Generating Image Embeddings"):
            try:
                image_pil = chunk['image_pil']
                if image_pil.mode != 'RGB':
                    image_pil = image_pil.convert('RGB')

                inputs = self.clip_processor(images=image_pil, return_tensors="pt", padding=True).to(self.DEVICE)
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                image_embedding_vector = image_features[0].cpu().float().numpy().tolist()

                if image_embedding_vector:
                    chunk['embedding'] = image_embedding_vector
                    self.embedded_data_clip.append(chunk)
                else:
                    self.progress_callback(f"Skipping chunk on page {chunk['page_num']} due to image embedding error.")
            except Exception as e:
                self.progress_callback(f"Error generating embedding for page {chunk['page_num']}: {e}")

        if self.embedded_data_clip:
            if self.embedding_dimension_clip and len(self.embedded_data_clip[0]['embedding']) != self.embedding_dimension_clip:
                self.embedding_dimension_clip = len(self.embedded_data_clip[0]['embedding'])

            self.progress_callback(f"Successfully generated {len(self.embedded_data_clip)} CLIP image embeddings.")
        else:
            self.progress_callback("No CLIP image embeddings were generated.")
            self.embedding_dimension_clip = None

    def _setup_qdrant(self):
        if not self.embedding_dimension_clip:
            self.progress_callback("Embedding dimension not determined. Skipping Qdrant setup.")
            return

        self.progress_callback("Connecting to Qdrant...")
        try:
            try:
                self.qdrant_client = QdrantClient(host="qdrant", port=6333, timeout=5)
                self.qdrant_client.get_collections()
                self.progress_callback("Connected to Qdrant using Docker alias.")
            except Exception:
                self.qdrant_client = QdrantClient(host="localhost", port=6333, timeout=5)
                self.qdrant_client.get_collections()
                self.progress_callback("Connected to Qdrant on localhost.")
        except Exception as e:
            self.progress_callback(f"Error connecting to Qdrant: {e}. Ensure it's running locally.")
            self.qdrant_client = None
            return

        if self.qdrant_client:
            try:
                collections_response = self.qdrant_client.get_collections()
                collection_names = [col.name for col in collections_response.collections]

                if self.collection_name in collection_names:
                    self.progress_callback(f"Collection '{self.collection_name}' exists. Recreating.")
                    self.qdrant_client.delete_collection(collection_name=self.collection_name)
                    time.sleep(1)

                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dimension_clip,
                        distance=models.Distance.COSINE
                    )
                )
                self.progress_callback(f"Collection '{self.collection_name}' created successfully.")
            except Exception as e:
                self.progress_callback(f"Error during Qdrant collection setup: {e}")
                self.qdrant_client = None

    def _ingest_data(self):
        BATCH_SIZE = 64
        if not self.qdrant_client or not self.embedded_data_clip:
            self.progress_callback("Skipping ingestion: Qdrant client not connected or no embeddings available.")
            return

        self.progress_callback(f"Ingesting {len(self.embedded_data_clip)} data points into Qdrant...")
        total_ingested = 0
        num_batches = (len(self.embedded_data_clip) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in tqdm(range(0, len(self.embedded_data_clip), BATCH_SIZE), desc="Ingesting Batches", total=num_batches):
            batch = self.embedded_data_clip[i : i + BATCH_SIZE]
            points_to_upsert = []
            for item in batch:
                if 'embedding' in item and isinstance(item['embedding'], list):
                    points_to_upsert.append(
                        models.PointStruct(
                            id=item['id'],
                            vector=item['embedding'],
                            payload={
                                "text": item['text'],
                                "page_num": item['page_num'],
                                "image_b64": item['image_b64']
                            }
                        )
                    )
                else:
                    self.progress_callback(f"Skipping ingestion for item missing embedding.")

            if points_to_upsert:
                try:
                    self.qdrant_client.upsert(collection_name=self.collection_name, points=points_to_upsert, wait=True)
                    total_ingested += len(points_to_upsert)
                except Exception as e:
                    self.progress_callback(f"Error upserting batch to Qdrant: {e}")

        self.progress_callback(f"Ingestion complete. Total points ingested: {total_ingested}")
        try:
            count = self.qdrant_client.count(collection_name=self.collection_name, exact=True)
            self.progress_callback(f"Verification: Qdrant reports {count.count} points in the collection.")
        except Exception as e:
            self.progress_callback(f"Could not verify count in Qdrant: {e}")

    def _get_clip_text_embedding(self, text_query):
        if not self.clip_model or not self.clip_processor:
            self.progress_callback("CLIP model or processor not loaded. Cannot generate text embedding.")
            return None
        try:
            inputs = self.clip_processor(text=[text_query], return_tensors="pt", padding=True).to(self.DEVICE)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features[0].cpu().float().numpy().tolist()
        except Exception as e:
            self.progress_callback(f"Error generating text query embedding: {e}")
            return None

    def _retrieve_context(self, query_embedding, top_k=3):
        if not self.qdrant_client or not query_embedding:
            self.progress_callback("Qdrant client not initialized or query embedding is missing.")
            return []

        try:
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True
            ).points
            return search_result
        except Exception as e:
            self.progress_callback(f"Error during Qdrant search: {e}")
            return []

    def _prepare_and_generate(self, query, retrieved_results):
        if not retrieved_results:
            yield "No relevant information found to generate a response."
            return

        os.makedirs("temp_images", exist_ok=True)
        prompt = f"I have a question about a document: {query}\n\nHere are relevant parts of the document to help you answer:\n\n"
        base64_images = []
        
        for i, result in enumerate(retrieved_results):
            if not result.payload:
                continue
                
            context_payload = result.payload
            context_text_content = context_payload.get('text', '')
            context_page = context_payload.get('page_num', 'N/A')
            relevance_score = result.score
            
            prompt += f"--- Document Page {context_page} (Relevance Score: {relevance_score:.4f}) ---\n"
            if context_text_content:
                prompt += f"Text: {context_text_content}\n\n"
            
            if 'image_b64' in context_payload and context_payload['image_b64']:
                img_data = base64.b64decode(context_payload['image_b64'])
                img_path = f"temp_images/page_{context_page}_{i}.jpg"
                with open(img_path, "wb") as f:
                    f.write(img_data)
                base64_images.append(context_payload['image_b64'])
        
        prompt += f"\nPlease answer my question using both the text and visual information from the document."
        
        try:
            generate_payload = {
                "model": self.ollama_model_name,
                "prompt": prompt,
                "stream": True
            }
            
            if base64_images:
                generate_payload["images"] = base64_images
                
            response = requests.post(
                f"{self.ollama_api_base}/api/generate",
                json=generate_payload,
                stream=True
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            if 'response' in json_response:
                                yield json_response['response']

                            if json_response.get('done', False):
                                break
                        except json.JSONDecodeError:
                            pass
                yield "\n"
            else:
                yield f"Error: Ollama API returned status {response.status_code} - {response.text}"

        except requests.exceptions.ConnectionError:
            yield f"Error: Could not connect to Ollama at {self.ollama_api_base}. Is Ollama running?"
        except Exception as e:
            yield f"Error during generation: {e}"

    def query(self, query_text):
        if os.path.exists("temp_images"):
            try:
                for file in os.listdir("temp_images"):
                    file_path = os.path.join("temp_images", file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
            except Exception as e:
                print(f"Error clearing temp_images: {e}")
        
        os.makedirs("temp_images", exist_ok=True)
        query_embedding = self._get_clip_text_embedding(query_text)

        if not query_embedding:
            return iter(["Error: Could not generate query embedding."])

        retrieved_results = self._retrieve_context(query_embedding)
        return self._prepare_and_generate(query_text, retrieved_results)