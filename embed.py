import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("embedding_generator")

class EmbeddingGenerator:
    def __init__(self, 
                 input_file: str = "processed_data.json",
                 collection_name: str = "uganda_medical",
                 embedding_model: str = "text-embedding-3-small"):
        """Initialize embedding generator"""
        self.input_file = input_file
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dim = 1536  # Dimension for text-embedding-3-small
        
        # Initialize OpenAI client
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize Qdrant Cloud client
        self.qdrant_url = os.environ.get("QDRANT_URL")
        self.qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
        
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        logger.info(f"Connected to Qdrant Cloud at {self.qdrant_url}")
        
    def create_collection(self) -> None:
        """Create or recreate the Qdrant collection"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection {self.collection_name} already exists. Recreating it...")
                self.qdrant_client.delete_collection(self.collection_name)
            
            # Create collection
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20000  # Appropriate for dataset size
                )
            )
            
            logger.info(f"Created collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    def generate_embeddings(self, batch_size: int = 100) -> None:
        """Generate embeddings for processed data and store in Qdrant"""
        try:
            # Load processed data
            if not os.path.exists(self.input_file):
                logger.error(f"Input file {self.input_file} not found")
                return
                
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            logger.info(f"Loaded {len(data)} chunks from {self.input_file}")
            
            # Create collection
            self.create_collection()
            
            # Process in batches
            total_batches = len(data) // batch_size + (1 if len(data) % batch_size > 0 else 0)
            
            for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(data))
                
                current_batch = data[start_idx:end_idx]
                
                # Extract content for embedding
                texts = [item["content"] for item in current_batch]
                
                try:
                    # Generate embeddings with OpenAI
                    response = self.openai_client.embeddings.create(
                        input=texts,
                        model=self.embedding_model
                    )
                    
                    embeddings = [embedding.embedding for embedding in response.data]
                    
                    # Prepare points for Qdrant
                    points = []
                    for i, embedding in enumerate(embeddings):
                        # Create a payload with content and metadata
                        payload = {
                            "content": current_batch[i]["content"],
                            **current_batch[i]["metadata"]
                        }
                        
                        points.append(models.PointStruct(
                            id=start_idx + i,
                            vector=embedding,
                            payload=payload
                        ))
                    
                    # Upsert to Qdrant
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    
                    logger.info(f"Processed batch {batch_idx+1}/{total_batches}: {len(points)} embeddings")
                    
                    # Avoid hitting rate limits
                    if batch_idx < total_batches - 1:
                        time.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx+1}: {str(e)}")
                    # Continue with next batch
                    continue
            
            logger.info(f"Successfully embedded and stored {len(data)} chunks in collection {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        logger.info("Starting embedding generation process")
        
        # Get input file as command-line argument or use default
        import sys
        input_file = sys.argv[1] if len(sys.argv) > 1 else "processed_data.json"
        
        generator = EmbeddingGenerator(input_file=input_file)
        generator.generate_embeddings()
        
        logger.info("Embedding generation process completed")
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")