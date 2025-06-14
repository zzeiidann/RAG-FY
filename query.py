# query.py

import argparse
import pickle
import faiss
import numpy as np
import gc
import torch
from pathlib import Path
from embedding_function import ImageEmbedder, TextEmbedder
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "faiss_db"
INDEX_FILE = Path(DB_PATH) / "vector_index.faiss"
META_FILE = Path(DB_PATH) / "metadata.pkl"

# Global embedders to avoid reloading models
_text_embedder = None
_image_embedder = None


def get_text_embedder():
    """Get or create text embedder singleton"""
    global _text_embedder
    if _text_embedder is None:
        logger.info("Loading text embedder...")
        _text_embedder = TextEmbedder()
    return _text_embedder


def get_image_embedder():
    """Get or create image embedder singleton"""
    global _image_embedder
    if _image_embedder is None:
        logger.info("Loading image embedder...")
        _image_embedder = ImageEmbedder()
    return _image_embedder


def load_db():
    """Load FAISS index and metadata"""
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError(f"FAISS index or metadata not found in {DB_PATH}. Run Data.py first.")
    
    logger.info("Loading FAISS database...")
    index = faiss.read_index(str(INDEX_FILE))
    
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
    
    logger.info(f"Loaded database with {len(metadata)} entries")
    return index, metadata


def embed_text(query_text):
    """Embed text query"""
    try:
        embedder = get_text_embedder()
        vector = embedder.extract(query_text)
        return np.array([vector]).astype("float32")
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        raise


def embed_image(image_path):
    """Embed image query"""
    try:
        embedder = get_image_embedder()
        vector = embedder.extract(image_path)
        return np.array([vector]).astype("float32")
    except Exception as e:
        logger.error(f"Error embedding image: {e}")
        raise


def search_faiss(query_vector, top_k=5):
    """Search FAISS index"""
    try:
        index, metadata = load_db()
        distances, indices = index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                results.append({
                    "rank": i + 1,
                    "distance": float(distances[0][i]),
                    "metadata": metadata[idx]
                })
        
        return results
    except Exception as e:
        logger.error(f"Error searching FAISS: {e}")
        raise


def search_text(query_text, top_k=5):
    """Search using text query"""
    logger.info(f"Searching for text: '{query_text}'")
    
    # Clear memory before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    try:
        vec = embed_text(query_text)
        results = search_faiss(vec, top_k)
        
        # Clear memory after processing
        del vec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return results
    except Exception as e:
        logger.error(f"Error in text search: {e}")
        raise


def search_image(image_path, top_k=5):
    """Search using image query"""
    logger.info(f"Searching for image: '{image_path}'")
    
    # Clear memory before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    try:
        vec = embed_image(image_path)
        results = search_faiss(vec, top_k)
        
        # Clear memory after processing
        del vec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return results
    except Exception as e:
        logger.error(f"Error in image search: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Search the RAG database")
    parser.add_argument("--text", type=str, help="Query using a text prompt.")
    parser.add_argument("--image", type=str, help="Query using an image path.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    try:
        if args.text:
            logger.info(f"Text search: {args.text}")
            results = search_text(args.text, args.top_k)
        elif args.image:
            logger.info(f"Image search: {args.image}")
            if not Path(args.image).exists():
                logger.error(f"Image file not found: {args.image}")
                return
            results = search_image(args.image, args.top_k)
        else:
            logger.error("Provide either --text or --image argument")
            parser.print_help()
            return

        # Display results
        print(f"\n🔍 Found {len(results)} results:")
        print("=" * 60)
        
        for r in results:
            meta = r["metadata"]
            print(f"\n#{r['rank']} - Distance: {r['distance']:.4f}")
            print(f"📁 File: {meta['filename']}")
            print(f"🔤 Type: {meta['type']}")
            print(f"🧠 Description: {meta['description']}")
            print("-" * 40)
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()