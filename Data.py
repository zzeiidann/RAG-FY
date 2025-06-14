# data.py

import argparse
import os
import json
import shutil
import pickle
import numpy as np
import faiss
import gc
import torch
from pathlib import Path
from embedding_function import ImageEmbedder, TextEmbedder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "faiss_db"
INDEX_FILE = os.path.join(DB_PATH, "vector_index.faiss")
META_FILE = os.path.join(DB_PATH, "metadata.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset FAISS DB.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to folder of images.")
    parser.add_argument("--json", type=str, required=True, help="Path to JSON file mapping image to description.")
    args = parser.parse_args()

    if args.reset:
        reset_faiss_db()

    embed_and_store(args.image_dir, args.json)


def embed_and_store(image_dir, json_path):
    try:
        os.makedirs(DB_PATH, exist_ok=True)

        with open(json_path, "r") as f:
            knowledge = json.load(f)

        logger.info(f"Loaded {len(knowledge)} image-descriptions from {json_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info("Initializing embedders...")
        img_embedder = ImageEmbedder()
        txt_embedder = TextEmbedder()

        vectors = []
        metadata = []

        for i, (fname, description) in enumerate(knowledge.items()):
            logger.info(f"Processing {i+1}/{len(knowledge)}: {fname}")
            
            image_path = Path(image_dir) / fname
            if not image_path.exists():
                logger.warning(f"[SKIPPED] File not found: {image_path}")
                continue

            try:
                if i % 2 == 0:  
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    logger.info(f"Memory cleared at iteration {i}")

                logger.info(f"🔗 Embedding image: {fname}")
                img_vec = img_embedder.extract(str(image_path))
                
                logger.info(f"🔗 Embedding text: {description[:50]}...")
                txt_vec = txt_embedder.extract(description)

                vectors.append(img_vec)
                metadata.append({"type": "image", "filename": fname, "description": description})

                vectors.append(txt_vec)
                metadata.append({"type": "text", "filename": fname, "description": description})

                logger.info(f"✅ Successfully processed: {fname}")
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to process {fname}: {e}")
                continue

        if not vectors:
            logger.error("No embeddings to store. Exiting.")
            return

        logger.info(f"Creating FAISS index with {len(vectors)} vectors...")
        
        dim = len(vectors[0])
        logger.info(f"Vector dimension: {dim}")
        
        index = faiss.IndexFlatL2(dim)
        vectors_array = np.array(vectors).astype("float32")
        index.add(vectors_array)

        logger.info(f"Saving to {INDEX_FILE}")
        faiss.write_index(index, INDEX_FILE)
        
        logger.info(f"Saving metadata to {META_FILE}")
        with open(META_FILE, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"✅ Successfully saved {len(metadata)} embeddings to {DB_PATH}")
        
        # Final memory cleanup
        del vectors, metadata, vectors_array, index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        logger.error(f"Fatal error in embed_and_store: {e}")
        raise


def reset_faiss_db():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        logger.info("🧹 FAISS DB cleared.")
    else:
        logger.info("🧹 FAISS DB directory doesn't exist, nothing to clear.")


if __name__ == "__main__":
    main()