# embedding_function.py

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import gc
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)

DEVICE = torch.device("cpu")
torch.set_num_threads(1)
logger.info(f"Using device: {DEVICE}")

class ImageEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        try:
            logger.info(f"Loading image model: {model_name}")
            
            self.processor = CLIPProcessor.from_pretrained(
                model_name, 
                local_files_only=False
            )
            logger.info("Processor loaded successfully")
       
            self.model = CLIPModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                local_files_only=False
            ).to(DEVICE)
            self.model.eval()
            logger.info("Image model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading image model: {e}")
            raise

    def extract(self, image_path):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            
            result = features.cpu().squeeze().tolist()
            
            del inputs, features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting image features from {image_path}: {e}")
            raise


class TextEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        try:
            logger.info(f"Loading text model: {model_name}")
            
            self.processor = CLIPProcessor.from_pretrained(
                model_name,
                local_files_only=False
            )
            logger.info("Processor loaded successfully")
            
            self.model = CLIPModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                local_files_only=False
            ).to(DEVICE)
            self.model.eval()
            logger.info("Text model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading text model: {e}")
            raise

    def extract(self, text):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        
            result = text_features.cpu().squeeze().tolist()
            
            del inputs, text_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text features from '{text}': {e}")
            raise