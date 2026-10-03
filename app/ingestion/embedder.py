# embedder.py — CLIP + DINOv2 embedding generator.

import logging
import time
import numpy as np
import open_clip
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from app.config import CLIP_MODEL_NAME, CLIP_PRETRAINED, DINO_MODEL_NAME

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self,
        clip_model_name: str = CLIP_MODEL_NAME,
        clip_pretrained: str = CLIP_PRETRAINED,
        dino_model_name: str = DINO_MODEL_NAME,
        device: str | None   = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Embedder initialising on device: %s", self.device)

        # CLIP 
        t0 = time.monotonic()
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.clip_model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        logger.info(
            "CLIP model loaded (%s / %s) in %.1fs",
            clip_model_name, clip_pretrained, time.monotonic() - t0,
        )

        # DINOv2 
        t0 = time.monotonic()
        self.dino_model     = AutoModel.from_pretrained(dino_model_name).to(self.device).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
        logger.info(
            "DINOv2 model loaded (%s) in %.1fs", dino_model_name, time.monotonic() - t0
        )

    # Individual encoders

    def embed_clip_image(self, image: Image.Image) -> np.ndarray:
        t0  = time.monotonic()
        inp = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(inp)
        logger.debug("CLIP image embed in %.0fms", (time.monotonic() - t0) * 1000)
        return emb.cpu().numpy()[0]

    def embed_clip_text(self, text: str) -> np.ndarray:
        t0     = time.monotonic()
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens)
        logger.debug("CLIP text embed in %.0fms", (time.monotonic() - t0) * 1000)
        return emb.cpu().numpy()[0]

    def embed_dino_image(self, image: Image.Image) -> np.ndarray:
        t0     = time.monotonic()
        inputs = self.dino_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        emb = outputs.last_hidden_state[:, 0, :]
        logger.debug("DINOv2 embed in %.0fms", (time.monotonic() - t0) * 1000)
        return emb.cpu().numpy()[0]

    # Utilities 

    @staticmethod
    def normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            logger.warning("Zero-norm vector detected — returning as-is")
            return vec
        return vec / norm

    def generate_embeddings(self, image_path: str, caption: str) -> dict:
        """Load image from disk and return normalised embeddings."""
        logger.debug("Generating embeddings for: %s", image_path)
        image = Image.open(image_path).convert("RGB")
        return self.generate_embeddings_from_pil(image, caption)

    def generate_embeddings_from_pil(self, image: Image.Image, caption: str) -> dict:
        """Accept an already-loaded PIL image and return normalised embeddings."""
        return {
            "clip_image": self.normalize(self.embed_clip_image(image)),
            "clip_text":  self.normalize(self.embed_clip_text(caption)),
            "dino_image": self.normalize(self.embed_dino_image(image)),
        }