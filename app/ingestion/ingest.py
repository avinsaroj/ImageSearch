# ingest.py — Batch ingestion pipeline.

import json
import logging
import os
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from app.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    CLIP_EMBED_DIM,
    DINO_EMBED_DIM,
    INGEST_BATCH_SIZE,
    LOG_LEVEL,
)
from app.ingestion.embedder import Embedder

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class Ingestor:
    def __init__(self):
        logger.info("Connecting to Qdrant at %s:%s", QDRANT_HOST, QDRANT_PORT)
        self.client   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.embedder = Embedder()

    def create_collection_if_missing(self):

        # Create the Qdrant collection only if it does not already exist.
        
        existing_names = [c.name for c in self.client.get_collections().collections]

        if COLLECTION_NAME in existing_names:
            logger.info(
                "Collection '%s' already exists — skipping creation.", COLLECTION_NAME
            )
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "clip_image": VectorParams(size=CLIP_EMBED_DIM, distance=Distance.COSINE),
                "clip_text":  VectorParams(size=CLIP_EMBED_DIM, distance=Distance.COSINE),
                "dino_image": VectorParams(size=DINO_EMBED_DIM, distance=Distance.COSINE),
            },
        )
        logger.info("Collection '%s' created.", COLLECTION_NAME)

    def load_metadata(self, path: str) -> list:
        with open(path, "r") as f:
            return json.load(f)

    def ingest(self, metadata_path: str, batch_size: int = INGEST_BATCH_SIZE):
        
        # Load metadata, generate embeddings, and upsert into Qdrant.
        
        data = self.load_metadata(metadata_path)
        logger.info("Loaded %d items from %s", len(data), metadata_path)

        points  = []
        skipped = 0

        for idx, item in enumerate(tqdm(data, desc="Ingesting")):
            image_path = item["filepath"].replace("\\", "/")
            caption    = item["caption"]

            if not os.path.exists(image_path):
                logger.warning("Missing file, skipping: %s", image_path)
                skipped += 1
                continue

            try:
                vectors = self.embedder.generate_embeddings(image_path, caption)
            except Exception as exc:
                logger.error("Embedding failed for %s: %s — skipping", image_path, exc)
                skipped += 1
                continue

            payload = {
                "filename": item["filename"],
                "filepath": image_path,
                "category": item["category"],
                "metal":    item["metal"],
                "caption":  item["caption"],
            }

            points.append(
                PointStruct(
                    id=idx,
                    vector={
                        "clip_image": vectors["clip_image"].tolist(),
                        "clip_text":  vectors["clip_text"].tolist(),
                        "dino_image": vectors["dino_image"].tolist(),
                    },
                    payload=payload,
                )
            )

            if len(points) >= batch_size:
                self.client.upsert(collection_name=COLLECTION_NAME, points=points)
                logger.info("Upserted batch ending at index %d", idx)
                points = []

        # Flush remaining points
        if points:
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info("Upserted final batch of %d points.", len(points))

        logger.info(
            "Ingestion complete. Indexed: %d  Skipped: %d",
            len(data) - skipped,
            skipped,
        )

if __name__ == "__main__":
    ingestor = Ingestor()
    ingestor.create_collection_if_missing()   # safe on re-run
    ingestor.ingest("data/metadata_cleaned.json")