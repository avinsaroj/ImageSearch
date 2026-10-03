import asyncio
import logging

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
)
from app.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)

# Qdrant named-vector keys — must match what was used during ingestion
CLIP_IMAGE_VEC = "clip_image"
CLIP_TEXT_VEC  = "clip_text"
DINO_IMAGE_VEC = "dino_image"


class SearchService:
    def __init__(self):
        logger.info("Connecting to Qdrant at %s:%s", QDRANT_HOST, QDRANT_PORT)
        self.client   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.embedder = Embedder()

    # Low-level helpers 

    def _query(self, using: str, vector: list, limit: int, category: str = None) -> list:
        """Run a single named-vector search against Qdrant (blocking)."""
        query_filter = None
        if category:
            query_filter = Filter(
                must=[FieldCondition(
                    key="category",
                    match=MatchValue(value=category),
                )]
            )
        return self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            using=using,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points

    @staticmethod
    def _fuse(weighted_batches: list[tuple[list, float]]) -> list:
        
        total_weight = sum(w for _, w in weighted_batches)

        scores: dict = {}
        for results, weight in weighted_batches:
            for r in results:
                entry = scores.setdefault(r.id, {"score": 0.0, "payload": r.payload})
                entry["score"] += (weight / total_weight) * r.score
                if r.payload:
                    entry["payload"] = r.payload
        return [{"id": k, **v} for k, v in scores.items()]

    @staticmethod
    def _format(scored_items: list, top_k: int) -> list:
        """Sort fused results and return top_k clean dicts."""
        ranked = sorted(scored_items, key=lambda x: x["score"], reverse=True)
        return [
            {
                "rank":    i + 1,
                "id":      item["id"],
                "score":   round(item["score"], 4),
                "payload": item["payload"],
            }
            for i, item in enumerate(ranked[:top_k])
        ]

    # Async public API

    async def search_by_text(self, query: str, top_k: int = 10, category: str = None) -> list:
        
        fetch_k = min(top_k * 3, 200)

        # Embed the query as-is — query_cleaner is disconnected
        text_vec = self.embedder.normalize(
            self.embedder.embed_clip_text(query)
        ).tolist()

        # Both Qdrant queries run concurrently
        clip_img_res, clip_txt_res = await asyncio.gather(
            asyncio.to_thread(self._query, CLIP_IMAGE_VEC, text_vec, fetch_k, category),
            asyncio.to_thread(self._query, CLIP_TEXT_VEC,  text_vec, fetch_k, category),
        )

        # Caption space is more aligned with free-text queries → higher weight
        items = self._fuse([
            (clip_img_res, 0.35),
            (clip_txt_res, 0.65),
        ])
        logger.info("Text search for %r returned %d candidates", query, len(items))
        return self._format(items, top_k)

    async def search_by_image_raw(self, image: Image.Image, top_k: int = 10, category: str = None) -> list:
        
        fetch_k = min(top_k * 3, 200)

        clip_vec = self.embedder.normalize(
            self.embedder.embed_clip_image(image)
        ).tolist()
        dino_vec = self.embedder.normalize(
            self.embedder.embed_dino_image(image)
        ).tolist()

        clip_results, dino_results = await asyncio.gather(
            asyncio.to_thread(self._query, CLIP_IMAGE_VEC, clip_vec, fetch_k, category),
            asyncio.to_thread(self._query, DINO_IMAGE_VEC, dino_vec, fetch_k, category),
        )

        items = self._fuse([
            (clip_results, 0.35),
            (dino_results, 0.65),
        ])
        logger.info("Image search returned %d candidates", len(items))
        return self._format(items, top_k)

    async def search_image_text(
        self,
        image: Image.Image,
        text: str,
        top_k: int = 10,
        category: str = None,
    ) -> list:

        fetch_k = min(top_k * 3, 200)

        # Embed text as-is — query_cleaner is disconnected
        clip_img_vec = self.embedder.normalize(
            self.embedder.embed_clip_image(image)
        ).tolist()
        dino_img_vec = self.embedder.normalize(
            self.embedder.embed_dino_image(image)
        ).tolist()
        clip_txt_vec = self.embedder.normalize(
            self.embedder.embed_clip_text(text)
        ).tolist()

        # All three Qdrant queries fire at the same time
        clip_img_res, dino_img_res, clip_txt_res = await asyncio.gather(
            asyncio.to_thread(self._query, CLIP_IMAGE_VEC, clip_img_vec, fetch_k, category),
            asyncio.to_thread(self._query, DINO_IMAGE_VEC, dino_img_vec, fetch_k, category),
            asyncio.to_thread(self._query, CLIP_TEXT_VEC,  clip_txt_vec, fetch_k, category),
        )

        items = self._fuse([
            (clip_img_res, 0.35),
            (dino_img_res, 0.35),
            (clip_txt_res, 0.30),
        ])
        logger.info(
            "Hybrid search for %r returned %d candidates", text, len(items)
        )
        return self._format(items, top_k)

    # Health helper (used by /health endpoint)

    def qdrant_is_healthy(self) -> bool:
        
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            return COLLECTION_NAME in existing
        except Exception as exc:
            logger.warning("Qdrant health check failed: %s", exc)
            return False