"""Reference-image search over the per-image product index, returning distinct products.

Ranking method (documented in README):
  1. Query each selected vector space (clip_image / dino_image) with the same payload filter.
  2. Fuse per image:  score = sum(w_s * cos_s) / sum(w_s). An image missing from one space's
     candidate list gets that list's lowest returned score (not 0) so it is not unfairly penalised.
  3. Group by ItemID; a product's score is its best image's fused score (max aggregation).
Scores are cosine similarities in [-1, 1]; they are NOT calibrated probabilities.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from PIL import Image
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from app import config
from app.sync import index

logger = logging.getLogger(__name__)

MODES = ("fusion", "clip", "dino")


@dataclass
class SearchFilters:
    category: str | None = None
    plain_gold: bool | None = None
    solitaire: bool | None = None
    valid: bool | None = None
    franchise: bool | None = None
    item_codes: list[str] = field(default_factory=list)


def build_filter(f: SearchFilters) -> Filter:
    must = [FieldCondition(key="is_active", match=MatchValue(value=True))]
    if f.category:
        must.append(FieldCondition(key="category_norm", match=MatchValue(value=f.category.strip().lower())))
    for key, val in (("IsPlainGold", f.plain_gold), ("IsSolitaire", f.solitaire),
                     ("IsValid", f.valid), ("IsFranchiseItem", f.franchise)):
        if val is not None:
            must.append(FieldCondition(key=key, match=MatchValue(value=val)))
    if f.item_codes:
        must.append(FieldCondition(key="item_code_norm",
                                   match=MatchAny(any=[c.strip().upper() for c in f.item_codes if c.strip()])))
    return Filter(must=must)


def _weights(mode: str, clip_w: float | None, dino_w: float | None) -> dict[str, float]:
    if mode == "clip":
        return {index.CLIP_VEC: 1.0}
    if mode == "dino":
        return {index.DINO_VEC: 1.0}
    w = {index.CLIP_VEC: config.SEARCH_WEIGHT_CLIP if clip_w is None else clip_w,
         index.DINO_VEC: config.SEARCH_WEIGHT_DINO if dino_w is None else dino_w}
    w = {k: v for k, v in w.items() if v > 0}
    if not w:
        raise ValueError("at least one fusion weight must be > 0")
    return w


def _image_view(payload: dict, score: float | None = None) -> dict:
    out = {k: payload.get(k) for k in ("ImgID", "ImageURL", "ImgViewID")}
    if score is not None:
        out["score"] = round(score, 4)
    return out


class ProductSearch:
    def __init__(self, embedder, client=None):
        self.embedder = embedder
        self.client = client or index.client()

    def _query(self, using: str, vec: list, limit: int, flt: Filter) -> list:
        return self.client.query_points(
            collection_name=config.PRODUCT_COLLECTION, query=vec, using=using, limit=limit,
            query_filter=flt, with_payload=True,
        ).points

    def _other_images(self, item_ids: list[int]) -> dict[int, list[dict]]:
        if not item_ids:
            return {}
        flt = Filter(must=[FieldCondition(key="ItemID", match=MatchAny(any=item_ids)),
                           FieldCondition(key="is_active", match=MatchValue(value=True))])
        out: dict[int, list[dict]] = {i: [] for i in item_ids}
        offset = None
        while True:
            pts, offset = self.client.scroll(config.PRODUCT_COLLECTION, scroll_filter=flt, limit=256,
                                             with_payload=True, offset=offset)
            for pt in pts:
                out[pt.payload["ItemID"]].append(_image_view(pt.payload))
            if offset is None:
                break
        for imgs in out.values():
            imgs.sort(key=lambda i: i["ImgID"])
        return out

    async def search(self, image: Image.Image, top_k: int, mode: str = "fusion",
                     filters: SearchFilters | None = None, min_score: float | None = None,
                     clip_weight: float | None = None, dino_weight: float | None = None) -> dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        weights = _weights(mode, clip_weight, dino_weight)
        flt = build_filter(filters or SearchFilters())
        fetch = min(max(top_k * 10, 50), 500)  # several views per product -> over-fetch images

        def embed() -> dict[str, list]:
            e = self.embedder
            vecs = {}
            if index.CLIP_VEC in weights:
                vecs[index.CLIP_VEC] = e.normalize(e.embed_clip_image(image)).tolist()
            if index.DINO_VEC in weights:
                vecs[index.DINO_VEC] = e.normalize(e.embed_dino_image(image)).tolist()
            return vecs

        vecs = await asyncio.to_thread(embed)
        results = await asyncio.gather(*(
            asyncio.to_thread(self._query, using, vec, fetch, flt) for using, vec in vecs.items()))

        floor = {u: (min(p.score for p in pts) if pts else 0.0) for u, pts in zip(vecs, results)}
        total_w = sum(weights.values())
        fused: dict[int, dict] = {}
        for using, pts in zip(vecs, results):
            for p in pts:
                e = fused.setdefault(p.payload["ImgID"], {"payload": p.payload, "per": {}})
                e["per"][using] = p.score
        for e in fused.values():
            e["score"] = sum(w * e["per"].get(u, floor[u]) for u, w in weights.items()) / total_w

        by_item: dict[int, dict] = {}
        for e in sorted(fused.values(), key=lambda x: x["score"], reverse=True):
            if min_score is not None and e["score"] < min_score:
                break
            iid = e["payload"]["ItemID"]
            if iid not in by_item:
                by_item[iid] = {"best": e, "matched": 0}
            by_item[iid]["matched"] += 1
        ranked = sorted(by_item.items(), key=lambda kv: kv[1]["best"]["score"], reverse=True)[:top_k]
        others = await asyncio.to_thread(self._other_images, [iid for iid, _ in ranked])

        results_out = []
        for rank, (iid, v) in enumerate(ranked, 1):
            p = v["best"]["payload"]
            results_out.append({
                "rank": rank, "item_id": iid, "item_code": p.get("ItemCode"),
                "category_code": p.get("CategoryCode"),
                "is_plain_gold": p.get("IsPlainGold"), "is_solitaire": p.get("IsSolitaire"),
                "is_valid": p.get("IsValid"), "is_franchise_item": p.get("IsFranchiseItem"),
                "status_remark": p.get("StatusRemark"),
                "score": round(v["best"]["score"], 4),
                "score_per_space": {k: round(s, 4) for k, s in v["best"]["per"].items()},
                "matched_images": v["matched"],
                "best_image": _image_view(p, v["best"]["score"]),
                "images": others.get(iid, []),
            })
        return {
            "score_type": "cosine similarity (" + ("+".join(weights) if len(weights) > 1 else next(iter(weights)))
                          + (", weighted mean" if len(weights) > 1 else "") + "); not a calibrated probability",
            "mode": mode, "weights": weights, "embedding_model_version": config.EMBEDDING_MODEL_VERSION,
            "total_returned": len(results_out), "results": results_out,
        }
