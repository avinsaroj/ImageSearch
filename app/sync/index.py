"""Qdrant side of the product-image index.

Schema (one point per product image):
  vectors  : clip_image (CLIP_EMBED_DIM, cosine), dino_image (DINO_EMBED_DIM, cosine) - L2-normalised
  point id : uuid5("kisna-img:<ImgID>")  -> upserts are idempotent
  payload  : ItemID, ItemCode, CategoryCode, category_norm, IsPlainGold, IsSolitaire, IsValid,
             IsFranchiseItem (normalised bools; *_raw keeps the SQL value), StatusRemark, CreatedOn,
             UpdatedOn, ImgID, ImageURL, ImgViewID, ContentHash, EmbeddingModelVersion,
             LastProcessedAt, is_active

PRODUCT_COLLECTION is an *alias*; real collections are "<alias>-<stamp>". A rebuild fills a new
collection and swaps the alias only after validation, so search keeps serving the old index.
"""

import logging
import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias, CreateAliasOperation, DeleteAlias, DeleteAliasOperation, Distance, FieldCondition,
    Filter, FilterSelector, HasIdCondition, MatchValue, PayloadSchemaType, PointStruct, ScalarQuantization,
    ScalarQuantizationConfig, ScalarType, VectorParams,
)

from app import config

logger = logging.getLogger(__name__)
_NS = uuid.UUID("6f1c1d0e-5a0b-4c1e-9d55-0a1b2c3d4e5f")

CLIP_VEC, DINO_VEC = "clip_image", "dino_image"

_INDEXES = {
    "ItemID": PayloadSchemaType.INTEGER,
    "ImgID": PayloadSchemaType.INTEGER,
    "category_norm": PayloadSchemaType.KEYWORD,
    "item_code_norm": PayloadSchemaType.KEYWORD,
    "IsPlainGold": PayloadSchemaType.BOOL,
    "IsSolitaire": PayloadSchemaType.BOOL,
    "IsValid": PayloadSchemaType.BOOL,
    "IsFranchiseItem": PayloadSchemaType.BOOL,
    "is_active": PayloadSchemaType.BOOL,
}


def client() -> QdrantClient:
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=60)


def point_id(img_id: int) -> str:
    return str(uuid.uuid5(_NS, f"kisna-img:{img_id}"))


def to_bool(v):
    """Normalise SQL flag values (bit/int/char) to bool; None stays None."""
    if v is None:
        return None
    if isinstance(v, (bool, int)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "y", "yes", "true", "t"):
        return True
    if s in ("0", "n", "no", "false", "f"):
        return False
    return None  # blank / unknown stays unknown rather than being guessed as False


def product_payload(p: dict) -> dict:
    """Product-level payload fields, shared by every image of the product."""
    def iso(x):
        return x.isoformat() if hasattr(x, "isoformat") else x
    code = p.get("ItemCode")
    cat = p.get("CategoryCode")
    return {
        "ItemID": p["ItemID"],
        "ItemCode": code,
        "item_code_norm": str(code).strip().upper() if code else None,
        "CategoryCode": cat,
        "category_norm": str(cat).strip().lower() if cat else None,
        "IsPlainGold": to_bool(p.get("IsPlainGold")), "IsPlainGold_raw": p.get("IsPlainGold"),
        "IsSolitaire": to_bool(p.get("IsSolitaire")), "IsSolitaire_raw": p.get("IsSolitaire"),
        "IsValid": to_bool(p.get("IsValid")), "IsValid_raw": p.get("IsValid"),
        "IsFranchiseItem": to_bool(p.get("IsFranchiseItem")), "IsFranchiseItem_raw": p.get("IsFranchiseItem"),
        "StatusRemark": p.get("StatusRemark"),
        "CreatedOn": iso(p.get("CreatedOn")), "UpdatedOn": iso(p.get("UpdatedOn")),
    }


def make_point(img: dict, product: dict, vectors: dict, content_hash: str) -> PointStruct:
    payload = product_payload(product)
    payload.update({
        "ImgID": img["ImgID"], "ImageURL": img["ImageURL"], "ImgViewID": img.get("ImgViewID"),
        "ContentHash": content_hash, "EmbeddingModelVersion": config.EMBEDDING_MODEL_VERSION,
        "LastProcessedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_active": True,
    })
    return PointStruct(
        id=point_id(img["ImgID"]),
        vector={CLIP_VEC: vectors[CLIP_VEC].tolist(), DINO_VEC: vectors[DINO_VEC].tolist()},
        payload=payload,
    )


# collections / alias
def resolve_alias(c: QdrantClient) -> str | None:
    for a in c.get_aliases().aliases:
        if a.alias_name == config.PRODUCT_COLLECTION:
            return a.collection_name
    return None


def create_collection(c: QdrantClient, name: str) -> None:
    c.create_collection(
        collection_name=name,
        # Millions of images: keep full-precision vectors on disk and an int8 copy in RAM for fast search.
        vectors_config={
            CLIP_VEC: VectorParams(size=config.CLIP_EMBED_DIM, distance=Distance.COSINE, on_disk=True),
            DINO_VEC: VectorParams(size=config.DINO_EMBED_DIM, distance=Distance.COSINE, on_disk=True),
        },
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True)),
    )
    for field, schema in _INDEXES.items():
        c.create_payload_index(name, field_name=field, field_schema=schema)
    logger.info("Created collection %s", name)


def new_collection_name() -> str:
    return f"{config.PRODUCT_COLLECTION}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"


def ensure_active_collection(c: QdrantClient) -> str:
    """Return the real collection behind the alias, creating collection + alias on first use."""
    real = resolve_alias(c)
    if real:
        return real
    real = new_collection_name()
    create_collection(c, real)
    switch_alias(c, real)
    return real


def switch_alias(c: QdrantClient, real: str) -> None:
    ops = []
    if resolve_alias(c):
        ops.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=config.PRODUCT_COLLECTION)))
    ops.append(CreateAliasOperation(create_alias=CreateAlias(
        collection_name=real, alias_name=config.PRODUCT_COLLECTION)))
    c.update_collection_aliases(change_aliases_operations=ops)  # applied atomically
    logger.info("Alias %s -> %s", config.PRODUCT_COLLECTION, real)


def count_points(c: QdrantClient, collection: str) -> int:
    return c.count(collection, exact=True).count


# writes
def upsert(c: QdrantClient, collection: str, points: list[PointStruct]) -> None:
    if points:
        c.upsert(collection_name=collection, points=points, wait=True)


def _item_filter(item_id: int) -> Filter:
    return Filter(must=[FieldCondition(key="ItemID", match=MatchValue(value=item_id))])


def update_product_payload(c: QdrantClient, collection: str, product: dict) -> None:
    """Refresh product-level metadata on all of a product's image points without re-embedding."""
    c.set_payload(collection_name=collection, payload=product_payload(product),
                  points=_item_filter(product["ItemID"]), wait=True)


def deactivate_images(c: QdrantClient, collection: str, img_ids: list[int]) -> None:
    # A has-id filter (unlike an explicit id list) does not fail when some points are missing.
    flt = Filter(must=[HasIdCondition(has_id=[point_id(i) for i in img_ids])])
    c.set_payload(collection_name=collection, payload={"is_active": False}, points=flt, wait=True)


def delete_images(c: QdrantClient, collection: str, img_ids: list[int]) -> None:
    c.delete(collection_name=collection, points_selector=[point_id(i) for i in img_ids], wait=True)


def indexed_items(c: QdrantClient, collection: str) -> dict[int, tuple[str | None, bool | None]]:
    """{ItemID: (normalised ItemCode, IsValid)} for every product that has points in the collection."""
    out: dict[int, tuple[str | None, bool | None]] = {}
    offset = None
    while True:
        pts, offset = c.scroll(collection, limit=5000, offset=offset, with_vectors=False,
                               with_payload=["ItemID", "ItemCode", "item_code_norm", "IsValid"])
        for pt in pts:
            pl = pt.payload or {}
            code = pl.get("item_code_norm") or (str(pl["ItemCode"]).strip().upper() if pl.get("ItemCode") else None)
            out[int(pl["ItemID"])] = (code, pl.get("IsValid"))
        if offset is None:
            return out


def delete_item(c: QdrantClient, collection: str, item_id: int) -> None:
    c.delete(collection_name=collection, points_selector=FilterSelector(filter=_item_filter(item_id)), wait=True)
