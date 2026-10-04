import logging
import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Path, Query, UploadFile

from app.api.routes.search import _decode_image, _service
from app.config import DEFAULT_TOP_K, MAX_TOP_K
from app.db import sqlserver as db
from app.services.product_search import MODES, ProductSearch, SearchFilters
from app.sync import index, state

router = APIRouter()
logger = logging.getLogger(__name__)

_product_search: ProductSearch | None = None
_categories_cache: tuple[float, list[str]] | None = None
CATEGORY_TTL_S = 300


def _search() -> ProductSearch:
    global _product_search
    if _product_search is None:
        _product_search = ProductSearch(_service.embedder)
    return _product_search


def _tri(value: Optional[str], name: str) -> Optional[bool]:
    """Form helper for 'all' / 'yes' / 'no' selectors."""
    if value is None or value.strip().lower() in ("", "all"):
        return None
    v = value.strip().lower()
    if v in ("yes", "true", "1"):
        return True
    if v in ("no", "false", "0"):
        return False
    raise HTTPException(status_code=422, detail=f"{name} must be one of: all, yes, no")


@router.get("/categories", tags=["Catalog"])
def categories():
    """Categories loaded dynamically from SQL Server (cached for 5 minutes)."""
    global _categories_cache
    if _categories_cache and time.monotonic() - _categories_cache[0] < CATEGORY_TTL_S:
        cats = _categories_cache[1]
    else:
        try:
            cats = db.fetch_categories()
        except Exception as exc:
            logger.warning("categories lookup failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="SQL Server unavailable")
        _categories_cache = (time.monotonic(), cats)
    return {"count": len(cats), "categories": [{"code": c, "normalized": c.strip().lower()} for c in cats]}


@router.post("/search/image", tags=["Search"])
async def search_product_image(
    file: UploadFile = File(...),
    top_k: int = Form(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
    mode: str = Form(default="fusion"),
    category: Optional[str] = Form(default=None),
    plain_gold: Optional[str] = Form(default=None),
    solitaire: Optional[str] = Form(default=None),
    valid: Optional[str] = Form(default=None),
    franchise: Optional[str] = Form(default=None),
    item_codes: Optional[str] = Form(default=None, description="comma-separated item codes"),
    min_score: Optional[float] = Form(default=None, ge=-1.0, le=1.0),
    clip_weight: Optional[float] = Form(default=None, ge=0),
    dino_weight: Optional[float] = Form(default=None, ge=0),
):
    """Upload a reference image; returns distinct products ranked by their best matching image."""
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {list(MODES)}")
    pil = _decode_image(file, await file.read())
    filters = SearchFilters(
        category=category or None,
        plain_gold=_tri(plain_gold, "plain_gold"), solitaire=_tri(solitaire, "solitaire"),
        valid=_tri(valid, "valid"), franchise=_tri(franchise, "franchise"),
        item_codes=[c for c in (item_codes or "").split(",") if c.strip()],
    )
    try:
        out = await _search().search(pil, top_k, mode, filters, min_score, clip_weight, dino_weight)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("Product image search failed")
        raise HTTPException(status_code=500, detail="Search failed - see server logs.")
    out["image_filename"] = file.filename
    return out


@router.get("/products/{item_id}", tags=["Catalog"])
def get_product(item_id: int = Path(ge=1)):
    try:
        rows = db.fetch_products_by_ids([item_id])
    except Exception as exc:
        logger.warning("product lookup failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="SQL Server unavailable")
    if not rows:
        raise HTTPException(status_code=404, detail=f"Product {item_id} not found")
    p = rows[0]
    return {**{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in p.items()},
            "normalized": {k: index.to_bool(p.get(k)) for k in
                           ("IsPlainGold", "IsSolitaire", "IsValid", "IsFranchiseItem")}}


@router.post("/search/cross-category", tags=["Search"])
async def search_cross_category(
    file: UploadFile = File(...),
    targets: str = Form(description="comma-separated category codes to search in"),
    per_category: int = Form(default=6, ge=1, le=MAX_TOP_K),
    mode: str = Form(default="fusion"),
    plain_gold: Optional[str] = Form(default=None),
    solitaire: Optional[str] = Form(default=None),
    valid: Optional[str] = Form(default=None),
    franchise: Optional[str] = Form(default=None),
    min_score: Optional[float] = Form(default=None, ge=-1.0, le=1.0),
):
    """Upload a product photo (e.g. a necklace); returns the closest designs in each target category."""
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {list(MODES)}")
    cats = list(dict.fromkeys(c.strip() for c in targets.split(",") if c.strip()))
    if not cats or len(cats) > 20:
        raise HTTPException(status_code=422, detail="targets must list 1-20 category codes")
    pil = _decode_image(file, await file.read())
    filters = SearchFilters(
        plain_gold=_tri(plain_gold, "plain_gold"), solitaire=_tri(solitaire, "solitaire"),
        valid=_tri(valid, "valid"), franchise=_tri(franchise, "franchise"),
    )
    try:
        out = await _search().cross_category(pil, cats, per_category, mode, filters, min_score)
    except Exception:
        logger.exception("Cross-category search failed")
        raise HTTPException(status_code=500, detail="Search failed - see server logs.")
    out["image_filename"] = file.filename
    return out


@router.get("/products/{item_id}/similar", tags=["Search"])
async def similar_products(
    item_id: int = Path(ge=1),
    top_k: int = Query(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
    mode: str = Query(default="fusion"),
    category: Optional[str] = Query(default=None),
    plain_gold: Optional[str] = Query(default=None),
    solitaire: Optional[str] = Query(default=None),
    valid: Optional[str] = Query(default=None),
    franchise: Optional[str] = Query(default=None),
    min_score: Optional[float] = Query(default=None, ge=-1.0, le=1.0),
):
    """Products most similar to an indexed product, using its stored image vectors (the product itself is excluded)."""
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {list(MODES)}")
    filters = SearchFilters(
        category=category or None,
        plain_gold=_tri(plain_gold, "plain_gold"), solitaire=_tri(solitaire, "solitaire"),
        valid=_tri(valid, "valid"), franchise=_tri(franchise, "franchise"),
    )
    try:
        out = await _search().similar(item_id, top_k, mode, filters, min_score)
    except Exception:
        logger.exception("Similar product search failed")
        raise HTTPException(status_code=500, detail="Search failed - see server logs.")
    if out is None:
        raise HTTPException(status_code=404, detail=f"Product {item_id} is not in the search index")
    return out


@router.get("/products/{item_id}/images", tags=["Catalog"])
def get_product_images(item_id: int = Path(ge=1)):
    try:
        imgs = db.fetch_images_for_items([item_id])
    except Exception as exc:
        logger.warning("image lookup failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="SQL Server unavailable")
    out = []
    for r in imgs:
        s = state.get_image(r["ImgID"])
        out.append({"ImgID": r["ImgID"], "ImageURL": r["ImageURL"], "ImgViewID": r["ImgViewID"],
                    "index_status": s["status"] if s else "unknown",
                    "last_error": s["last_error"] if s else None})
    if not out:
        raise HTTPException(status_code=404, detail=f"No images for product {item_id}")
    return {"item_id": item_id, "count": len(out), "images": out}
