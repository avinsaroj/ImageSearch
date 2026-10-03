import io
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from qdrant_client.models import PointStruct

from app.config import (
    ALLOWED_CONTENT_TYPES,
    MAX_FILE_MB,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    COLLECTION_NAME,
)
from app.ingestion.embedder import Embedder
from app.services.search_service import SearchService

router   = APIRouter()
logger   = logging.getLogger(__name__)

_service  = SearchService()
_embedder = _service.embedder


# Shared helpers

def _decode_image(file: UploadFile, raw: bytes) -> Image.Image:
    """Validate content-type and size, then decode to a PIL Image."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported image type: {file.content_type}. "
                f"Accepted: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )
    if len(raw) / (1024 * 1024) > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_MB} MB.",
        )
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")


# Endpoints 

@router.post("/search-image", tags=["Search"])
async def search_image(
    query: Optional[str]        = Form(default=None),
    top_k: int                  = Form(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
    page: int                   = Form(default=1, ge=1),
    page_size: int              = Form(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    category: Optional[str]     = Form(default=None),
    file:  Optional[UploadFile] = File(default=None),
):
    
    has_text  = bool(query and query.strip())
    has_image = file is not None

    if not has_text and not has_image:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: 'query' (text) or 'file' (image).",
        )

    # Use the raw query directly — query_cleaner is disconnected
    raw_query = query.strip() if has_text else None

    # Decode image if provided
    pil_image:      Optional[Image.Image] = None
    image_filename: Optional[str]         = None

    if has_image:
        raw            = await file.read()
        pil_image      = _decode_image(file, raw)
        image_filename = file.filename or "upload"

    try:
        if has_text and has_image:
            mode    = "hybrid"
            results = await _service.search_image_text(pil_image, raw_query, top_k=top_k, category=category or None)
        elif has_image:
            mode    = "image"
            results = await _service.search_by_image_raw(pil_image, top_k=top_k, category=category or None)
        else:
            mode    = "text"
            results = await _service.search_by_text(raw_query, top_k=top_k, category=category or None)

    except Exception:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail="Search failed — see server logs.")

    return {
        "mode":           mode,
        "query":          raw_query,
        "image_filename": image_filename,
        "total_returned": len(results),
        "total":          len(results),
        "page":           page,
        "page_size":      page_size,
        "total_pages":    max(1, -(-len(results) // page_size)),  # ceiling division
        "results":        results[(page - 1) * page_size : page * page_size],
    }


@router.post("/index-image", tags=["Index"])
async def index_image(
    file:     UploadFile    = File(...),
    caption:  str           = Form(...),
    category: Optional[str] = Form(default=None),
    metal:    Optional[str] = Form(default=None),
    item_id:  Optional[str] = Form(default=None),
):
    raw       = await file.read()
    pil_image = _decode_image(file, raw)

    # Sanitize item_id — strip commas/spaces, validate it's a plain int or UUID
    if item_id:
        cleaned_id = item_id.replace(",", "").replace(" ", "").strip()
        try:
            point_id = int(cleaned_id)       # try integer first
        except ValueError:
            try:
                import uuid as _uuid
                point_id = str(_uuid.UUID(cleaned_id))   # try UUID
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid item_id '{item_id}'. Must be a plain integer or UUID."
                )
    else:
        point_id = str(uuid.uuid4())

    try:
        vectors = _embedder.generate_embeddings_from_pil(pil_image, caption)
        payload = {
            "filename": file.filename or point_id,
            "caption":  caption,
            "category": category,
            "metal":    metal,
        }

        _service.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector={
                        "clip_image": vectors["clip_image"].tolist(),
                        "clip_text":  vectors["clip_text"].tolist(),
                        "dino_image": vectors["dino_image"].tolist(),
                    },
                    payload=payload,
                )
            ],
        )
        logger.info("Indexed image id=%s filename=%s", point_id, file.filename)

    except Exception:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail="Indexing failed — see server logs.")

    return {
        "status":   "indexed",
        "id":       point_id,
        "filename": file.filename,
        "caption":  caption,
    }