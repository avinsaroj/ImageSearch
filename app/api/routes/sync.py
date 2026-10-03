import hmac
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app import config
from app.db import sqlserver as db
from app.sync import index, state

router = APIRouter(prefix="/sync", tags=["Sync"])
logger = logging.getLogger(__name__)


def require_admin(x_admin_token: Optional[str] = Header(default=None)) -> None:
    """Protect admin endpoints with a shared token (header X-Admin-Token)."""
    if not config.ADMIN_API_TOKEN:
        raise HTTPException(status_code=503, detail="ADMIN_API_TOKEN is not configured; admin endpoints disabled")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, config.ADMIN_API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")


class SyncScope(BaseModel):
    """Which products (and their images) a sync covers. All fields optional; empty means everything."""
    categories: list[str] = []
    plain_gold: Optional[bool] = None  # True / False / None (any); same for the flags below
    solitaire: Optional[bool] = None
    valid: Optional[bool] = None
    franchise: Optional[bool] = None
    search_text: Optional[str] = Field(default=None, max_length=100)  # substring of item code or status remark
    max_products: Optional[int] = Field(default=None, ge=1)  # embed only the first N matching products (test runs)


class FullSyncRequest(BaseModel):
    confirm: bool = False  # must be true to build a new index when one already exists
    scope: Optional[SyncScope] = None  # None keeps the last-used scope (or the SYNC_* env defaults)


def _qdrant_status() -> dict:
    try:
        c = index.client()
        real = index.resolve_alias(c)
        if not real:
            return {"status": "no_index", "alias": config.PRODUCT_COLLECTION}
        return {"status": "ok", "alias": config.PRODUCT_COLLECTION, "collection": real,
                "points": index.count_points(c, real)}
    except Exception as exc:
        return {"status": "unreachable", "error": type(exc).__name__}


def _sql_totals() -> dict:
    try:
        return {"products": db.count_products(), "images": db.count_images()}
    except Exception as exc:
        return {"products": None, "images": None, "error": type(exc).__name__}


@router.get("/status")
def sync_status(_: None = Depends(require_admin)):
    counts = state.image_counts()
    return {
        "current_job": state.current_job(),
        "last_success": state.last_success(),
        "model_version": config.EMBEDDING_MODEL_VERSION,
        "sql_server": _sql_totals(),
        "images": {"indexed": counts["indexed"], "pending": counts["pending"],
                   "failed": counts["failed"], "removed": counts["removed"]},
        "qdrant": _qdrant_status(),
        "scheduler": {"timezone": config.SCHEDULER_TIMEZONE, "daily_at": config.SCHEDULER_DAILY_AT,
                      "last_fired_date": state.kv_get("scheduler.last_date")},
    }


@router.get("/history")
def sync_history(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0),
                 _: None = Depends(require_admin)):
    return {"limit": limit, "offset": offset, "jobs": state.job_history(limit, offset)}


@router.get("/failures")
def sync_failures(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                  _: None = Depends(require_admin)):
    return {"limit": limit, "offset": offset, "failures": state.failed_images(limit, offset)}


@router.post("/retry-failed", status_code=202)
def retry_failed(_: None = Depends(require_admin)):
    """Reset failed images (attempt counter included) and queue an incremental job to retry them."""
    n = state.requeue_images(("failed",), reset_attempts=True)
    job = state.enqueue_job("incremental", "retry-failed")
    if job is None:
        raise HTTPException(status_code=409, detail="A sync job is already active; failures were re-queued for it")
    return {"job_id": job, "requeued_images": n}


@router.get("/scope")
def current_scope(_: None = Depends(require_admin)):
    """Scope the next incremental/scheduled job will use: the last full sync's, else the env defaults."""
    saved = state.kv_get("sync.scope")
    return {"scope": db.scope_defaults() if saved is None else saved, "source": "env" if saved is None else "last_run"}


@router.post("/preview")
def preview_scope(scope: SyncScope, _: None = Depends(require_admin)):
    """Count the products/images a scope would embed, without queuing anything."""
    try:
        return db.count_scope(scope.model_dump())
    except Exception as exc:
        logger.exception("scope preview failed")
        raise HTTPException(status_code=502, detail=f"SQL Server query failed: {type(exc).__name__}")


def _submit(kind: str, trigger: str, scope: Optional[dict] = None) -> dict:
    job_id = state.enqueue_job(kind, trigger, scope)
    if job_id is None:
        raise HTTPException(status_code=409, detail="A sync job is already running or queued")
    logger.info("queued %s job %s (trigger=%s)", kind, job_id, trigger)
    return {"job_id": job_id, "kind": kind, "status": "queued"}


@router.post("/full", status_code=202)
def start_full(req: FullSyncRequest, _: None = Depends(require_admin)):
    """Queue a full (re)build. Fills a new collection and swaps the alias only after validation."""
    existing = _qdrant_status()
    if existing.get("points") and not req.confirm:
        raise HTTPException(status_code=409, detail="An index already exists; resend with confirm=true to rebuild")
    return _submit("full", "manual", req.scope.model_dump() if req.scope else None)


@router.post("/incremental", status_code=202)
def start_incremental(_: None = Depends(require_admin)):
    return _submit("incremental", "manual")
