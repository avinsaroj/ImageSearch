import hmac
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

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


class FullSyncRequest(BaseModel):
    confirm: bool = False  # must be true to build a new index when one already exists


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


def _submit(kind: str, trigger: str) -> dict:
    job_id = state.enqueue_job(kind, trigger)
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
    return _submit("full", "manual")


@router.post("/incremental", status_code=202)
def start_incremental(_: None = Depends(require_admin)):
    return _submit("incremental", "manual")
