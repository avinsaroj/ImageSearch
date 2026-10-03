"""Full and incremental synchronization: SQL Server -> image download -> embeddings -> Qdrant.

Both kinds share one flow:
  1. scan product master (keyset pages) and diff against stored fingerprints
  2. scan image rows (metadata only) and register new/changed images as 'pending'
  3. apply the deletion policy to images/products that disappeared
  4. process 'pending' images in bounded batches (download -> validate -> embed -> idempotent upsert)
Pending status is persisted, so an interrupted job resumes where it stopped.
"""

import hashlib
import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import requests
from PIL import Image

from app import config
from app.db import sqlserver as db
from app.sync import index, state

logger = logging.getLogger(__name__)

STOP = threading.Event()  # set by the worker on SIGTERM for a graceful stop


class SyncInterrupted(Exception):
    pass


@lru_cache(maxsize=1)
def get_embedder():
    from app.ingestion.embedder import Embedder  # heavy import, keep lazy
    return Embedder()


def _check_stop() -> None:
    if STOP.is_set():
        raise SyncInterrupted("shutdown requested")


def _fingerprint(*parts) -> str:
    return hashlib.sha1("|".join("" if p is None else str(p) for p in parts).encode()).hexdigest()


def _product_fp(p: dict) -> str:
    return _fingerprint(*(p.get(k) for k in (
        "ItemCode", "CategoryCode", "IsPlainGold", "IsSolitaire", "IsValid",
        "IsFranchiseItem", "StatusRemark", "UpdatedOn")))


class Progress:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.c = {"products_scanned": 0, "products_changed": 0, "images_scanned": 0, "images_registered": 0,
                  "images_removed": 0, "to_process": 0, "processed": 0, "indexed": 0, "skipped": 0, "failed": 0}
        self._last = 0.0

    def stage(self, name: str) -> None:
        logger.info("job=%s stage=%s", self.job_id, name)
        state.update_job(self.job_id, stage=name, counters=self.c)

    def tick(self, force: bool = False) -> None:
        if force or time.monotonic() - self._last > 2:
            state.update_job(self.job_id, counters=self.c)
            self._last = time.monotonic()


# download / validate
_session = requests.Session()
_session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=config.SYNC_DOWNLOAD_CONCURRENCY))
_session.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=config.SYNC_DOWNLOAD_CONCURRENCY))


def download(url: str, etag: str | None = None):
    """Return (bytes|None, etag, not_modified). Retries transient errors with exponential backoff."""
    headers = {"If-None-Match": etag} if etag else {}
    limit = config.SYNC_MAX_IMAGE_MB * 1024 * 1024
    last_exc: Exception | None = None
    for attempt in range(config.SYNC_DOWNLOAD_RETRIES):
        try:
            with _session.get(url, headers=headers, timeout=config.SYNC_DOWNLOAD_TIMEOUT_S, stream=True) as r:
                if r.status_code == 304:
                    return None, etag, True
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    raise PermissionError(f"HTTP {r.status_code}")  # permanent, do not retry
                r.raise_for_status()
                buf = io.BytesIO()
                for chunk in r.iter_content(64 * 1024):
                    buf.write(chunk)
                    if buf.tell() > limit:
                        raise PermissionError(f"image larger than {config.SYNC_MAX_IMAGE_MB} MB")
                return buf.getvalue(), r.headers.get("ETag"), False
        except PermissionError:
            raise
        except Exception as exc:  # timeouts, connection errors, 5xx, 429
            last_exc = exc
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"download failed after retries: {type(last_exc).__name__}")


def decode(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.verify()  # detects truncated/corrupt files
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if min(img.size) < 16:
        raise ValueError(f"image too small: {img.size}")
    return img


def _fetch(row, force: bool):
    """Worker thread: download one image. Returns (row, result|exception)."""
    try:
        if not row["url"]:
            raise ValueError("missing image URL")
        use_etag = row["etag"] if (not force and row["model_version"] == config.EMBEDDING_MODEL_VERSION) else None
        return row, download(row["url"], use_etag)
    except Exception as exc:
        return row, exc


def _view_id(v):
    return int(v) if isinstance(v, str) and v.lstrip("-").isdigit() else v


# phases
def _scan_catalog(p: Progress, c, collection: str) -> set[int]:
    """Phases 1-3. Returns the set of ItemIDs present in the master."""
    p.stage("scanning products")
    master_ids: set[int] = set()
    for batch in db.iter_products(config.SYNC_PRODUCT_BATCH):
        _check_stop()
        ids = [r["ItemID"] for r in batch]
        known = state.get_product_fingerprints(ids)
        for prod in batch:
            master_ids.add(prod["ItemID"])
            fp = _product_fp(prod)
            old = known.get(prod["ItemID"])
            if old != fp:
                if old is not None:  # existing product with changed metadata -> refresh payload only
                    index.update_product_payload(c, collection, prod)
                    p.c["products_changed"] += 1
                state.set_product_fingerprint(prod["ItemID"], fp)
        p.c["products_scanned"] += len(batch)
        p.tick()

    p.stage("scanning images")
    seen: set[int] = set()
    for batch in db.iter_all_image_rows(config.SYNC_PRODUCT_BATCH * 5):
        _check_stop()
        for r in batch:
            if r["ItemID"] not in master_ids:
                continue
            seen.add(r["ImgID"])
            fp = _fingerprint(r["ItemID"], r["ImageURL"], r["ImgViewID"])
            state.upsert_image_seen(r["ImgID"], r["ItemID"], r["ImageURL"], r["ImgViewID"], fp)
        p.c["images_scanned"] += len(batch)
        p.tick()

    p.stage("applying deletion policy")
    gone = [r["img_id"] for r in state.images_not_seen_ids(seen)]
    for i in range(0, len(gone), 500):
        chunk = gone[i:i + 500]
        if config.SYNC_DELETION_POLICY == "delete":
            index.delete_images(c, collection, chunk)
        else:
            index.deactivate_images(c, collection, chunk)
        for img_id in chunk:
            state.mark_image(img_id, "removed")
    p.c["images_removed"] = len(gone)
    return master_ids


def _process_pending(p: Progress, c, collection: str, force: bool) -> None:
    p.stage("embedding images")
    embedder = get_embedder()
    p.c["to_process"] = state.image_counts()["pending"]
    pool = ThreadPoolExecutor(max_workers=config.SYNC_DOWNLOAD_CONCURRENCY)
    chunk_size = config.SYNC_EMBED_BATCH * 4
    try:
        while True:
            _check_stop()
            rows = state.pending_images(chunk_size, config.SYNC_MAX_ATTEMPTS)
            if not rows:
                break
            products = {pr["ItemID"]: pr for pr in db.fetch_products_by_ids(sorted({r["item_id"] for r in rows}))}
            fetched = list(pool.map(lambda r: _fetch(r, force), rows))
            points = []
            for row, res in fetched:
                p.c["processed"] += 1
                img_id = row["img_id"]
                prod = products.get(row["item_id"])
                if prod is None:
                    state.mark_image(img_id, "removed"); p.c["skipped"] += 1
                    continue
                if isinstance(res, Exception):
                    state.mark_image(img_id, "failed", error=str(res)); p.c["failed"] += 1
                    continue
                data, etag, not_modified = res
                img_meta = {"ImgID": img_id, "ImageURL": row["url"], "ImgViewID": _view_id(row["view_id"])}
                try:
                    if not_modified:
                        index.update_product_payload(c, collection, prod)
                        c.set_payload(collection, {"ImageURL": row["url"], "ImgViewID": img_meta["ImgViewID"],
                                                   "is_active": True}, points=[index.point_id(img_id)])
                        state.mark_image(img_id, "indexed", model_version=config.EMBEDDING_MODEL_VERSION)
                        p.c["skipped"] += 1
                        continue
                    chash = hashlib.sha256(data).hexdigest()
                    if (not force and row["content_hash"] == chash
                            and row["model_version"] == config.EMBEDDING_MODEL_VERSION):
                        c.set_payload(collection, {"ImageURL": row["url"], "ImgViewID": img_meta["ImgViewID"],
                                                   "is_active": True}, points=[index.point_id(img_id)])
                        state.mark_image(img_id, "indexed", etag=etag, model_version=config.EMBEDDING_MODEL_VERSION)
                        p.c["skipped"] += 1
                        continue
                    pil = decode(data)
                    vectors = {
                        index.CLIP_VEC: embedder.normalize(embedder.embed_clip_image(pil)),
                        index.DINO_VEC: embedder.normalize(embedder.embed_dino_image(pil)),
                    }
                    points.append((img_id, etag, chash, index.make_point(
                        {**img_meta, "ImgID": img_id}, prod, vectors, chash)))
                except Exception as exc:
                    state.mark_image(img_id, "failed", error=f"{type(exc).__name__}: {exc}")
                    p.c["failed"] += 1
            if points:
                try:
                    index.upsert(c, collection, [pt for *_, pt in points])
                    for img_id, etag, chash, _ in points:
                        state.mark_image(img_id, "indexed", content_hash=chash, etag=etag,
                                         model_version=config.EMBEDDING_MODEL_VERSION)
                        p.c["indexed"] += 1
                except Exception as exc:
                    for img_id, *_ in points:
                        state.mark_image(img_id, "failed", error=f"qdrant upsert: {type(exc).__name__}")
                        p.c["failed"] += 1
            p.tick(force=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def run(job_id: str, kind: str) -> None:
    """Execute a sync job. kind: 'incremental' | 'full' (full = rebuild into a new collection + alias swap)."""
    p = Progress(job_id)
    c = index.client()
    rebuild = kind == "full"
    if rebuild:
        target = state.kv_get("rebuild.target")
        if not target:  # fresh rebuild (an interrupted one resumes into the same target)
            target = index.new_collection_name()
            index.create_collection(c, target)
            state.kv_set("rebuild.target", target)
            state.requeue_images(("indexed", "failed"))
    else:
        target = index.ensure_active_collection(c)
        state.requeue_images(("failed",), max_attempts=config.SYNC_MAX_ATTEMPTS)
        state.requeue_stale_model(config.EMBEDDING_MODEL_VERSION)

    if rebuild and index.resolve_alias(c) is None:
        index.switch_alias(c, target)  # first ever index: serve it while it fills

    _scan_catalog(p, c, target)
    _process_pending(p, c, target, force=rebuild)

    if rebuild:
        p.stage("validating rebuild")
        counts = state.image_counts()
        n_points = index.count_points(c, target)
        if counts["pending"] == 0 and n_points >= counts["indexed"]:
            index.switch_alias(c, target)
            state.kv_set("rebuild.target", None)
        else:
            raise RuntimeError(
                f"rebuild not promoted: points={n_points} indexed={counts['indexed']} pending={counts['pending']}")
    state.kv_set("last_catalog_counts", state.image_counts())
