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

import numpy as np
import requests
from PIL import Image

from app import config
from app.db import sqlserver as db
from app.sync import index, state

logger = logging.getLogger(__name__)

STOP = threading.Event()  # set by the worker on SIGTERM for a graceful stop


class SyncInterrupted(Exception):
    pass


class SyncCancelled(Exception):
    """Raised at a checkpoint when the user stopped the job from the UI/API."""


_active_job: str | None = None  # job being executed (the worker runs one at a time)


@lru_cache(maxsize=1)
def get_embedder():
    from app.ingestion.embedder import Embedder  # heavy import, keep lazy
    return Embedder()


def _check_stop() -> None:
    if STOP.is_set():
        raise SyncInterrupted("shutdown requested")
    if _active_job and state.cancel_requested(_active_job):
        raise SyncCancelled("cancelled by user")


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
                  "images_removed": 0, "duplicates": 0, "to_process": 0, "processed": 0, "indexed": 0, "reused": 0, "skipped": 0, "failed": 0}
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
def _evict(p: Progress, c, collection: str, item_id: int, duplicates: set[int], indexed: bool = False) -> None:
    """`indexed` = the item is known to have points, so delete them even if the state DB lost its rows."""
    duplicates.add(item_id)
    p.c["duplicates"] += 1
    if state.remove_item_images(item_id) or indexed:
        index.delete_item(c, collection, item_id)  # duplicate that was indexed earlier


def _drop_duplicate_codes(p: Progress, c, collection: str, batch: list, duplicates: set[int],
                          master_ids: set[int]) -> list:
    """One product per ItemCode (a valid product beats an invalid one, then the lowest ItemID).

    Losers are skipped and their indexed images deleted; an earlier owner displaced by a better
    claimant in this batch is evicted the same way."""
    claims = [(str(r["ItemCode"]).strip().upper(), int(r["ItemID"]), index.to_bool(r.get("IsValid")))
              for r in batch if r.get("ItemCode") and str(r["ItemCode"]).strip()]
    owners, displaced = state.claim_item_codes(claims)
    for item_id in displaced:
        master_ids.discard(item_id)
        _evict(p, c, collection, item_id, duplicates)
    keep = []
    for r in batch:
        item_id = int(r["ItemID"])
        if owners.get(item_id, item_id) == item_id:
            keep.append(r)
        elif item_id not in displaced:
            _evict(p, c, collection, item_id, duplicates)
    return keep


def _dedupe_existing(p: Progress, c, collection: str) -> None:
    """Once per collection: delete duplicate ItemCodes that are already indexed, and seed code ownership."""
    flag = f"dedupe.{collection}"
    if state.kv_get(flag):
        return
    p.stage("removing duplicate item codes")
    by_code: dict[str, list] = {}
    for item_id, (code, valid) in index.indexed_items(c, collection).items():
        if code:
            by_code.setdefault(code, []).append((item_id, index.to_bool(valid)))
    claims, losers = [], []
    for code, items in by_code.items():
        items.sort(key=lambda t: state._code_rank(t[1], t[0]))
        claims += [(code, item_id, valid) for item_id, valid in items[:1]]
        losers += [item_id for item_id, _ in items[1:]]
    state.claim_item_codes(claims)
    for item_id in losers:
        _check_stop()
        _evict(p, c, collection, item_id, set(), indexed=True)
    state.kv_set(flag, True)


def _scan_catalog(p: Progress, c, collection: str, scopes: list, rebuild: bool) -> set[int]:
    """Phases 1-3 over the union of `scopes`. Returns the set of ItemIDs present in the master.

    A rebuild only holds its own scope, so everything outside it is dropped from the tracking state.
    An additive run (incremental) keeps the one shared collection: images outside the scanned scopes
    are left untouched, and only images that no longer exist in SQL Server are deactivated."""
    p.stage("scanning products")
    master_ids: set[int] = set()
    duplicates: set[int] = set()  # ItemIDs whose ItemCode already belongs to another product
    if rebuild:
        state.clear_item_codes()  # a rebuild only holds its own scope, so code ownership starts over
    else:
        _dedupe_existing(p, c, collection)
    for scope in scopes:
        for batch in db.iter_products(config.SYNC_PRODUCT_BATCH, scope):
            _check_stop()
            batch = [r for r in batch if int(r["ItemID"]) not in master_ids and int(r["ItemID"]) not in duplicates]  # seen via another scope
            batch = _drop_duplicate_codes(p, c, collection, batch, duplicates, master_ids)
            ids = [r["ItemID"] for r in batch]
            known = state.get_product_fingerprints(ids)
            for prod in batch:
                master_ids.add(int(prod["ItemID"]))
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
    window = (min(master_ids), max(master_ids)) if master_ids else None  # ItemID window of the synced products
    for batch in (db.iter_all_image_rows(config.SYNC_PRODUCT_BATCH * 5, *window) if window else ()):
        _check_stop()
        for r in batch:
            item_id, img_id = int(r["ItemID"]), int(r["ImgID"])  # SQLite keys are ints; keep `seen` comparable
            if item_id not in master_ids:
                continue
            seen.add(img_id)
            fp = _fingerprint(item_id, r["ImageURL"], r["ImgViewID"])
            state.upsert_image_seen(img_id, item_id, r["ImageURL"], r["ImgViewID"], fp)
        p.c["images_scanned"] += len(batch)
        p.tick()

    p.stage("applying deletion policy")
    gone_rows = state.images_not_seen_ids(seen)
    if not rebuild and gone_rows:
        # Outside the scanned scopes is not "deleted": keep those images searchable and only drop the
        # ones whose row is really gone from SQL Server.
        alive = db.existing_image_ids([r["img_id"] for r in gone_rows])
        gone_rows = [r for r in gone_rows if r["img_id"] not in alive]
    # Only images that were actually indexed have a point to deactivate/delete.
    indexed_gone = [r["img_id"] for r in gone_rows if r["status"] == "indexed"]
    for i in range(0, len(indexed_gone), 500):
        chunk = indexed_gone[i:i + 500]
        if config.SYNC_DELETION_POLICY == "delete":
            index.delete_images(c, collection, chunk)
        else:
            index.deactivate_images(c, collection, chunk)
    state.mark_removed_bulk([r["img_id"] for r in gone_rows])
    p.c["images_removed"] = len(gone_rows)
    return master_ids


def _donor_vectors(c, collection: str, rows) -> dict:
    """Vectors of already-indexed images that share a URL with a pending row: {url: (vectors, hash, etag)}."""
    donors = state.indexed_by_url([r["url"] for r in rows if r["url"]], config.EMBEDDING_MODEL_VERSION)
    if not donors:
        return {}
    recs = c.retrieve(collection, ids=[index.point_id(d["img_id"]) for d in donors.values()], with_vectors=True)
    by_id = {str(rec.id): rec.vector for rec in recs}
    out = {}
    for url, d in donors.items():
        vec = by_id.get(index.point_id(d["img_id"]))
        if vec and index.CLIP_VEC in vec and index.DINO_VEC in vec:
            out[url] = ({k: np.asarray(v, dtype="float32") for k, v in vec.items()}, d["content_hash"], d["etag"])
    return out


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
            # Many image rows point at the same file: reuse stored vectors instead of re-downloading/embedding.
            produced = _donor_vectors(c, collection, rows)  # url -> (vectors, content_hash, etag)
            # The tracking state is not per collection: an image can be marked indexed while its point lives in
            # another collection. Only trust the "unchanged" shortcuts when the point is really in this one.
            have = {str(pt.id) for pt in c.retrieve(
                collection, ids=[index.point_id(r["img_id"]) for r in rows], with_payload=False, with_vectors=False)}
            needs_embed = lambda r: force or index.point_id(r["img_id"]) not in have  # noqa: E731
            failed_urls: dict[str, str] = {}
            leaders, shared, first_seen = [], [], set()
            for r in rows:
                u = r["url"]
                if u and (u in produced or u in first_seen):
                    shared.append(r)
                else:
                    leaders.append(r)
                    if u:
                        first_seen.add(u)
            fetched = list(pool.map(lambda r: _fetch(r, needs_embed(r)), leaders))
            points = []
            for row, res in fetched:
                p.c["processed"] += 1
                img_id = row["img_id"]
                prod = products.get(row["item_id"])
                if prod is None:
                    state.mark_image(img_id, "removed"); p.c["skipped"] += 1
                    continue
                if isinstance(res, Exception):
                    failed_urls[row["url"]] = str(res)
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
                    if (not needs_embed(row) and row["content_hash"] == chash
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
                    produced[row["url"]] = (vectors, chash, etag)
                except Exception as exc:
                    failed_urls[row["url"]] = f"{type(exc).__name__}: {exc}"
                    state.mark_image(img_id, "failed", error=failed_urls[row["url"]])
                    p.c["failed"] += 1
            for row in shared:  # duplicates of an already-indexed or just-processed URL
                u, img_id = row["url"], row["img_id"]
                prod = products.get(row["item_id"])
                if prod is None:
                    state.mark_image(img_id, "removed"); p.c["processed"] += 1; p.c["skipped"] += 1
                elif u in produced:
                    vectors, chash, etag = produced[u]
                    points.append((img_id, etag, chash, index.make_point(
                        {"ImgID": img_id, "ImageURL": u, "ImgViewID": _view_id(row["view_id"])},
                        prod, vectors, chash)))
                    p.c["processed"] += 1; p.c["reused"] += 1
                elif u in failed_urls:
                    state.mark_image(img_id, "failed", error=failed_urls[u])
                    p.c["processed"] += 1; p.c["failed"] += 1
                # else: the first row with this URL was skipped/unchanged; stays pending and is reused next pass
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


def resolve_scopes(scope: dict | None, rebuild: bool) -> list:
    """Scopes this job scans, persisting the list for later runs.

    The saved list is the union of everything added to the index; the daily job refreshes all of it.
    A job carrying a scope adds it (incremental) or replaces the list (full rebuild); one without keeps it.
    [None] means no filters recorded yet, i.e. the SYNC_* environment defaults."""
    saved = state.kv_get("sync.scopes")
    if saved is None:  # state written before scopes were a list
        legacy = state.kv_get("sync.scope")
        saved = [legacy] if legacy else []
    scopes = ([scope] if rebuild else saved + ([scope] if scope not in saved else [])) if scope is not None else saved
    if scopes != state.kv_get("sync.scopes"):
        state.kv_set("sync.scopes", scopes)
    return scopes or [None]


def run(job_id: str, kind: str, scope: dict | None = None) -> None:
    """Execute a sync job.

    kind 'incremental' adds/refreshes images in the live collection (no alias change, nothing outside the
    scanned scopes is deactivated). kind 'full' rebuilds from scratch into a new collection + alias swap."""
    global _active_job
    _active_job = job_id
    rebuild = kind == "full"
    try:
        _run(job_id, rebuild, scope)
    except SyncCancelled:
        if rebuild:
            state.kv_set("rebuild.target", None)  # abandon the half-built collection; next rebuild starts fresh
        raise
    finally:
        _active_job = None


def _run(job_id: str, rebuild: bool, scope: dict | None) -> None:
    p = Progress(job_id)
    c = index.client()
    scopes = resolve_scopes(scope, rebuild)
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

    _scan_catalog(p, c, target, scopes, rebuild)
    _process_pending(p, c, target, force=rebuild)

    if rebuild:
        p.stage("validating rebuild")
        counts = state.image_counts()
        n_points = index.count_points(c, target)
        if counts["indexed"] == 0 and n_points == 0:
            raise RuntimeError("rebuild not promoted: no images were indexed (check catalog filters / image scan)")
        if counts["pending"] == 0 and n_points >= counts["indexed"]:
            index.switch_alias(c, target)
            state.kv_set("rebuild.target", None)
        else:
            raise RuntimeError(
                f"rebuild not promoted: points={n_points} indexed={counts['indexed']} pending={counts['pending']}")
    state.kv_set("last_catalog_counts", state.image_counts())
