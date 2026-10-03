"""Persistent synchronization state (SQLite on a mounted volume).

Holds job history, per-image processing status/hashes, product fingerprints and
checkpoints so syncs can resume after restarts and skip unchanged images.
"""

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from app import config

STALE_JOB_SECONDS = 600  # a running job with no heartbeat for this long is considered dead

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, trigger TEXT,
    stage TEXT, started_at TEXT, finished_at TEXT, heartbeat_at TEXT,
    counters TEXT DEFAULT '{}', error TEXT
);
CREATE TABLE IF NOT EXISTS images (
    img_id INTEGER PRIMARY KEY, item_id INTEGER NOT NULL, url TEXT, view_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',      -- pending | indexed | failed | removed
    attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT,
    content_hash TEXT, etag TEXT, model_version TEXT, fingerprint TEXT, last_processed_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_images_status ON images(status);
CREATE INDEX IF NOT EXISTS ix_images_item ON images(item_id);
CREATE INDEX IF NOT EXISTS ix_images_url ON images(url);
CREATE TABLE IF NOT EXISTS products (
    item_id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL, seen_at TEXT
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""

_local = threading.local()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _conn() -> sqlite3.Connection:
    c = getattr(_local, "conn", None)
    if c is None:
        os.makedirs(os.path.dirname(config.SYNC_STATE_DB) or ".", exist_ok=True)
        c = sqlite3.connect(config.SYNC_STATE_DB, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(_SCHEMA)
        if "params" not in {r["name"] for r in c.execute("PRAGMA table_info(jobs)")}:
            c.execute("ALTER TABLE jobs ADD COLUMN params TEXT")  # existing state DBs predate job scopes
        _local.conn = c
    return c


@contextmanager
def _tx():
    c = _conn()
    c.execute("BEGIN IMMEDIATE")
    try:
        yield c
        c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK")
        raise


# key/value checkpoints
def kv_get(key: str, default=None):
    row = _conn().execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def kv_set(key: str, value) -> None:
    _conn().execute(
        "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value)),
    )


# jobs
def enqueue_job(kind: str, trigger: str, params: dict | None = None) -> str | None:
    """Queue a job for the worker to pick up. Returns None if a live job is already active.

    params is the sync scope (category/ItemID filters); None keeps the last-used scope."""
    with _tx() as c:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=STALE_JOB_SECONDS)).isoformat(timespec="seconds")
        c.execute(
            "UPDATE jobs SET status='interrupted', finished_at=?, error='no heartbeat (worker died)' "
            "WHERE status='running' AND heartbeat_at < ?", (now(), cutoff))
        if c.execute("SELECT 1 FROM jobs WHERE status IN ('running','queued') AND heartbeat_at >= ?",
                     (cutoff,)).fetchone():
            return None
        job_id = uuid.uuid4().hex
        c.execute("INSERT INTO jobs(job_id,kind,status,trigger,stage,started_at,heartbeat_at,params) "
                  "VALUES(?,?,?,?,?,?,?,?)",
                  (job_id, kind, "queued", trigger, "queued", now(), now(),
                   None if params is None else json.dumps(params)))
        return job_id


def recover_running_jobs() -> int:
    """On worker start every 'running' job is orphaned; re-queue it so it resumes from persisted state."""
    return _conn().execute(
        "UPDATE jobs SET status='queued', stage='re-queued after restart', heartbeat_at=? WHERE status='running'",
        (now(),)).rowcount


def claim_queued_job() -> sqlite3.Row | None:
    with _tx() as c:
        row = c.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY started_at LIMIT 1").fetchone()
        if row:
            c.execute("UPDATE jobs SET status='running', heartbeat_at=? WHERE job_id=?", (now(), row["job_id"]))
        return row


def update_job(job_id: str, stage: str | None = None, counters: dict | None = None) -> None:
    sets, args = ["heartbeat_at=?"], [now()]
    if stage is not None:
        sets.append("stage=?"); args.append(stage)
    if counters is not None:
        sets.append("counters=?"); args.append(json.dumps(counters))
    _conn().execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id=?", (*args, job_id))


def finish_job(job_id: str, status: str, error: str | None = None) -> None:
    _conn().execute(
        "UPDATE jobs SET status=?, finished_at=?, error=?, stage=? WHERE job_id=?",
        (status, now(), (error or "")[:2000] or None, status, job_id))


def _job_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["counters"] = json.loads(d.get("counters") or "{}")
    d["params"] = json.loads(d["params"]) if d.get("params") else None
    return d


def current_job() -> dict | None:
    r = _conn().execute(
        "SELECT * FROM jobs WHERE status IN ('running','queued') ORDER BY started_at DESC LIMIT 1").fetchone()
    return _job_dict(r) if r else None


def job_history(limit: int = 50, offset: int = 0) -> list[dict]:
    rows = _conn().execute(
        "SELECT * FROM jobs ORDER BY started_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
    return [_job_dict(r) for r in rows]


def last_success(kind: str | None = None) -> dict | None:
    q, a = "SELECT * FROM jobs WHERE status='succeeded'", []
    if kind:
        q += " AND kind=?"; a.append(kind)
    r = _conn().execute(q + " ORDER BY finished_at DESC LIMIT 1", a).fetchone()
    return _job_dict(r) if r else None


# images
def get_image(img_id: int) -> sqlite3.Row | None:
    return _conn().execute("SELECT * FROM images WHERE img_id=?", (img_id,)).fetchone()


def get_images_bulk(img_ids: list[int]) -> dict[int, sqlite3.Row]:
    if not img_ids:
        return {}
    marks = ",".join("?" * len(img_ids))
    rows = _conn().execute(f"SELECT * FROM images WHERE img_id IN ({marks})", img_ids).fetchall()
    return {r["img_id"]: r for r in rows}


def upsert_image_seen(img_id: int, item_id: int, url: str | None, view_id, fingerprint: str) -> None:
    """Register an image row from SQL Server; resets to pending only when its fingerprint changed."""
    _conn().execute(
        "INSERT INTO images(img_id,item_id,url,view_id,fingerprint,status) VALUES(?,?,?,?,?, 'pending') "
        "ON CONFLICT(img_id) DO UPDATE SET item_id=excluded.item_id, url=excluded.url, view_id=excluded.view_id, "
        "status=CASE WHEN images.fingerprint IS NOT excluded.fingerprint OR images.status='removed' "
        "THEN 'pending' ELSE images.status END, "
        "attempts=CASE WHEN images.fingerprint IS NOT excluded.fingerprint THEN 0 ELSE images.attempts END, "
        "fingerprint=excluded.fingerprint",
        (img_id, item_id, url, None if view_id is None else str(view_id), fingerprint))


def mark_image(img_id: int, status: str, error: str | None = None, content_hash: str | None = None,
               etag: str | None = None, model_version: str | None = None) -> None:
    done = status == "indexed"
    _conn().execute(
        "UPDATE images SET status=?, last_error=?, attempts=attempts+?, "
        "content_hash=COALESCE(?,content_hash), etag=COALESCE(?,etag), model_version=COALESCE(?,model_version), "
        "last_processed_at=? WHERE img_id=?",
        (status, (error or "")[:500] or None, 0 if done else 1, content_hash, etag, model_version, now(), img_id))


def requeue_images(statuses=("failed",), max_attempts: int | None = None, reset_attempts: bool = False) -> int:
    marks = ",".join("?" * len(statuses))
    q = f"UPDATE images SET status='pending'{', attempts=0' if reset_attempts else ''} WHERE status IN ({marks})"
    a = list(statuses)
    if max_attempts:
        q += " AND attempts < ?"; a.append(max_attempts)
    return _conn().execute(q, a).rowcount


def requeue_stale_model(model_version: str) -> int:
    return _conn().execute(
        "UPDATE images SET status='pending' WHERE status='indexed' AND model_version IS NOT ?",
        (model_version,)).rowcount


def pending_images(limit: int, max_attempts: int, after_img_id: int = 0) -> list[sqlite3.Row]:
    return _conn().execute(
        "SELECT * FROM images WHERE status='pending' AND attempts < ? AND img_id > ? ORDER BY img_id LIMIT ?",
        (max_attempts, after_img_id, limit)).fetchall()


def indexed_by_url(urls: list[str], model_version: str) -> dict[str, sqlite3.Row]:
    """One already-indexed donor row per URL (same model version) so identical files are not re-embedded."""
    out: dict[str, sqlite3.Row] = {}
    for i in range(0, len(urls), 500):
        chunk = urls[i:i + 500]
        marks = ",".join("?" * len(chunk))
        for r in _conn().execute(
                f"SELECT * FROM images WHERE status='indexed' AND model_version=? AND content_hash IS NOT NULL "
                f"AND url IN ({marks})", (model_version, *chunk)):
            out.setdefault(r["url"], r)
    return out


def images_not_seen_ids(seen_ids: set[int]) -> list[sqlite3.Row]:
    rows = _conn().execute("SELECT img_id,item_id,status FROM images WHERE status!='removed'").fetchall()
    return [r for r in rows if r["img_id"] not in seen_ids]


def mark_removed_bulk(img_ids: list[int]) -> None:
    """Mark many images removed in a single transaction (per-row commits are far too slow for millions)."""
    with _tx() as c:
        for i in range(0, len(img_ids), 5000):
            chunk = img_ids[i:i + 5000]
            c.executemany("UPDATE images SET status='removed', last_processed_at=? WHERE img_id=?",
                          [(now(), x) for x in chunk])


def image_counts() -> dict:
    rows = _conn().execute("SELECT status, COUNT(*) n FROM images GROUP BY status").fetchall()
    out = {"pending": 0, "indexed": 0, "failed": 0, "removed": 0}
    out.update({r["status"]: r["n"] for r in rows})
    return out


def failed_images(limit: int = 100, offset: int = 0) -> list[dict]:
    rows = _conn().execute(
        "SELECT img_id,item_id,url,attempts,last_error,last_processed_at FROM images "
        "WHERE status='failed' ORDER BY last_processed_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
    return [dict(r) for r in rows]


# products
def get_product_fingerprints(item_ids: list[int]) -> dict[int, str]:
    if not item_ids:
        return {}
    marks = ",".join("?" * len(item_ids))
    rows = _conn().execute(f"SELECT item_id,fingerprint FROM products WHERE item_id IN ({marks})", item_ids)
    return {r["item_id"]: r["fingerprint"] for r in rows}


def set_product_fingerprint(item_id: int, fingerprint: str) -> None:
    _conn().execute(
        "INSERT INTO products(item_id,fingerprint,seen_at) VALUES(?,?,?) "
        "ON CONFLICT(item_id) DO UPDATE SET fingerprint=excluded.fingerprint, seen_at=excluded.seen_at",
        (item_id, fingerprint, now()))
