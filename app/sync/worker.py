"""Sync worker + daily scheduler (run as its own container: `python -m app.sync.worker`).

Executes queued jobs one at a time and enqueues the daily incremental job at SCHEDULER_DAILY_AT
in SCHEDULER_TIMEZONE. A missed run (worker was down at the scheduled time) is caught up on start.
"""

import json
import logging
import signal
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from app import config
from app.sync import pipeline, state

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("sync.worker")

POLL_SECONDS = 5
MAX_JOB_ATTEMPTS = 3
BACKOFF_BASE_S = 30
BACKOFF_CAP_S = 600


def _run_job(job) -> None:
    job_id, kind = job["job_id"], job["kind"]
    scope = json.loads(job["params"]) if job["params"] else None
    logger.info("job=%s kind=%s trigger=%s starting", job_id, kind, job["trigger"])
    for attempt in range(1, MAX_JOB_ATTEMPTS + 1):
        try:
            pipeline.run(job_id, kind, scope)
            state.kv_set("last_success_at", state.now())
            state.finish_job(job_id, "succeeded")
            logger.info("job=%s succeeded", job_id)
            return
        except pipeline.SyncInterrupted:
            state.finish_job(job_id, "interrupted", "stopped by shutdown; will resume")
            state.enqueue_job(kind, "resume")  # picked up after restart
            logger.info("job=%s interrupted by shutdown", job_id)
            return
        except Exception as exc:
            logger.exception("job=%s attempt %d/%d failed", job_id, attempt, MAX_JOB_ATTEMPTS)
            if attempt == MAX_JOB_ATTEMPTS:
                state.finish_job(job_id, "failed", f"{type(exc).__name__}: {exc}")
                return
            delay = min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_CAP_S)
            state.update_job(job_id, stage=f"retry {attempt + 1} in {delay}s after {type(exc).__name__}")
            if pipeline.STOP.wait(delay):
                state.finish_job(job_id, "interrupted", "stopped by shutdown during backoff")
                return


def _maybe_schedule() -> None:
    tz = ZoneInfo(config.SCHEDULER_TIMEZONE)
    now = datetime.now(tz)
    hh, mm = (int(x) for x in config.SCHEDULER_DAILY_AT.split(":"))
    due = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    today = now.date().isoformat()
    if now >= due and state.kv_get("scheduler.last_date") != today:
        job_id = state.enqueue_job("incremental", "schedule")
        # Record the date either way: if a job was already active we do not stack a second one.
        state.kv_set("scheduler.last_date", today)
        logger.info("daily schedule fired for %s -> %s", today, job_id or "skipped (job already active)")


def main() -> None:
    signal.signal(signal.SIGTERM, lambda *_: pipeline.STOP.set())
    signal.signal(signal.SIGINT, lambda *_: pipeline.STOP.set())
    recovered = state.recover_running_jobs()
    logger.info("worker started (tz=%s daily_at=%s, recovered=%d)",
                config.SCHEDULER_TIMEZONE, config.SCHEDULER_DAILY_AT, recovered)
    pipeline.get_embedder()  # model warm-up
    while not pipeline.STOP.is_set():
        try:
            _maybe_schedule()
            job = state.claim_queued_job()
            if job:
                _run_job(job)
                continue
        except Exception:
            logger.exception("worker loop error")
        pipeline.STOP.wait(POLL_SECONDS)
    logger.info("worker stopped")


if __name__ == "__main__":
    main()
