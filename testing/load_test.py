"""
load_test.py — Concurrency test for Jewellery Image Search API
Tests: /health, /search-image (text), /search-image (image upload)

Usage:
    pip install httpx pillow
    python load_test.py --url http://localhost:8000 --users 50 --duration 30

Requirements:
    pip install httpx pillow
"""

import argparse
import asyncio
import io
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
from PIL import Image, ImageDraw


# ── Synthetic test image (no file needed) ─────────────────────────────────────

def make_test_image_bytes() -> bytes:
    """Generate a small in-memory JPEG so we don't need a real image file."""
    img = Image.new("RGB", (224, 224), color=(180, 140, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 174, 174], fill=(220, 180, 120), outline=(100, 70, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


TEST_IMAGE_BYTES = make_test_image_bytes()

TEXT_QUERIES = [
    "rose gold ring",
    "diamond necklace",
    "silver bracelet",
    "pearl earrings",
    "gold bangle",
    "emerald pendant",
    "ruby ring",
    "platinum chain",
    "sapphire earrings",
    "vintage brooch",
]


# ── Result tracking ────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    endpoint: str
    status:   int
    latency:  float          # seconds
    error:    Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and 200 <= self.status < 300


@dataclass
class Stats:
    results: list = field(default_factory=list)

    def add(self, r: RequestResult):
        self.results.append(r)

    def report(self) -> dict:
        if not self.results:
            return {}

        successes = [r for r in self.results if r.success]
        failures  = [r for r in self.results if not r.success]
        latencies = [r.latency for r in successes]

        by_endpoint = {}
        for r in self.results:
            by_endpoint.setdefault(r.endpoint, {"ok": 0, "fail": 0, "latencies": []})
            if r.success:
                by_endpoint[r.endpoint]["ok"]      += 1
                by_endpoint[r.endpoint]["latencies"].append(r.latency)
            else:
                by_endpoint[r.endpoint]["fail"] += 1

        return {
            "total":        len(self.results),
            "success":      len(successes),
            "failure":      len(failures),
            "success_rate": f"{len(successes)/len(self.results)*100:.1f}%",
            "latency": {
                "min":    f"{min(latencies)*1000:.0f}ms" if latencies else "—",
                "max":    f"{max(latencies)*1000:.0f}ms" if latencies else "—",
                "mean":   f"{statistics.mean(latencies)*1000:.0f}ms" if latencies else "—",
                "median": f"{statistics.median(latencies)*1000:.0f}ms" if latencies else "—",
                "p95":    f"{sorted(latencies)[int(len(latencies)*0.95)-1]*1000:.0f}ms" if len(latencies) >= 20 else "—",
                "p99":    f"{sorted(latencies)[int(len(latencies)*0.99)-1]*1000:.0f}ms" if len(latencies) >= 100 else "—",
            },
            "by_endpoint": {
                ep: {
                    "ok":        d["ok"],
                    "fail":      d["fail"],
                    "mean_ms":   f"{statistics.mean(d['latencies'])*1000:.0f}ms" if d["latencies"] else "—",
                }
                for ep, d in by_endpoint.items()
            },
            "errors": list({r.error for r in failures if r.error})[:10],
        }


# ── Virtual user ──────────────────────────────────────────────────────────────

async def virtual_user(
    user_id:  int,
    base_url: str,
    duration: float,
    stats:    Stats,
    semaphore: asyncio.Semaphore,
):
    """
    Simulates a single user hammering the API for `duration` seconds.
    Alternates between text search and image search.
    """
    deadline = time.monotonic() + duration
    query_idx = user_id % len(TEXT_QUERIES)

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        iteration = 0
        while time.monotonic() < deadline:
            async with semaphore:
                endpoint = "/search-image"
                t0 = time.monotonic()
                try:
                    if iteration % 2 == 0:
                        # Text search
                        r = await client.post(
                            endpoint,
                            data={"query": TEXT_QUERIES[query_idx % len(TEXT_QUERIES)], "top_k": 5},
                        )
                    else:
                        # Image search
                        r = await client.post(
                            endpoint,
                            data={"top_k": 5},
                            files={"file": ("test.jpg", TEST_IMAGE_BYTES, "image/jpeg")},
                        )
                    latency = time.monotonic() - t0
                    stats.add(RequestResult(endpoint, r.status_code, latency))

                except httpx.TimeoutException:
                    latency = time.monotonic() - t0
                    stats.add(RequestResult(endpoint, 0, latency, error="Timeout"))
                except Exception as e:
                    latency = time.monotonic() - t0
                    stats.add(RequestResult(endpoint, 0, latency, error=str(e)[:80]))

            query_idx += 1
            iteration += 1
            await asyncio.sleep(0.1)   # slight pause between requests per user


# ── Health check ──────────────────────────────────────────────────────────────

async def health_check(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
            r = await client.get("/health")
            return r.status_code == 200
    except Exception:
        return False


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_load_test(base_url: str, users: int, duration: float, max_concurrent: int):
    print(f"\n{'='*60}")
    print(f"  Jewellery Search API — Load Test")
    print(f"{'='*60}")
    print(f"  Target:      {base_url}")
    print(f"  Users:       {users}")
    print(f"  Duration:    {duration}s per user")
    print(f"  Max concurrent requests: {max_concurrent}")
    print(f"{'='*60}\n")

    print("  Checking API health…", end=" ", flush=True)
    if not await health_check(base_url):
        print("FAILED — API is not reachable. Aborting.")
        return
    print("OK\n")

    stats     = Stats()
    semaphore = asyncio.Semaphore(max_concurrent)

    print(f"  Spawning {users} virtual users…")
    wall_start = time.monotonic()

    tasks = [
        asyncio.create_task(
            virtual_user(i, base_url, duration, stats, semaphore)
        )
        for i in range(users)
    ]

    # Progress ticker
    async def ticker():
        while not all(t.done() for t in tasks):
            elapsed = time.monotonic() - wall_start
            done    = sum(1 for t in tasks if t.done())
            print(f"\r  Running… {elapsed:.0f}s | users done: {done}/{users} | requests: {len(stats.results)}", end="", flush=True)
            await asyncio.sleep(1)

    await asyncio.gather(*tasks, ticker())
    wall_elapsed = time.monotonic() - wall_start
    print(f"\r  Completed in {wall_elapsed:.1f}s" + " " * 30)

    # ── Print report ──────────────────────────────────────────────────────────
    report = stats.report()
    if not report:
        print("  No results recorded.")
        return

    rps = report["total"] / wall_elapsed

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total requests  : {report['total']}")
    print(f"  Successful      : {report['success']}  ({report['success_rate']})")
    print(f"  Failed          : {report['failure']}")
    print(f"  Throughput      : {rps:.1f} req/s")
    print(f"\n  Latency (successful requests)")
    lat = report["latency"]
    print(f"  ├─ min    : {lat['min']}")
    print(f"  ├─ mean   : {lat['mean']}")
    print(f"  ├─ median : {lat['median']}")
    print(f"  ├─ p95    : {lat['p95']}")
    print(f"  ├─ p99    : {lat['p99']}")
    print(f"  └─ max    : {lat['max']}")

    print(f"\n  Per-endpoint breakdown")
    for ep, d in report["by_endpoint"].items():
        print(f"  {ep:30s}  ok={d['ok']}  fail={d['fail']}  mean={d['mean_ms']}")

    if report["errors"]:
        print(f"\n  Errors seen (sample):")
        for e in report["errors"]:
            print(f"    • {e}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    success_pct = report["success"] / report["total"] * 100 if report["total"] else 0
    if success_pct >= 99 and report["failure"] == 0:
        verdict = "PASS — app handled all 50 concurrent users with no errors"
    elif success_pct >= 95:
        verdict = f"PASS (with warnings) — {report['failure']} requests failed ({100-success_pct:.1f}%)"
    else:
        verdict = f"FAIL — {report['failure']} requests failed ({100-success_pct:.1f}% failure rate)"
    print(f"  VERDICT: {verdict}")
    print(f"{'='*60}\n")

    # Save JSON report
    out = {
        "config": {"url": base_url, "users": users, "duration": duration},
        "wall_time_seconds": round(wall_elapsed, 2),
        "throughput_rps":    round(rps, 2),
        **report,
    }
    with open("load_test_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Full report saved → load_test_report.json\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test for Jewellery Search API")
    parser.add_argument("--url",        default="http://localhost:8000", help="API base URL")
    parser.add_argument("--users",      type=int, default=50,            help="Number of virtual users (default 50)")
    parser.add_argument("--duration",   type=int, default=30,            help="Seconds each user runs (default 30)")
    parser.add_argument("--concurrent", type=int, default=50,            help="Max simultaneous in-flight requests")
    args = parser.parse_args()

    asyncio.run(run_load_test(args.url, args.users, args.duration, args.concurrent))