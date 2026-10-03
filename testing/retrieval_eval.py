"""
retrieval_eval.py  —  Phase 5: Retrieval Quality & Similarity Validation
=========================================================================
Tests three things:
  1. Precision@K  — of the top-K results, how many share the correct category?
  2. Category coherence — do results cluster around the right jewellery type?
  3. Score sanity — are similarity scores in a valid range and well-ordered?

Usage:
    pip install httpx pillow rich
    python retrieval_eval.py --url http://localhost:8000 --metadata data/metadata_cleaned.json

The script samples images from your metadata, queries the API, and
measures retrieval quality automatically — no human labelling needed.

Output:
    • Terminal report (rich table)
    • retrieval_eval_report.json
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
from PIL import Image

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH = True
except ImportError:
    RICH = False


# ── Helper: safely extract category from a field that may be str or list ───────

def extract_category(val, default="—") -> str:
    if isinstance(val, list):
        val = val[0] if val else default
    return str(val).lower() if val else default


# ── Text queries with known expected category ──────────────────────────────────

TEXT_PROBES = [
    {"query": "gold ring",            "expected_category": "ring"},
    {"query": "diamond ring",         "expected_category": "ring"},
    {"query": "silver necklace",      "expected_category": "necklace"},
    {"query": "pearl necklace",       "expected_category": "necklace"},
    {"query": "gold bracelet",        "expected_category": "bracelet"},
    {"query": "silver bracelet",      "expected_category": "bracelet"},
    {"query": "diamond earrings",     "expected_category": "earring"},
    {"query": "gold earrings",        "expected_category": "earring"},
    {"query": "gold pendant",         "expected_category": "necklace"},
    {"query": "silver pendant",       "expected_category": "necklace"},
]


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    probe:          str
    mode:           str            # "text" or "image"
    expected:       str
    top_k:          int
    results:        list
    latency_ms:     float
    error:          Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.results)

    def _get_category(self, result: dict) -> str:
        return extract_category((result.get("payload") or {}).get("category", "—"))

    def precision_at_k(self, k: int) -> float:
        """Fraction of top-k results whose category matches expected."""
        if not self.results:
            return 0.0
        top  = self.results[:k]
        hits = sum(
            1 for r in top
            if self._get_category(r) == self.expected.lower()
        )
        return hits / len(top)

    def scores_are_ordered(self) -> bool:
        """Check that results come back in descending score order."""
        scores = [r.get("score", 0) for r in self.results]
        return all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def scores_in_range(self) -> bool:
        """Cosine similarity scores should be between -1 and 1 (fused may be higher)."""
        scores = [r.get("score", 0) for r in self.results]
        return all(-1.0 <= s <= 2.0 for s in scores)

    def top1_category(self) -> str:
        if not self.results:
            return "—"
        return self._get_category(self.results[0])


# ── API helpers ────────────────────────────────────────────────────────────────

async def query_text(client: httpx.AsyncClient, query: str, top_k: int) -> tuple:
    t0 = time.monotonic()
    r  = await client.post("/search-image", data={"query": query, "top_k": top_k}, timeout=60)
    ms = (time.monotonic() - t0) * 1000
    r.raise_for_status()
    return r.json().get("results", []), ms


async def query_image(client: httpx.AsyncClient, img_path: str, top_k: int) -> tuple:
    with open(img_path, "rb") as f:
        raw = f.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    t0 = time.monotonic()
    r  = await client.post(
        "/search-image",
        data={"top_k": top_k},
        files={"file": ("query.jpg", buf.read(), "image/jpeg")},
        timeout=60,
    )
    ms = (time.monotonic() - t0) * 1000
    r.raise_for_status()
    return r.json().get("results", []), ms


async def health_ok(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=5) as c:
            return (await c.get("/health")).status_code == 200
    except Exception:
        return False


# ── Metadata sampling ──────────────────────────────────────────────────────────

def sample_images(metadata_path: str, per_category: int = 3) -> list:
    """
    Return up to `per_category` images per category from metadata.
    Each entry: {"filepath", "category", "filename"}
    """
    if not os.path.exists(metadata_path):
        return []
    with open(metadata_path) as f:
        records = json.load(f)

    by_cat: dict = {}
    for rec in records:
        cat  = extract_category(rec.get("category", "unknown"))   # ← fixed
        path = rec.get("filepath", "").replace("\\", os.sep)
        if os.path.exists(path):
            by_cat.setdefault(cat, []).append(rec)

    samples = []
    for cat, recs in by_cat.items():
        for rec in recs[:per_category]:
            samples.append({
                "filepath": rec.get("filepath", "").replace("\\", os.sep),
                "category": cat,
                "filename": rec.get("filename", ""),
            })
    return samples


# ── Main evaluation ────────────────────────────────────────────────────────────

async def run_eval(base_url: str, metadata_path: str, top_k: int, images_per_cat: int):
    print(f"\n{'='*64}")
    print(f"  Phase 5 — Retrieval Quality & Similarity Validation")
    print(f"{'='*64}")
    print(f"  API:      {base_url}")
    print(f"  Top-K:    {top_k}")
    print(f"{'='*64}\n")

    if not await health_ok(base_url):
        print("  ERROR: API unreachable. Start your FastAPI server first.")
        return

    probes: list[ProbeResult] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:

        # ── 1. Text probes ─────────────────────────────────────────────────────
        print(f"  Running {len(TEXT_PROBES)} text probes…")
        for tp in TEXT_PROBES:
            try:
                results, ms = await query_text(client, tp["query"], top_k)
                probes.append(ProbeResult(
                    probe      = tp["query"],
                    mode       = "text",
                    expected   = tp["expected_category"],
                    top_k      = top_k,
                    results    = results,
                    latency_ms = ms,
                ))
            except Exception as e:
                probes.append(ProbeResult(
                    probe      = tp["query"],
                    mode       = "text",
                    expected   = tp["expected_category"],
                    top_k      = top_k,
                    results    = [],
                    latency_ms = 0,
                    error      = str(e)[:80],
                ))
            await asyncio.sleep(0.2)

        # ── 2. Image probes ────────────────────────────────────────────────────
        image_samples = sample_images(metadata_path, per_category=images_per_cat)
        if image_samples:
            print(f"  Running {len(image_samples)} image probes ({images_per_cat} per category)…")
            for sample in image_samples:
                try:
                    results, ms = await query_image(client, sample["filepath"], top_k)
                    probes.append(ProbeResult(
                        probe      = sample["filename"],
                        mode       = "image",
                        expected   = sample["category"],
                        top_k      = top_k,
                        results    = results,
                        latency_ms = ms,
                    ))
                except Exception as e:
                    probes.append(ProbeResult(
                        probe      = sample["filename"],
                        mode       = "image",
                        expected   = sample["category"],
                        top_k      = top_k,
                        results    = [],
                        latency_ms = 0,
                        error      = str(e)[:80],
                    ))
                await asyncio.sleep(0.2)
        else:
            print("  No local images found — skipping image probes.")
            print(f"  (Check metadata path: {metadata_path})")

    # ── Compute metrics ────────────────────────────────────────────────────────
    successful = [p for p in probes if p.success]
    failed     = [p for p in probes if not p.success]

    text_probes  = [p for p in successful if p.mode == "text"]
    image_probes = [p for p in successful if p.mode == "image"]

    def avg_precision(ps, k):
        if not ps:
            return None
        return statistics.mean(p.precision_at_k(k) for p in ps)

    def pct(v):
        return f"{v*100:.1f}%" if v is not None else "—"

    metrics = {
        "text": {
            "count":    len(text_probes),
            "p@1":      avg_precision(text_probes, 1),
            "p@3":      avg_precision(text_probes, 3),
            "p@5":      avg_precision(text_probes, 5),
            "p@10":     avg_precision(text_probes, 10),
            "ordered":  sum(1 for p in text_probes if p.scores_are_ordered()),
            "in_range": sum(1 for p in text_probes if p.scores_in_range()),
            "latency_mean": statistics.mean(p.latency_ms for p in text_probes) if text_probes else 0,
        },
        "image": {
            "count":    len(image_probes),
            "p@1":      avg_precision(image_probes, 1),
            "p@3":      avg_precision(image_probes, 3),
            "p@5":      avg_precision(image_probes, 5),
            "p@10":     avg_precision(image_probes, 10),
            "ordered":  sum(1 for p in image_probes if p.scores_are_ordered()),
            "in_range": sum(1 for p in image_probes if p.scores_in_range()),
            "latency_mean": statistics.mean(p.latency_ms for p in image_probes) if image_probes else 0,
        },
    }

    # ── Print report ───────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  RETRIEVAL QUALITY REPORT")
    print(f"{'='*64}\n")

    print(f"  {'Metric':<28} {'Text search':>14} {'Image search':>14}")
    print(f"  {'-'*56}")
    rows = [
        ("Probes run",        str(metrics['text']['count']),   str(metrics['image']['count'])),
        ("Precision@1",       pct(metrics['text']['p@1']),     pct(metrics['image']['p@1'])),
        ("Precision@3",       pct(metrics['text']['p@3']),     pct(metrics['image']['p@3'])),
        ("Precision@5",       pct(metrics['text']['p@5']),     pct(metrics['image']['p@5'])),
        ("Precision@10",      pct(metrics['text']['p@10']),    pct(metrics['image']['p@10'])),
        ("Scores descending", f"{metrics['text']['ordered']}/{metrics['text']['count']}",
                              f"{metrics['image']['ordered']}/{metrics['image']['count']}"),
        ("Scores in range",   f"{metrics['text']['in_range']}/{metrics['text']['count']}",
                              f"{metrics['image']['in_range']}/{metrics['image']['count']}"),
        ("Mean latency",      f"{metrics['text']['latency_mean']:.0f}ms",
                              f"{metrics['image']['latency_mean']:.0f}ms"),
    ]
    for label, tv, iv in rows:
        print(f"  {label:<28} {tv:>14} {iv:>14}")

    print(f"\n  Per-probe detail")
    print(f"  {'-'*64}")
    print(f"  {'Mode':<6} {'Probe':<30} {'Exp cat':<12} {'Top1 cat':<12} {'P@5':>6} {'ms':>6}")
    print(f"  {'-'*64}")
    for p in probes:
        if not p.success:
            print(f"  {'ERR':<6} {p.probe[:30]:<30} {p.expected:<12} {'ERROR':<12}  {'—':>6} {'—':>6}  ← {p.error}")
            continue
        p5   = f"{p.precision_at_k(5)*100:.0f}%"
        ms   = f"{p.latency_ms:.0f}"
        top1 = p.top1_category()
        flag = "" if top1 == p.expected.lower() else "  ← MISMATCH"
        print(f"  {p.mode:<6} {p.probe[:30]:<30} {p.expected:<12} {top1:<12} {p5:>6} {ms:>6}{flag}")

    if failed:
        print(f"\n  Failed probes: {len(failed)}")
        for p in failed:
            print(f"    • [{p.mode}] {p.probe}: {p.error}")

    # ── Interpretation ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  INTERPRETATION GUIDE")
    print(f"{'='*64}")
    print("""
  Precision@K = fraction of top-K results matching the expected category.

  Benchmarks for semantic image search (industry typical):
    P@1  ≥ 70%  → good top-result accuracy
    P@5  ≥ 60%  → good page-1 relevance
    P@10 ≥ 50%  → acceptable recall depth

  Score ordering: results MUST be in descending score order.
  If any probe fails this, there is a bug in the fusion logic.

  Score range: fused scores (weighted cosine) should stay in [0, 2].
  Values outside this range indicate a normalisation issue.
    """)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    report = {
        "config":  {"url": base_url, "top_k": top_k},
        "summary": {
            "total_probes":    len(probes),
            "successful":      len(successful),
            "failed":          len(failed),
            "text_precision":  {k: pct(v) for k, v in metrics["text"].items()  if k.startswith("p@")},
            "image_precision": {k: pct(v) for k, v in metrics["image"].items() if k.startswith("p@")},
        },
        "probes": [
            {
                "probe":      p.probe,
                "mode":       p.mode,
                "expected":   p.expected,
                "top1_cat":   p.top1_category(),
                "p@1":        pct(p.precision_at_k(1)),
                "p@3":        pct(p.precision_at_k(3)),
                "p@5":        pct(p.precision_at_k(5)),
                "latency_ms": round(p.latency_ms),
                "ordered":    p.scores_are_ordered(),
                "in_range":   p.scores_in_range(),
                "error":      p.error,
            }
            for p in probes
        ],
    }
    with open("retrieval_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report saved → retrieval_eval_report.json\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 retrieval quality evaluation")
    parser.add_argument("--url",            default="http://localhost:8000",      help="API base URL")
    parser.add_argument("--metadata",       default="data/metadata_cleaned.json", help="Path to metadata JSON")
    parser.add_argument("--top-k",          type=int, default=10,                 help="Results per query (default 10)")
    parser.add_argument("--images-per-cat", type=int, default=3,                  help="Image probes per category (default 3)")
    args = parser.parse_args()

    asyncio.run(run_eval(args.url, args.metadata, args.top_k, args.images_per_cat))