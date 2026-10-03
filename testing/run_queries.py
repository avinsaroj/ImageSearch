"""
run_queries.py — Custom query retrieval test
Runs 29 real-world jewellery queries and produces a detailed report.

Usage:
    pip install httpx
    python run_queries.py --url http://localhost:8000 --top-k 5
"""

import argparse
import json
import sys
import time
import statistics
from dataclasses import dataclass
from typing import Optional

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)

QUERIES = [
    "elegant gold ring for daily wear",
    "minimalist diamond necklace",
    "heavy bridal jewellery set",
    "simple silver bracelet for office",
    "diamond ring with small stones",
    "emerald pendant in gold",
    "ruby earrings traditional style",
    "sapphire engagement ring",
    "bridal gold choker set",
    "wedding jewellery for lehenga",
    "engagement ring for women diamond",
    "party wear earrings shiny",
    "jewellery for red saree",
    "necklace to match black dress",
    "traditional earrings for kurti",
    "modern jewellery for western outfit",
    "lightweight gold ring under budget",
    "affordable diamond jewellery",
    "simple daily wear chain gold",
    "premium looking necklace",
    "vintage style gold ring",
    "royal looking jewellery set",
    "modern geometric earrings",
    "antique temple jewellery",
    "round diamond ring with gold band",
    "thin gold chain with small pendant",
    "big statement earrings shiny stones",
    "floral design necklace gold",
    "pearl earrings simple elegant",
]

EXPECTED = {
    "elegant gold ring for daily wear":        "ring",
    "minimalist diamond necklace":             "necklace",
    "heavy bridal jewellery set":              "multiple",
    "simple silver bracelet for office":       "bracelet",
    "diamond ring with small stones":          "ring",
    "emerald pendant in gold":                 "necklace",
    "ruby earrings traditional style":         "earring",
    "sapphire engagement ring":                "ring",
    "bridal gold choker set":                  "necklace",
    "wedding jewellery for lehenga":           "multiple",
    "engagement ring for women diamond":       "ring",
    "party wear earrings shiny":               "earring",
    "jewellery for red saree":                 "multiple",
    "necklace to match black dress":           "necklace",
    "traditional earrings for kurti":          "earring",
    "modern jewellery for western outfit":     "multiple",
    "lightweight gold ring under budget":      "ring",
    "affordable diamond jewellery":            "ring",
    "simple daily wear chain gold":            "necklace",
    "premium looking necklace":                "necklace",
    "vintage style gold ring":                 "ring",
    "royal looking jewellery set":             "multiple",
    "modern geometric earrings":               "earring",
    "antique temple jewellery":                "multiple",
    "round diamond ring with gold band":       "ring",
    "thin gold chain with small pendant":      "necklace",
    "big statement earrings shiny stones":     "earring",
    "floral design necklace gold":             "necklace",
    "pearl earrings simple elegant":           "earring",
}


@dataclass
class QueryResult:
    query:      str
    expected:   str
    results:    list
    latency_ms: float
    error:      Optional[str] = None

    @property
    def ok(self):
        return self.error is None and bool(self.results)

    def top1_category(self):
        if not self.results:
            return "—"
        cat = (self.results[0].get("payload") or {}).get("category", "—")
        if isinstance(cat, list):
            cat = cat[0] if cat else "—"
        return str(cat).lower()

    def top1_score(self):
        if not self.results:
            return 0.0
        return self.results[0].get("score", 0.0)

    def precision_at(self, k):
        top = self.results[:k]
        if not top:
            return 0.0
        hits = sum(
            1 for r in top
            if self._extract_category(r) == self.expected.lower()
        )
        return hits / len(top)

    def top_categories(self, k=5):
        return [self._extract_category(r) for r in self.results[:k]]

    @staticmethod
    def _extract_category(result):
        cat = (result.get("payload") or {}).get("category", "—")
        if isinstance(cat, list):
            cat = cat[0] if cat else "—"
        return str(cat).lower()

    def top_filenames(self, k=3):
        return [(r.get("payload") or {}).get("filename", "—") for r in self.results[:k]]

    def top_scores(self, k=5):
        return [round(r.get("score", 0), 4) for r in self.results[:k]]

    def top1_hit(self):
        return self.top1_category() == self.expected.lower()


def run_query_sync(base_url, query, top_k):
    t0 = time.monotonic()
    try:
        with httpx.Client(base_url=base_url, timeout=60) as client:
            r = client.post("/search-image", data={"query": query, "top_k": top_k})
        ms = (time.monotonic() - t0) * 1000
        r.raise_for_status()
        results = r.json().get("results", [])
        return QueryResult(query=query, expected=EXPECTED.get(query, "—"), results=results, latency_ms=ms)
    except Exception as e:
        ms = (time.monotonic() - t0) * 1000
        return QueryResult(query=query, expected=EXPECTED.get(query, "—"), results=[], latency_ms=ms, error=str(e)[:120])


def health_check(base_url):
    try:
        with httpx.Client(base_url=base_url, timeout=5) as c:
            return c.get("/health").status_code == 200
    except Exception as e:
        print(f"  Health check error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",   default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    base_url = args.url
    top_k    = args.top_k

    print(f"\n{'='*68}")
    print(f"  Custom Query Retrieval Test — {len(QUERIES)} queries")
    print(f"{'='*68}")
    print(f"  API:    {base_url}")
    print(f"  Top-K:  {top_k}")
    print(f"{'='*68}\n")

    print("  Checking API health...", end=" ", flush=True)
    if not health_check(base_url):
        print("FAILED")
        print(f"\n  Cannot reach {base_url}")
        print("  Make sure docker-compose is running:\n    docker-compose up -d")
        sys.exit(1)
    print("OK\n")

    print(f"  Running {len(QUERIES)} queries one by one...\n")

    results = []
    for i, query in enumerate(QUERIES, 1):
        r = run_query_sync(base_url, query, top_k)
        results.append(r)

        if not r.ok:
            status = "ERR"
        elif r.top1_hit():
            status = "OK "
        else:
            status = "~~ "

        print(f"  [{status}] #{i:<2} {query[:48]:<48}  {r.latency_ms:>6.0f}ms  top1={r.top1_category()}")

    ok      = [r for r in results if r.ok]
    failed  = [r for r in results if not r.ok]
    hits_p1 = [r for r in ok if r.top1_hit()]
    lats    = [r.latency_ms for r in ok]

    p1_all = statistics.mean(r.precision_at(1) for r in ok) if ok else 0
    p3_all = statistics.mean(r.precision_at(3) for r in ok) if ok else 0
    p5_all = statistics.mean(r.precision_at(5) for r in ok) if ok else 0

    print(f"\n{'='*68}")
    print(f"  SUMMARY")
    print(f"{'='*68}")
    print(f"  Queries run     : {len(results)}")
    print(f"  Successful      : {len(ok)}")
    print(f"  Failed          : {len(failed)}")
    if ok:
        print(f"  Top-1 correct   : {len(hits_p1)} / {len(ok)}  ({len(hits_p1)/len(ok)*100:.1f}%)")
    print(f"\n  Avg Precision@1 : {p1_all*100:.1f}%")
    print(f"  Avg Precision@3 : {p3_all*100:.1f}%")
    print(f"  Avg Precision@5 : {p5_all*100:.1f}%")

    if lats:
        print(f"\n  Latency")
        print(f"  |- min    : {min(lats):.0f}ms")
        print(f"  |- mean   : {statistics.mean(lats):.0f}ms")
        print(f"  |- median : {statistics.median(lats):.0f}ms")
        print(f"  `- max    : {max(lats):.0f}ms")

    print(f"\n{'='*68}")
    print(f"  DETAILED RESULTS")
    print(f"{'='*68}")
    print(f"  {'#':<3} {'Query':<40} {'Exp':<10} {'Top1':<10} {'P@1':>5} {'P@3':>5} {'P@5':>5} {'ms':>6}")
    print(f"  {'-'*68}")

    mismatches = []
    for i, r in enumerate(results, 1):
        if not r.ok:
            print(f"  {i:<3} {r.query[:40]:<40}  ERROR: {r.error[:30]}")
            continue
        p1   = f"{r.precision_at(1)*100:.0f}%"
        p3   = f"{r.precision_at(3)*100:.0f}%"
        p5   = f"{r.precision_at(5)*100:.0f}%"
        flag = "" if r.top1_hit() else "  <- MISMATCH"
        print(f"  {i:<3} {r.query[:40]:<40} {r.expected:<10} {r.top1_category():<10} {p1:>5} {p3:>5} {p5:>5} {r.latency_ms:>6.0f}{flag}")
        if not r.top1_hit():
            mismatches.append(r)

    if mismatches:
        print(f"\n{'='*68}")
        print(f"  MISMATCH DEEP-DIVE  ({len(mismatches)} queries)")
        print(f"{'='*68}")
        for r in mismatches:
            print(f"\n  Query    : \"{r.query}\"")
            print(f"  Expected : {r.expected}")
            print(f"  Got cats : {r.top_categories(top_k)}")
            print(f"  Scores   : {r.top_scores(top_k)}")
            print(f"  Files    : {r.top_filenames(3)}")

    print(f"\n{'='*68}")
    print(f"  TOP-1 CATEGORY DISTRIBUTION")
    print(f"{'='*68}")
    cat_count: dict = {}
    for r in ok:
        c = r.top1_category()
        cat_count[c] = cat_count.get(c, 0) + 1
    for cat, cnt in sorted(cat_count.items(), key=lambda x: -x[1]):
        bar = "#" * cnt
        print(f"  {cat:<14}  {bar:<32}  {cnt}")

    report = {
        "config": {"url": base_url, "top_k": top_k, "total_queries": len(QUERIES)},
        "summary": {
            "successful":         len(ok),
            "failed":             len(failed),
            "top1_accuracy":      f"{len(hits_p1)/len(ok)*100:.1f}%" if ok else "0%",
            "avg_precision_at_1": f"{p1_all*100:.1f}%",
            "avg_precision_at_3": f"{p3_all*100:.1f}%",
            "avg_precision_at_5": f"{p5_all*100:.1f}%",
            "latency_mean_ms":    round(statistics.mean(lats)) if lats else 0,
            "latency_median_ms":  round(statistics.median(lats)) if lats else 0,
        },
        "queries": [
            {
                "query":      r.query,
                "expected":   r.expected,
                "top1_cat":   r.top1_category(),
                "top1_score": round(r.top1_score(), 4),
                "top_cats":   r.top_categories(top_k),
                "top_scores": r.top_scores(top_k),
                "p@1":        f"{r.precision_at(1)*100:.0f}%",
                "p@3":        f"{r.precision_at(3)*100:.0f}%",
                "p@5":        f"{r.precision_at(5)*100:.0f}%",
                "latency_ms": round(r.latency_ms),
                "top1_hit":   r.top1_hit(),
                "error":      r.error,
            }
            for r in results
        ],
    }

    with open("query_test_report_2.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved -> query_test_report.json\n")


if __name__ == "__main__":
    main()