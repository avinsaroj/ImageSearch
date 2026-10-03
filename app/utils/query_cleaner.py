import logging
import os
import re

logger = logging.getLogger(__name__)

# ── Stage 1A: Strip conversational OPENERS from the START ─────────────────────
# Each pattern ends with \s* so the following word is preserved cleanly.
OPENER_PATTERNS = [
    r"^show\s+me\s+(?:something\s+)?(?:like\s+)?",
    r"^find\s+me\s+(?:a\s+|an\s+|some\s+)?",
    r"^find\s+(?:a\s+|an\s+|some\s+)?",
    r"^i\s+want\s+(?:a\s+|an\s+|some\s+)?",
    r"^i\s+need\s+(?:a\s+|an\s+|some\s+)?",
    r"^i\s+am\s+looking\s+for\s+(?:a\s+|an\s+|some\s+)?",
    r"^i'm\s+looking\s+for\s+(?:a\s+|an\s+|some\s+)?",
    r"^looking\s+for\s+(?:a\s+|an\s+|some\s+)?",
    r"^i\s+would\s+like\s+(?:a\s+|an\s+|some\s+)?",
    r"^give\s+me\s+(?:a\s+|an\s+|some\s+)?",
    r"^get\s+me\s+(?:a\s+|an\s+|some\s+)?",
    r"^suggest\s+me\s+(?:a\s+|an\s+|some\s+)?",
    r"^can\s+you\s+find\s+me\s+(?:a\s+|an\s+|some\s+)?",
    r"^can\s+you\s+show\s+me\s+(?:a\s+|an\s+|some\s+)?",
    r"^can\s+you\s+get\s+me\s+(?:a\s+|an\s+|some\s+)?",
    r"^do\s+you\s+have\s+(?:a\s+|an\s+|some\s+)?",
    r"^search\s+for\s+(?:a\s+|an\s+|some\s+)?",
]
_OPENERS = [re.compile(p, re.IGNORECASE) for p in OPENER_PATTERNS]

# ── Stage 1B: Strip TAIL clauses (everything from these words onward) ─────────
TAIL_PATTERNS = [
    r"\s+matching\s+(?:to|with)\b.*$",
    r"\s+that\s+(?:go|goes|match(?:es)?|suit(?:s)?|complement(?:s)?)\b.*$",
    r"\s+to\s+(?:go\s+with|match|suit|pair\s+with)\b.*$",
    r"\s+for\s+(?:pairing|wearing)\s+with\b.*$",
    r"\s+similar\s+to\b.*$",
    # outfit context — strip "for <colour> <garment>"
    r"\s+for\s+(?:my\s+)?(?:red|black|blue|green|white|pink|yellow|purple|"
    r"orange|grey|gray|navy|maroon|beige|cream|gold|silver)\s+"
    r"(?:saree|lehenga|kurti|dress|outfit|suit|gown|top|shirt|blouse)\b.*$",
    r"\s+under\s+(?:budget|my\s+budget|\d+[kK]?)\b.*$",
    # "for office / for party / for daily" — context-only qualifiers
    r"\s+for\s+(?:office|party|daily\s+wear|work|college|gym)\b.*$",
]
_TAILS = [re.compile(p, re.IGNORECASE) for p in TAIL_PATTERNS]

# ── Stage 1C: Remove isolated filler words/phrases ───────────────────────────
INLINE_FILLERS = [
    r"\baffordable\b\s*",
    r"\bpremium[- ]?looking\b\s*",
    r"\bthat\s+looks?\s+premium\b\s*",
    r"\b(a|an|the|some|any)\b\s+",   # stray articles
]
_INLINE = [re.compile(p, re.IGNORECASE) for p in INLINE_FILLERS]

# ── Stage 1D: Metal normalisation ─────────────────────────────────────────────
METALS = {
    r"\byellow\s+gold\b": "gold",
    r"\brose\s+gold\b":   "rose gold",
    r"\bwhite\s+gold\b":  "white gold",
}
_METALS = [(re.compile(p, re.IGNORECASE), norm) for p, norm in METALS.items()]


def _rule_based_clean(query: str) -> str:
    q = query.strip()

    # 1A — strip openers (apply all, longest match wins naturally via ordering)
    for pattern in _OPENERS:
        before = q
        q = pattern.sub("", q)
        if q != before:
            break   # only strip one opener per query

    # 1B — strip tail clauses
    for pattern in _TAILS:
        q = pattern.sub("", q)

    # 1C — strip inline fillers
    for pattern in _INLINE:
        q = pattern.sub(" ", q)

    # 1D — normalise metals
    for pattern, norm in _METALS:
        q = pattern.sub(norm, q)

    # Collapse whitespace + trailing punctuation
    q = re.sub(r"\s{2,}", " ", q).strip().rstrip(".,!?;:")

    return q


def _llm_clean(query: str) -> str:
    """
    Use Claude Haiku to extract the core search intent.
    Only called when rule-based result is still longer than 6 words.
    Requires ANTHROPIC_API_KEY in environment.
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg    = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 30,
            messages   = [{
                "role":    "user",
                "content": (
                    "Extract the core jewellery search intent from this query as a "
                    "short 2-5 word phrase for image search. "
                    "Return ONLY the short phrase, nothing else.\n\n"
                    f'Query: "{query}"'
                ),
            }],
        )
        result = msg.content[0].text.strip().lower().rstrip(".,!?")
        logger.info("LLM cleaned: %r -> %r", query, result)
        return result
    except Exception as e:
        logger.warning("LLM query cleaning failed: %s", e)
        return None


def clean_query(query: str, use_llm: bool = False) -> str:
    """
    Clean a natural language jewellery search query into a short,
    CLIP-friendly phrase.

    Args:
        query:   Raw user input, e.g. "Show me gold rings matching to necklaces"
        use_llm: If True and ANTHROPIC_API_KEY is set, use Claude Haiku as a
                 fallback when rule-based result is still too long (>5 words).

    Returns:
        Cleaned query string, e.g. "gold rings"
    """
    if not query or not query.strip():
        return query

    original = query.strip()
    cleaned  = _rule_based_clean(original)

    # Optional LLM fallback for stubborn long queries
    if use_llm and len(cleaned.split()) > 5:
        llm_result = _llm_clean(original)
        if llm_result:
            cleaned = llm_result

    if cleaned != original:
        logger.info("Query cleaned: %r -> %r", original, cleaned)

    return cleaned or original


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TESTS = [
        ("Show me gold rings matching to the gold necklaces", "gold rings"),
        ("Find me a minimalist diamond necklace",             "minimalist diamond necklace"),
        ("I want simple silver bracelet for office",          "simple silver bracelet"),
        ("ruby earrings traditional style",                   "ruby earrings traditional style"),
        ("jewellery for red saree",                           "jewellery for red saree"),
        ("show me something like pearl earrings",             "pearl earrings"),
        ("necklace to match black dress",                     "necklace"),
        ("I am looking for an affordable diamond ring",       "diamond ring"),
        ("give me vintage style gold ring",                   "vintage style gold ring"),
        ("lightweight gold ring under budget",                "lightweight gold ring"),
        ("wedding jewellery for lehenga",                     "wedding jewellery for lehenga"),
        ("big statement earrings with shiny stones",          "big statement earrings with shiny stones"),
        ("thin gold chain with small pendant",                "thin gold chain with small pendant"),
        ("bridal gold choker set",                            "bridal gold choker set"),
        ("can you find me antique temple jewellery",          "antique temple jewellery"),
    ]

    print(f"\n{'='*74}")
    print(f"  Query Cleaner — Self Test")
    print(f"{'='*74}")
    print(f"  {'Original':<48}  {'Cleaned':<26}  Result")
    print(f"  {'-'*74}")

    passed = 0
    for original, expected in TESTS:
        cleaned = clean_query(original)
        ok      = cleaned.lower() == expected.lower()
        if ok:
            passed += 1
        result = "PASS" if ok else f"FAIL — expected: {expected!r}"
        print(f"  {original[:48]:<48}  {cleaned[:26]:<26}  {result}")

    print(f"\n  {passed}/{len(TESTS)} tests passed\n")