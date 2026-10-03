"""

recaption.py — Regenerate richer captions for jewellery images using BLIP-2.

WHAT IT DOES
------------
1. Loads metadata_cleaned.json (your existing metadata)
2. For each image, runs BLIP-2 TWICE:
     Pass 1 — "Describe this jewellery piece in detail."  → visual description
     Pass 2 — Q&A prompts for occasion, style, material   → structured tags
3. Merges everything into a rich natural-language caption
4. Writes metadata_recaptioned.json — ready to re-ingest into Qdrant.

CAPTION FORMAT
--------------
  "<description>, <material>, <style> style, suitable for <occasion>"

  Example:
    "a gold bracelet with small heart charm, yellow gold, classic style,
     suitable for valentines and anniversaries"

NOTE: Pass 2 (Q&A) only works with BLIP-2. If you use --model blip,
      occasion/style fall back to keyword rules automatically.

USAGE
-----
  python recaption.py --images_dir data/images --metadata metadata_cleaned.json
  python recaption.py --images_dir data/images --metadata metadata_cleaned.json --model blip
  python recaption.py --images_dir data/images --metadata metadata_cleaned.json --resume
  python recaption.py --images_dir data/images --metadata metadata_cleaned.json --limit 100
  python recaption.py --images_dir data/images --metadata metadata_cleaned.json --debug

"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── BLIP-2 Q&A prompts ───────────────────────────────────────────────────────
# Asked directly to the model with the image in context (Pass 2).
# max_new_tokens=20 is enough — we only need a short label back.

QA_PROMPTS = {
    "occasion": (
        "Question: What occasion or event is this jewellery most suitable for? "
        "Choose from: wedding, party, office, casual, festive, engagement, anniversary, everyday. Answer:"
    ),
}

# ── Normalisation maps — clean up free-form BLIP-2 Q&A answers ───────────────
# Key = substring to match in lowercased answer, Value = canonical label.

OCCASION_NORMALISE = {
    "wed":        "weddings and bridal ceremonies",
    "bridal":     "weddings and bridal ceremonies",
    "bride":      "weddings and bridal ceremonies",
    "engagement": "engagements and anniversaries",
    "anniversar": "engagements and anniversaries",
    "valentine":  "engagements and anniversaries",
    "party":      "parties and evening events",
    "cocktail":   "parties and evening events",
    "evening":    "parties and evening events",
    "festive":    "festive and religious occasions",
    "festival":   "festive and religious occasions",
    "religious":  "festive and religious occasions",
    "puja":       "festive and religious occasions",
    "office":     "everyday and office wear",
    "work":       "everyday and office wear",
    "profession": "everyday and office wear",
    "everyday":   "everyday and office wear",
    "daily":      "everyday and office wear",
    "casual":     "casual and outdoor occasions",
    "outdoor":    "casual and outdoor occasions",
    "beach":      "casual and outdoor occasions",
}

STYLE_NORMALISE = {
    "tradit":   "traditional",
    "ethnic":   "traditional",
    "classic":  "classic",
    "minimal":  "minimalist",
    "simple":   "minimalist",
    "delicate": "minimalist",
    "statement":"statement",
    "bold":     "statement",
    "chunky":   "statement",
    "boho":     "boho",
    "bohemian": "boho",
    "fine":     "fine jewellery",
    "luxury":   "fine jewellery",
    "nature":   "nature-inspired",
    "floral":   "nature-inspired",
    "glamor":   "glamorous",
    "glam":     "glamorous",
    "sparkl":   "glamorous",
}

# ── Keyword fallback rules (used when BLIP answer is unusable or model=blip) ──

OCCASION_RULES = [
    ({"bridal", "wedding", "mangalsutra", "choker", "maang tikka", "nath"},
     "weddings and bridal ceremonies"),
    ({"temple", "deity", "antique", "oxidised", "oxidized", "tribal", "festive"},
     "festive and religious occasions"),
    ({"party", "cocktail", "statement", "bold", "chunky", "diamond"},
     "parties and evening events"),
    ({"office", "minimal", "minimalist", "delicate", "subtle", "everyday", "chain"},
     "everyday and office wear"),
    ({"casual", "beach", "boho", "bohemian", "thread", "jute", "beaded", "tassel"},
     "casual and outdoor occasions"),
    ({"heart", "engagement", "anniversary", "solitaire"},
     "engagements and anniversaries"),
]

STYLE_RULES = [
    ({"temple", "antique", "oxidised", "filigree", "jadau", "kundan", "polki", "meenakari"},
     "traditional"),
    ({"minimal", "delicate", "thin", "solitaire", "geometric", "chain"},
     "minimalist"),
    ({"statement", "bold", "chunky", "layered"},
     "statement"),
    ({"boho", "bohemian", "thread", "jute", "beaded", "tassel"},
     "boho"),
    ({"diamond", "tennis", "eternity", "halo", "18k"},
     "fine jewellery"),
    ({"floral", "leaf", "butterfly", "bird", "peacock", "nature"},
     "nature-inspired"),
    ({"cocktail", "glamour", "crystal", "sparkle"},
     "glamorous"),
]

MATERIAL_RULES = [
    ({"diamond"},         "diamond-studded"),
    ({"ruby"},            "ruby-studded"),
    ({"emerald"},         "emerald-studded"),
    ({"sapphire"},        "sapphire-studded"),
    ({"pearl"},           "pearl"),
    ({"kundan"},          "kundan"),
    ({"polki"},           "polki"),
    ({"beaded", "beads"}, "beaded"),
    ({"enamel", "meenakari"}, "enamel"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(answer: str, normalise_map: dict, fallback: str) -> str:
    """Substring-match a BLIP answer against normalise_map; return fallback if none match."""
    ans = answer.lower().strip()
    ans = re.sub(r"^(it is|this is|i think|probably|maybe|the|a|an)\s+", "", ans)
    for key, label in normalise_map.items():
        if key in ans:
            return label
    return fallback


def _keyword_match(text: str, rules: list) -> str | None:
    text_lower = text.lower()
    for keywords, label in rules:
        if any(kw in text_lower for kw in keywords):
            return label
    return None


# ── BLIP-2 Q&A (Pass 2) ───────────────────────────────────────────────────────

def ask_blip2(image: Image.Image, question: str, processor, model, device: str) -> str:
    """Ask a single Q&A prompt against the image; return the raw short answer."""
    inputs = processor(image, text=question, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=20,
            num_beams=3,
            repetition_penalty=1.2,
        )
    answer = processor.decode(out[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return answer.strip()


def infer_tags_blip2(image, processor, model, device, caption, item):
    """
    Three Q&A passes against the image.
    Falls back to keyword rules if the model's answer is too vague.
    Returns (material, style, occasion).
    """
    metal      = item.get("metal", "")
    metal_conf = item.get("metal_confidence", 0.0)
    category   = item.get("category", "")

    # Occasion
    raw_occasion = ask_blip2(image, QA_PROMPTS["occasion"], processor, model, device)
    logger.debug("  [Q&A] occasion raw: %r", raw_occasion)
    occasion = _normalise(raw_occasion, OCCASION_NORMALISE, "__unknown__")
    if occasion == "__unknown__":
        occasion = _keyword_match(f"{caption} {category}", OCCASION_RULES) or "special occasions"

    # Style
    raw_style = ask_blip2(image, QA_PROMPTS["style"], processor, model, device)
    logger.debug("  [Q&A] style raw:    %r", raw_style)
    style = _normalise(raw_style, STYLE_NORMALISE, "__unknown__")
    if style == "__unknown__":
        style = _keyword_match(f"{caption} {category}", STYLE_RULES) or "classic"

    # Material
    raw_material = ask_blip2(image, QA_PROMPTS["material"], processor, model, device)
    logger.debug("  [Q&A] material raw: %r", raw_material)
    # Accept BLIP answer if it's concise (≤5 words)
    material = raw_material if len(raw_material.split()) <= 5 else ""

    # Enrich/override with structured metal + gem detection
    gem = _keyword_match(caption, MATERIAL_RULES)
    if metal and metal_conf >= 0.80:
        if gem and gem.lower() not in material.lower():
            material = f"{metal}, {gem}"
        elif not material or material.lower() in ("metal", "metal alloy", ""):
            material = metal
    elif not material:
        material = gem or _keyword_match(raw_material, MATERIAL_RULES) or "metal alloy"

    return material, style, occasion


def infer_tags_keywords(caption, item):
    """Keyword-only fallback used when model=blip (no Q&A support)."""
    metal      = item.get("metal", "")
    metal_conf = item.get("metal_confidence", 0.0)
    category   = item.get("category", "")
    combined   = f"{caption} {category} {metal}"

    gem      = _keyword_match(caption, MATERIAL_RULES)
    material = f"{metal}, {gem}" if (gem and metal and metal_conf >= 0.80) else \
               (gem or (metal if metal_conf >= 0.80 else "metal alloy"))
    style    = _keyword_match(combined, STYLE_RULES)    or "classic"
    occasion = _keyword_match(combined, OCCASION_RULES) or "special occasions"
    return material, style, occasion


# ── Caption assembly ──────────────────────────────────────────────────────────

def _build_rich_caption(blip_caption, material, style, occasion):
    description = blip_caption.strip().rstrip(".")
    parts = [description]
    if material.lower() not in description.lower():
        parts.append(material)
    parts.append(f"{style} style")
    parts.append(f"suitable for {occasion}")
    return ", ".join(parts)


def _clean_blip_output(text):
    text = re.sub(r'\b(\w+) \1\b', r'\1', text)
    text = text.rsplit(" ", 1)[0] if len(text) > 80 and text[-1] not in ".!?" else text
    return text.strip()


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_blip2(device):
    model_id = "Salesforce/blip2-opt-2.7b"
    logger.info("Loading BLIP-2 from %s ...", model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    if device == "cuda":
        try:
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_id, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16)
            logger.info("BLIP-2 loaded with 8-bit quantization on GPU")
        except Exception:
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16).to(device)
            logger.info("BLIP-2 loaded in fp16 on GPU")
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
        logger.info("BLIP-2 loaded on CPU (expect ~3-6s per image)")
    return processor, model


def load_blip(device):
    model_id = "Salesforce/blip-image-captioning-large"
    logger.info("Loading BLIP (large) from %s ...", model_id)
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    logger.info("BLIP loaded on %s", device)
    return processor, model


# ── Caption generation (Pass 1) ───────────────────────────────────────────────

def generate_caption_blip2(image, processor, model, device):
    prompt = "Question: Describe this jewellery piece in detail. Answer:"
    inputs = processor(image, text=prompt, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, num_beams=3, repetition_penalty=1.3)
    caption = processor.decode(out[0], skip_special_tokens=True)
    if "Answer:" in caption:
        caption = caption.split("Answer:")[-1].strip()
    return _clean_blip_output(caption)


def generate_caption_blip(image, processor, model, device):
    inputs = processor(image, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, num_beams=3, repetition_penalty=1.3)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return _clean_blip_output(caption)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--metadata",   required=True)
    parser.add_argument("--output",     default="metadata_recaptioned.json")
    parser.add_argument("--model",      choices=["blip2", "blip"], default="blip2")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--debug",      action="store_true",
                        help="Print raw BLIP-2 Q&A answers for each image")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    with open(args.metadata) as f:
        metadata = json.load(f)
    logger.info("Loaded %d items from %s", len(metadata), args.metadata)

    already_done = {}
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            existing = json.load(f)
        already_done = {item["filename"]: item for item in existing}
        logger.info("Resuming — %d items already processed", len(already_done))

    if args.model == "blip2":
        processor, model = load_blip2(device)
        gen_fn = generate_caption_blip2
        use_qa = True
    else:
        processor, model = load_blip(device)
        gen_fn = generate_caption_blip
        use_qa = False
        logger.info("BLIP (base) selected — occasion/style will use keyword rules")

    model.eval()

    items_to_process = metadata[:args.limit] if args.limit else metadata
    results = []
    skipped = failed = 0
    t_start = time.monotonic()

    for item in tqdm(items_to_process, desc="Captioning", unit="img"):
        filename = item["filename"]

        if filename in already_done:
            results.append(already_done[filename])
            skipped += 1
            continue

        img_path = images_dir / item.get("filepath", "").lstrip("/")
        if not img_path.exists():
            matches = list(images_dir.rglob(filename))
            if not matches:
                logger.warning("Image not found: %s — keeping original caption", filename)
                results.append(item)
                failed += 1
                continue
            img_path = matches[0]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning("Cannot open %s: %s", img_path, e)
            results.append(item)
            failed += 1
            continue

        try:
            # Pass 1 — visual description
            blip_caption = gen_fn(image, processor, model, device)

            # Pass 2 — Q&A tags or keyword fallback
            if use_qa:
                material, style, occasion = infer_tags_blip2(
                    image, processor, model, device, blip_caption, item)
            else:
                material, style, occasion = infer_tags_keywords(blip_caption, item)

            rich_caption = _build_rich_caption(blip_caption, material, style, occasion)

        except Exception as e:
            logger.warning("Failed on %s: %s — keeping original caption", filename, e)
            results.append(item)
            failed += 1
            continue

        results.append({
            "filename":         item["filename"],
            "filepath":         item["filepath"],
            "category":         item["category"],
            "metal":            item["metal"],
            "metal_confidence": item["metal_confidence"],
            "material":         material,
            "style":            style,
            "occasion":         occasion,
            "caption":          rich_caption,
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed   = time.monotonic() - t_start
    processed = len(results) - skipped - failed
    logger.info("─" * 60)
    logger.info("Done in %.1fs  |  Processed: %d  Skipped: %d  Failed: %d",
                elapsed, processed, skipped, failed)
    logger.info("Output: %s", args.output)
    logger.info("─" * 60)

    # ── Quality comparison ────────────────────────────────────────────────────
    import random
    original_map = {i["filename"]: i["caption"] for i in metadata}
    samples = [r for r in results if r["filename"] not in already_done and "material" in r]
    print("\n── Caption quality comparison (5 samples) ──────────────────────")
    for r in random.sample(samples, min(5, len(samples))):
        print(f"\n  File     : {r['filename']}")
        print(f"  Before   : {original_map.get(r['filename'], '—')}")
        print(f"  After    : {r['caption']}")
        print(f"  Material : {r['material']}")
        print(f"  Style    : {r['style']}")
        print(f"  Occasion : {r['occasion']}")
    print()


if __name__ == "__main__":
    main()