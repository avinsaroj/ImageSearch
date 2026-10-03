"""
generate_metadata_v3.py

Production metadata generator — calibrated to the actual data/images folder layout:

    data/images/
    ├── bracelet/              ← flat (no subfolders), product images
    ├── earring/
    │   ├── diamond/
    │   ├── drops/
    │   ├── gold/
    │   ├── pearl/
    │   └── studs/
    ├── lifestyle/             ← flat, is_product_image = False
    ├── necklaces/
    │   ├── diamond/
    │   ├── gemstone/
    │   ├── gold/
    │   └── pearl/
    ├── rings/
    │   ├── diamond/
    │   ├── gemstone/
    │   ├── gold/
    │   ├── pearl/
    │   ├── platinum/
    │   └── solitaire/
    └── watches/
        ├── men/
        └── women/

What this script does:
  1. Walks the entire data/images tree
  2. Derives category, gemstone, metal, style, and audience directly from the
     folder path — no CLIP needed for fields the folder already tells us
  3. Uses CLIP zero-shot only to fill in fields the folder CANNOT tell us
     (metal for earring/rings/necklaces subfolders that use gem names, etc.)
  4. Generates 3 diverse caption variants per image
  5. Sets is_product_image=False for everything under lifestyle/
  6. Saves metadata.json

Bugs fixed vs original generate_metadata.py:
  - gemstone subfolder named 'gemstone' was being stored as gemstone value
  - 'men'/'women' watch subfolders were leaking into gemstone field
  - 'drops'/'studs'/'solitaire' are now recognised as jewellery styles, not gems
  - All captions followed one rigid template; now 3 diverse variants per image
  - lifestyle/ images had no is_product_image flag

Usage:
    pip install torch transformers Pillow tqdm open_clip_torch

    # Full run with CLIP predictions
    python generate_metadata_v3.py --image-dir data/images --output metadata.json

    # Fast run — folder/filename metadata only, no AI
    python generate_metadata_v3.py --image-dir data/images --skip-ai

    # Dry run — prints everything, saves nothing
    python generate_metadata_v3.py --image-dir data/images --dry-run

    # Test on first 40 images
    python generate_metadata_v3.py --image-dir data/images --limit 40 --dry-run
"""

import json
import logging
import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("metadata_v3")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_MODEL_ID = "patrickjohncyh/fashion-clip"

# ---------------------------------------------------------------------------
# FOLDER → CANONICAL CATEGORY
# Covers every top-level folder visible in the screenshots.
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    "bracelet":  "bracelet",
    "bracelets": "bracelet",
    "earring":   "earring",
    "earrings":  "earring",
    "necklace":  "necklace",
    "necklaces": "necklace",
    "ring":      "ring",
    "rings":     "ring",
    "watches":   "watches",
    "watch":     "watches",
    "pendant":   "pendant",
    "pendants":  "pendant",
    "bangle":    "bangle",
    "bangles":   "bangle",
    # lifestyle folder → category stays "multiple", is_product_image = False
    "lifestyle": "multiple",
    "multiple":  "multiple",
    "styled":    "multiple",
}

# Top-level folders where is_product_image should be False
LIFESTYLE_FOLDERS = {"lifestyle", "multiple", "styled"}

# ---------------------------------------------------------------------------
# SUBFOLDER INTERPRETATION TABLE
#
# Each entry maps (parent_category, subfolder_name) → dict of field overrides.
# This is the single source of truth for what each subfolder means.
#
# Fields you can set: gemstone, metal, style, audience
# ---------------------------------------------------------------------------
SUBFOLDER_RULES: dict[tuple[str, str], dict] = {

    # ── earring subfolders ──────────────────────────────────────────────────
    ("earring", "diamond"):  {"gemstone": "diamond"},
    ("earring", "drops"):    {"style": "drops"},          # style, NOT a gemstone
    ("earring", "gold"):     {"metal": "yellow gold"},
    ("earring", "pearl"):    {"gemstone": "pearl"},
    ("earring", "studs"):    {"style": "studs"},          # style, NOT a gemstone

    # ── necklaces subfolders ────────────────────────────────────────────────
    ("necklace", "diamond"): {"gemstone": "diamond"},
    ("necklace", "gemstone"): {},                         # generic — let CLIP decide the gem
    ("necklace", "gold"):    {"metal": "yellow gold"},
    ("necklace", "pearl"):   {"gemstone": "pearl"},

    # ── rings subfolders ────────────────────────────────────────────────────
    ("ring", "diamond"):     {"gemstone": "diamond"},
    ("ring", "gemstone"):    {},                          # generic — CLIP decides
    ("ring", "gold"):        {"metal": "yellow gold"},
    ("ring", "pearl"):       {"gemstone": "pearl"},
    ("ring", "platinum"):    {"metal": "platinum"},
    ("ring", "solitaire"):   {"style": "solitaire"},      # style, NOT a gemstone

    # ── watches subfolders ──────────────────────────────────────────────────
    ("watches", "men"):      {"audience": "men"},
    ("watches", "women"):    {"audience": "women"},

    # ── bracelet (flat — no subfolders in screenshot) ───────────────────────
    # Nothing to add; CLIP will predict metal/gemstone.
}

# ---------------------------------------------------------------------------
# CLIP zero-shot label sets
# ---------------------------------------------------------------------------
METAL_LABELS = [
    "yellow gold jewellery",
    "white gold jewellery",
    "rose gold jewellery",
    "silver jewellery",
    "platinum jewellery",
]
METAL_NAMES = ["yellow gold", "white gold", "rose gold", "silver", "platinum"]

GEMSTONE_LABELS = [
    "diamond jewellery",
    "ruby jewellery",
    "emerald jewellery",
    "sapphire jewellery",
    "pearl jewellery",
    "jewellery with no gemstone",
]
GEMSTONE_NAMES = ["diamond", "ruby", "emerald", "sapphire", "pearl", "none"]

PHOTO_TYPE_LABELS = [
    "a jewellery product photo on a plain white background",
    "a model or person wearing jewellery",
]

CONFIDENCE_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# CLIP Classifier
# ---------------------------------------------------------------------------
class CLIPClassifier:
    def __init__(self, model_path: Optional[str] = None):
        source = model_path if (model_path and Path(model_path).exists()) else DEFAULT_MODEL_ID
        logger.info(f"Loading CLIP from '{source}' ...")
        self._model     = CLIPModel.from_pretrained(source)
        self._processor = CLIPProcessor.from_pretrained(source)
        self._model.eval()
        logger.info("CLIP ready.")

    def classify(self, image: Image.Image, labels: list[str]) -> tuple[int, float, list[float]]:
        """Returns (best_idx, best_score, all_scores)."""
        with torch.inference_mode():
            inputs = self._processor(text=labels, images=image,
                                     return_tensors="pt", padding=True)
            logits = self._model(**inputs).logits_per_image
            probs  = F.softmax(logits, dim=-1).squeeze(0).tolist()
        best_idx = probs.index(max(probs))
        return best_idx, probs[best_idx], probs


# ---------------------------------------------------------------------------
# Folder metadata parser — uses SUBFOLDER_RULES
# ---------------------------------------------------------------------------
def parse_folder_metadata(img_path: Path, image_dir: Path) -> dict:
    """
    Walk the path parts relative to image_dir and return a metadata dict.

    Returns keys: category, gemstone, metal, style, audience, is_product_image
    All values default to "unknown" (or False for is_product_image).
    """
    result = {
        "category":         "unknown",
        "gemstone":         "unknown",
        "metal":            "unknown",
        "style":            "unknown",
        "audience":         "unknown",
        "is_product_image": True,
    }

    try:
        rel_parts = img_path.relative_to(image_dir).parts   # e.g. ('rings', 'gold', 'ring_001.jpg')
    except ValueError:
        return result

    if len(rel_parts) < 2:
        return result   # image sitting directly in image_dir root — skip

    top_folder = rel_parts[0].lower()

    # ── is_product_image ────────────────────────────────────────────────────
    if top_folder in LIFESTYLE_FOLDERS:
        result["is_product_image"] = False

    # ── category from top-level folder ──────────────────────────────────────
    result["category"] = CATEGORY_MAP.get(top_folder, top_folder)

    # ── subfolder rules ─────────────────────────────────────────────────────
    if len(rel_parts) >= 3:
        sub_folder = rel_parts[1].lower()
        rule_key   = (result["category"], sub_folder)
        overrides  = SUBFOLDER_RULES.get(rule_key, None)

        if overrides is not None:
            result.update(overrides)
        else:
            # Subfolder not in the rules table — log it so you can add it
            logger.debug(
                f"Unknown subfolder rule: ({result['category']}, {sub_folder}) "
                f"— CLIP will fill in metal/gemstone for {img_path.name}"
            )

    return result


# ---------------------------------------------------------------------------
# Caption generator — 3 diverse variants
# ---------------------------------------------------------------------------
def make_captions(
    category: str,
    metal:    str,
    gemstone: str,
    style:    str,
    audience: str,
) -> list[str]:
    cat = category if category not in ("unknown", "none", "multiple") else "jewellery"
    met = metal    if metal    not in ("unknown", "none") else ""
    gem = gemstone if gemstone not in ("unknown", "none") else ""
    sty = style    if style    not in ("unknown", "none") else ""
    aud = audience if audience not in ("unknown", "none") else ""

    captions = []

    # Variant 1: original-style (backward compatible with existing ingest code)
    parts = []
    if gem: parts.append(gem)
    parts.append(cat)
    if met: parts.append(f"in {met}")
    if sty: parts.append(f"{sty} style")
    parts.append("jewellery")
    captions.append(" ".join(parts))

    # Variant 2: natural noun phrase
    if gem and met:
        captions.append(f"{met} {cat} with {gem}")
    elif gem:
        captions.append(f"{cat} with {gem} stone")
    elif met:
        captions.append(f"{met} {cat}")
    elif sty:
        captions.append(f"{sty} {cat}")
    else:
        captions.append(f"{cat} jewellery piece")

    # Variant 3: descriptive / searchable phrase
    if aud:
        suffix = f" for {aud}"
    else:
        suffix = ""

    if gem and met:
        captions.append(f"{gem} set {cat} crafted in {met}{suffix}")
    elif gem:
        captions.append(f"{gem} {cat}{suffix}")
    elif met:
        captions.append(f"{cat} made in {met}{suffix}")
    elif sty:
        captions.append(f"fine {sty} {cat}{suffix}")
    else:
        captions.append(f"fine {cat}{suffix}")

    return captions


# ---------------------------------------------------------------------------
# Main metadata generator
# ---------------------------------------------------------------------------
def generate_metadata(
    image_dir:  str | Path,
    output:     str | Path    = "metadata.json",
    model_path: Optional[str] = None,
    limit:      Optional[int] = None,
    dry_run:    bool           = False,
    skip_ai:    bool           = False,
) -> list[dict]:
    """
    Full pipeline — folder-first metadata with CLIP fill-in for unknown fields.

    Parameters
    ----------
    image_dir  : root image directory (e.g. data/images)
    output     : output path for metadata.json
    model_path : optional fine-tuned CLIP model directory
    limit      : cap total images processed (for testing)
    dry_run    : print everything, save nothing
    skip_ai    : skip CLIP predictions entirely (folder metadata only)
    """
    image_dir = Path(image_dir)
    output    = Path(output)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # ── Load CLIP ────────────────────────────────────────────────────────────
    classifier = None
    if not skip_ai:
        classifier = CLIPClassifier(model_path=model_path)

    # ── Discover images ──────────────────────────────────────────────────────
    all_paths = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    )
    if limit:
        all_paths = all_paths[:limit]
    logger.info(f"Discovered {len(all_paths)} images under '{image_dir}'")

    # ── Process ──────────────────────────────────────────────────────────────
    records:         list[dict]  = []
    category_counts: Counter     = Counter()
    metal_counts:    Counter     = Counter()
    gemstone_counts: Counter     = Counter()
    skipped                      = 0

    for img_path in tqdm(all_paths, desc="Generating metadata"):

        # Step A: folder-derived fields
        meta             = parse_folder_metadata(img_path, image_dir)
        category         = meta["category"]
        gemstone         = meta["gemstone"]
        metal            = meta["metal"]
        style            = meta["style"]
        audience         = meta["audience"]
        is_product_image = meta["is_product_image"]
        metal_confidence = 0.0

        # Step B: CLIP fill-in for fields the folder couldn't tell us
        if classifier:
            try:
                img = Image.open(img_path).convert("RGB")

                # Predict metal only when folder gave no metal hint
                if metal == "unknown":
                    m_idx, m_conf, _ = classifier.classify(img, METAL_LABELS)
                    if m_conf >= CONFIDENCE_THRESHOLD:
                        metal            = METAL_NAMES[m_idx]
                        metal_confidence = round(m_conf, 4)

                # Predict gemstone only when folder gave no gemstone hint
                if gemstone == "unknown":
                    g_idx, g_conf, _ = classifier.classify(img, GEMSTONE_LABELS)
                    if g_conf >= CONFIDENCE_THRESHOLD:
                        gemstone = GEMSTONE_NAMES[g_idx]

                # Confirm product vs lifestyle (only for non-lifestyle folders)
                if is_product_image:
                    pt_idx, pt_conf, _ = classifier.classify(img, PHOTO_TYPE_LABELS)
                    is_product_image   = (pt_idx == 0)

            except UnidentifiedImageError:
                logger.warning(f"Corrupt image skipped: {img_path.name}")
                skipped += 1
                continue
            except OSError as e:
                logger.warning(f"OS error on {img_path.name}: {e}")
                skipped += 1
                continue
            except Exception as e:
                logger.error(f"Unexpected error on {img_path.name}: {e}")
                skipped += 1
                continue

        # Step C: captions
        captions     = make_captions(category, metal, gemstone, style, audience)
        main_caption = captions[0]

        record = {
            "filename":          img_path.name,
            "filepath":          str(img_path),
            "sku":               img_path.stem,
            "category":          category,
            "metal":             metal,
            "metal_confidence":  metal_confidence,
            "gemstone":          gemstone,
            "style":             style,       # e.g. "studs", "drops", "solitaire"
            "audience":          audience,    # e.g. "men", "women" (watches)
            "caption":           main_caption,
            "caption_variants":  captions,
            "is_product_image":  is_product_image,
        }
        records.append(record)

        category_counts[category] += 1
        metal_counts[metal]       += 1
        gemstone_counts[gemstone] += 1

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Metadata Generation Complete  (v3 — folder-structure-aware)")
    print(f"{'='*65}")
    print(f"  Images processed : {len(records)}")
    print(f"  Images skipped   : {skipped}")
    print(f"\n  Category breakdown:")
    for k, v in category_counts.most_common():
        bar = "█" * min(30, v // max(1, max(category_counts.values()) // 30))
        print(f"    {k:<20} {v:>5}  {bar}")
    print(f"\n  Metal breakdown:")
    for k, v in metal_counts.most_common():
        print(f"    {k:<20} {v:>5}")
    print(f"\n  Gemstone breakdown:")
    for k, v in gemstone_counts.most_common():
        print(f"    {k:<20} {v:>5}")

    lifestyle_count = sum(1 for r in records if not r["is_product_image"])
    print(f"\n  Product images   : {len(records) - lifestyle_count}")
    print(f"  Lifestyle images : {lifestyle_count}")

    if records:
        print(f"\n  Sample record (first image):")
        print(json.dumps(records[0], indent=4))
    print(f"{'='*65}\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    if dry_run:
        print(f"  DRY RUN — nothing saved. Remove --dry-run to write '{output}'")
    else:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(records)} records → {output}")

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate metadata.json from data/images folder structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with CLIP metal/gemstone prediction
  python generate_metadata_v3.py --image-dir data/images --output metadata.json

  # Fast run — folder metadata only, no AI (good for a first check)
  python generate_metadata_v3.py --image-dir data/images --skip-ai

  # Dry run — prints everything, saves nothing
  python generate_metadata_v3.py --image-dir data/images --dry-run

  # Test on 40 images
  python generate_metadata_v3.py --image-dir data/images --limit 40 --dry-run

  # Use a fine-tuned model instead of FashionCLIP
  python generate_metadata_v3.py --image-dir data/images --model-path models/my_clip
        """
    )
    parser.add_argument("--image-dir",   type=str, required=True,
                        help="Root image directory, e.g. data/images")
    parser.add_argument("--output",      type=str, default="metadata.json",
                        help="Output JSON path (default: metadata.json)")
    parser.add_argument("--model-path",  type=str, default=None,
                        help="Fine-tuned CLIP model directory (optional)")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Process only first N images (for testing)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print everything, save nothing")
    parser.add_argument("--skip-ai",     action="store_true",
                        help="Skip CLIP predictions (folder metadata only)")
    args = parser.parse_args()

    generate_metadata(
        image_dir  = args.image_dir,
        output     = args.output,
        model_path = args.model_path,
        limit      = args.limit,
        dry_run    = args.dry_run,
        skip_ai    = args.skip_ai,
    )