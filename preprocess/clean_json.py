import json

INPUT_FILE = "metadata.json"
OUTPUT_FILE = "metadata_cleaned.json"

# Fields to remove
REMOVE_FIELDS = {"sku", "gemstone", "style", "audience", "caption_variants", "is_product_image"}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Remove fields from each product
cleaned_data = []
for item in data:
    cleaned_item = {k: v for k, v in item.items() if k not in REMOVE_FIELDS}
    cleaned_data.append(cleaned_item)

# Save cleaned JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

print(f"✅ Cleaned JSON saved to {OUTPUT_FILE}")