# 💎 Jewellery Image Semantic Search

> A production-grade multimodal search system for jewellery e-commerce — find products by image, text description, or both, powered by **CLIP** + **DINOv2** embeddings stored in **Qdrant**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Architecture Overview](#3-architecture-overview)
4. [Folder Structure](#4-folder-structure)
5. [Application Flow](#5-application-flow)
6. [API Reference](#6-api-reference)
7. [Configuration & Environment Variables](#7-configuration--environment-variables)
8. [Local Development Setup (venv)](#8-local-development-setup-venv)
9. [Docker Setup](#9-docker-setup)
10. [Data Ingestion Pipeline](#10-data-ingestion-pipeline)
11. [Preprocessing Pipeline](#11-preprocessing-pipeline)
12. [Streamlit UI](#12-streamlit-ui)
13. [Testing](#13-testing)
14. [Troubleshooting](#14-troubleshooting)
15. [Onboarding Notes for New Developers](#15-onboarding-notes-for-new-developers)

---

## 1. Project Overview

The **Jewellery Image Semantic Search** system enables shoppers and operators to find jewellery products using natural language, reference images, or a combination of both. It solves three core problems for jewellery e-commerce:

| Problem | Solution |
|---|---|
| Text queries fail to describe visual attributes ("rose gold band with oval stone") | CLIP embeds both text and images into a shared semantic space |
| Visually similar items look different in keywords | DINOv2 captures fine-grained visual structure beyond CLIP's coarser representation |
| Category drift in results (searching rings but getting necklaces) | Weighted fusion of three named-vector spaces with optional category filter |

### What It Does

- **Indexes** jewellery product images by generating three complementary embedding vectors per item: CLIP image, CLIP text (from caption), and DINOv2 image — all stored as named vectors in Qdrant.
- **Searches** by image upload, text description, or both simultaneously using weighted score fusion across all embedding spaces.
- **Ingests** at batch scale from a structured metadata JSON file, with a full preprocessing pipeline to generate captions automatically.
- **Exposes** a FastAPI REST API and a Streamlit web UI for interactive search.

### Key Numbers

| Metric | Value |
|---|---|
| Embedding dimension (CLIP ViT-L/14) | 768 |
| Embedding dimension (DINOv2-base) | 768 |
| Named vector spaces per item | 3 (`clip_image`, `clip_text`, `dino_image`) |
| Supported search modes | Text, Image, Hybrid |
| Max upload size | 10 MB (configurable) |
| Default top-K results | 10 (max 500) |

---

## 2. Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI + Uvicorn | Async HTTP server |
| **Vision-Language Model** | CLIP (`ViT-L-14`, `datacomp_xl_s13b_b90k`) | Joint image-text embedding |
| **Vision Backbone** | DINOv2 (`facebook/dinov2-base`) | Fine-grained image embedding |
| **Vector Database** | Qdrant | Multi-vector storage and search |
| **ML Framework** | PyTorch + `open_clip_torch` + HuggingFace Transformers | Model loading and inference |
| **Image Processing** | Pillow | Image decode, resize, format conversion |
| **UI** | Streamlit | Interactive search frontend |
| **Containerization** | Docker + Docker Compose | Service packaging and orchestration |
| **Metadata Generation** | CLIP zero-shot + folder heuristics | Automated caption and attribute extraction |

---

## 3. Architecture Overview

```
┌───────────────────────────────────────────────────┐
│             Client Layer                           │
│   Browser (Streamlit UI)  │  REST API Consumer     │
└───────────────┬───────────┴───────────────────────┘
                │ HTTP
                ▼
┌───────────────────────────────────────────────────┐
│             FastAPI Application  (:8000)           │
│  ┌──────────────────┐  ┌───────────────────────┐  │
│  │   Middleware      │  │  Routes               │  │
│  │  - Request log   │  │  POST /search-image   │  │
│  │  - Latency log   │  │  POST /index-image    │  │
│  └──────────────────┘  │  GET  /health         │  │
│                         └───────────────────────┘  │
└──────────────────────────┬────────────────────────┘
                           │
          ┌────────────────┴──────────────────┐
          │         Service Layer              │
          │  ┌───────────────────────────────┐ │
          │  │       SearchService           │ │
          │  │  - search_by_text()           │ │
          │  │  - search_by_image_raw()      │ │
          │  │  - search_image_text()        │ │
          │  │  - Weighted score fusion      │ │
          │  └──────────────┬────────────────┘ │
          └─────────────────│──────────────────┘
                            │
          ┌─────────────────┴──────────────────┐
          │       Embedder (CLIP + DINOv2)      │
          │  embed_clip_image()                 │
          │  embed_clip_text()                  │
          │  embed_dino_image()                 │
          └─────────────────┬──────────────────┘
                            │ vectors
                            ▼
          ┌─────────────────────────────────────┐
          │            Qdrant  (:6333)           │
          │  Collection: jewellery-2             │
          │  Named Vectors:                      │
          │    clip_image  (768d, cosine)        │
          │    clip_text   (768d, cosine)        │
          │    dino_image  (768d, cosine)        │
          └─────────────────────────────────────┘
```

### Key Design Decisions

- **Three named vectors per point**: Qdrant's named-vector API lets CLIP image, CLIP text (from caption), and DINOv2 live as separate search spaces in the same collection. At query time you pick which spaces to search and fuse the scores.
- **Weighted score fusion**: Rather than a simple average, each space gets a calibrated weight (`clip_image: 0.35`, `dino_image: 0.65` for image search; `clip_img: 0.35`, `clip_txt: 0.65` for text search) based on empirical retrieval quality.
- **Async concurrency**: All Qdrant queries for a single request run concurrently via `asyncio.gather` and `asyncio.to_thread`, keeping latency low even when firing multiple searches.
- **Stateless API**: No session state is kept in the API process. All state lives in Qdrant, making horizontal scaling straightforward.

---

## 4. Folder Structure

```
Task-3/
├── docker/
│   ├── Dockerfile              # Container image (Python 3.10-slim)
│   ├── requirements.txt        # Python dependencies
│   └── .dockerignore           # Files excluded from Docker build context
├── docker-compose.yml          # Orchestrates: Qdrant + API + Streamlit
├── streamlit_app.py            # Streamlit search UI
├── app/
│   ├── config.py               # All config from env vars with defaults
│   ├── api/
│   │   ├── main.py             # FastAPI app, middleware, health endpoint
│   │   └── routes/
│   │       └── search.py       # /search-image and /index-image endpoints
│   ├── services/
│   │   └── search_service.py   # SearchService: embedding + Qdrant queries + fusion
│   ├── ingestion/
│   │   ├── embedder.py         # Embedder class: CLIP + DINOv2 model loading & inference
│   │   └── ingest.py           # Batch ingestion pipeline (JSON metadata → Qdrant)
│   └── utils/
│       └── query_cleaner.py    # Text normalisation utilities (available, not wired in)
├── preprocess/
│   ├── generate_metadata.py    # CLIP + folder-path metadata + caption generator
│   ├── recaption.py            # Recaptioning pass for existing metadata
│   └── clean_json.py           # Metadata cleaning and validation
└── testing/
    ├── retrieval_eval.py        # Precision@K + category coherence evaluation
    ├── run_queries.py           # Batch query runner
    ├── load_test.py             # Concurrency / throughput load test
    ├── retrieval_eval_report.json
    ├── load_test_report.json
    └── query_test_report_2.json
```

### Layer Responsibilities

| Layer | Responsibility |
|---|---|
| `app/api/` | Routing, request validation, file decode, error handling |
| `app/services/` | Business logic: embed → search → fuse → rank |
| `app/ingestion/` | Model loading, embedding generation, batch upsert to Qdrant |
| `app/utils/` | Shared text utilities |
| `preprocess/` | Offline data preparation — run once before ingestion |
| `testing/` | Quality and performance evaluation scripts |

---

## 5. Application Flow

### Startup Sequence

```
App Start
    ├── Configure logging (level from LOG_LEVEL env var)
    ├── Instantiate SearchService
    │     ├── Connect to Qdrant (QDRANT_HOST:QDRANT_PORT)
    │     └── Instantiate Embedder
    │           ├── Load CLIP ViT-L/14 (datacomp_xl_s13b_b90k)
    │           └── Load DINOv2 facebook/dinov2-base
    └── Register routes + middleware → serve on :8000
```

### Search Flow (Hybrid Mode)

```
POST /search-image  { file=<image>, query="diamond ring", top_k=10 }
    ├── Validate content-type (jpeg/png/webp only) and file size (≤10MB)
    ├── Decode uploaded file → PIL Image
    ├── Detect mode: text-only / image-only / hybrid
    │
    ├── [Hybrid path] SearchService.search_image_text()
    │     ├── embed_clip_image(pil)    → clip_img_vec (768d, normalised)
    │     ├── embed_dino_image(pil)    → dino_img_vec (768d, normalised)
    │     ├── embed_clip_text(query)   → clip_txt_vec (768d, normalised)
    │     │
    │     ├── asyncio.gather:
    │     │     ├── Qdrant query(clip_image, clip_img_vec, limit=30)
    │     │     ├── Qdrant query(dino_image, dino_img_vec, limit=30)
    │     │     └── Qdrant query(clip_text,  clip_txt_vec, limit=30)
    │     │
    │     ├── Weighted score fusion:
    │     │     clip_image × 0.35 + dino_image × 0.35 + clip_text × 0.30
    │     └── Sort → return top_k results
    │
    └── Return JSON: { mode, query, total_returned, results: [...] }
```

### Ingestion Flow

```
python app/ingestion/ingest.py
    ├── Connect to Qdrant
    ├── Create collection 'jewellery-2' (idempotent — skips if exists)
    │     └── Named vectors: clip_image, clip_text, dino_image (768d, cosine)
    ├── Load metadata_cleaned.json
    └── For each item in batches of 32:
          ├── Load image from disk → PIL Image
          ├── generate_embeddings(image_path, caption)
          │     ├── embed_clip_image → normalise
          │     ├── embed_clip_text  → normalise
          │     └── embed_dino_image → normalise
          └── Qdrant upsert: PointStruct { id, vectors, payload }
```

---

## 6. API Reference

### Endpoint Summary

| Method | Endpoint | Description | Response |
|---|---|---|---|
| `GET` | `/health` | Liveness + Qdrant connectivity check | JSON |
| `POST` | `/search-image` | Semantic search by image, text, or both | JSON |
| `POST` | `/index-image` | Index a single image into the collection | JSON |

---

### `GET /health`

Returns service status and whether Qdrant is reachable and the collection exists.

```json
// 200 OK — healthy
{
  "status": "ok",
  "qdrant": "reachable",
  "collection": "jewellery-2"
}

// 503 Service Unavailable — degraded
{
  "status": "degraded",
  "qdrant": "unreachable",
  "collection": "jewellery-2",
  "detail": "Cannot reach Qdrant at qdrant:6333 or collection 'jewellery-2' does not exist."
}
```

---

### `POST /search-image`

Multipart form endpoint. At least one of `query` (text) or `file` (image) must be provided.

**Form fields:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | No* | — | Natural language search query |
| `file` | file | No* | — | Image file (JPEG / PNG / WebP, ≤10 MB) |
| `top_k` | int | No | 10 | Number of results to return (max 500) |
| `page` | int | No | 1 | Page number for pagination |
| `page_size` | int | No | 20 | Results per page (max 100) |
| `category` | string | No | — | Filter by category (`rings`, `earring`, etc.) |

*At least one of `query` or `file` is required.

**Search modes (auto-detected):**

| Mode | Trigger | Embedding spaces used |
|---|---|---|
| `text` | `query` only | `clip_text` (0.65) + `clip_image` (0.35) |
| `image` | `file` only | `dino_image` (0.65) + `clip_image` (0.35) |
| `hybrid` | Both `query` and `file` | `clip_image` (0.35) + `dino_image` (0.35) + `clip_text` (0.30) |

**Response:**
```json
{
  "mode": "hybrid",
  "query": "diamond ring",
  "image_filename": "ref.jpg",
  "total_returned": 10,
  "total": 10,
  "page": 1,
  "page_size": 20,
  "total_pages": 1,
  "results": [
    {
      "rank": 1,
      "id": 42,
      "score": 0.8731,
      "payload": {
        "filename": "rings/diamond/img_042.jpg",
        "filepath": "data/images/rings/diamond/img_042.jpg",
        "category": "rings",
        "metal": "platinum",
        "caption": "Platinum solitaire ring with oval diamond centre stone"
      }
    }
  ]
}
```

**Error responses:**

| HTTP Code | Cause |
|---|---|
| 415 | Unsupported image type |
| 413 | File exceeds 10 MB |
| 422 | Neither `query` nor `file` provided |
| 500 | Search failed (see server logs) |

---

### `POST /index-image`

Index a single image into the Qdrant collection at runtime. Useful for adding new products without re-running the full batch ingestion.

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Image file (JPEG / PNG / WebP) |
| `caption` | string | Yes | Text description of the image |
| `category` | string | No | Product category |
| `metal` | string | No | Metal type |
| `item_id` | string | No | Integer or UUID — auto-generated if omitted |

**Response:**
```json
{
  "status": "indexed",
  "id": "a3f1c2d4-...",
  "filename": "ring_platinum_001.jpg",
  "caption": "Platinum band with three-stone diamond setting"
}
```

---

## 7. Configuration & Environment Variables

All configuration lives in `app/config.py` and is sourced from environment variables with sensible defaults for local development.

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant HTTP port |
| `COLLECTION_NAME` | `jewellery-2` | Qdrant collection name |
| `CLIP_MODEL_NAME` | `ViT-L-14` | CLIP model architecture |
| `CLIP_PRETRAINED` | `datacomp_xl_s13b_b90k` | CLIP pretrained weights |
| `DINO_MODEL_NAME` | `facebook/dinov2-base` | DINOv2 model from HuggingFace |
| `MAX_FILE_MB` | `10` | Max upload size in megabytes |
| `DEFAULT_TOP_K` | `10` | Default number of search results |
| `MAX_TOP_K` | `500` | Hard cap on search results |
| `DEFAULT_PAGE_SIZE` | `20` | Default pagination page size |
| `MAX_PAGE_SIZE` | `100` | Max pagination page size |
| `INGEST_BATCH_SIZE` | `32` | Batch size for Qdrant upsert during ingestion |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `API_URL` | `http://localhost:8000` | Used by Streamlit to call the FastAPI backend |

---

## 8. Local Development Setup (venv)

Use this path for active development with hot-reload and direct debugging.

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.12 recommended |
| Docker | 24+ | Only needed to run Qdrant |
| Git | Any | — |

> **GPU note:** The embedding models run on CPU by default. If you have a CUDA-capable GPU, PyTorch will detect it automatically. Expect ~5–10x faster embedding generation with a GPU.

### Step 1 — Clone and navigate

```bash
git clone <your-repo-url>
cd Task-3
```

### Step 2 — Create virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate         # Linux / macOS
# .venv\Scripts\activate          # Windows PowerShell
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r docker/requirements.txt

# Extra packages needed only for testing / preprocessing
pip install httpx rich tqdm
```

### Step 4 — Start Qdrant with Docker

The API needs Qdrant running. The easiest way is a single Docker command:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Verify it's up:

```bash
curl http://localhost:6333/healthz
# {"title":"qdrant - vector search engine"}
```

### Step 5 — Set environment variables

```bash
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export PYTHONPATH=$(pwd)          # so `app.*` imports resolve
export LOG_LEVEL=DEBUG            # optional, more verbose
```

### Step 6 — Prepare data directory

```bash
mkdir -p data/images data/models
```

Place your jewellery images under `data/images/` following the expected structure:
```
data/images/
├── bracelet/
├── earring/
│   ├── diamond/
│   ├── gold/
│   └── ...
├── necklaces/
├── rings/
│   ├── diamond/
│   └── ...
└── watches/
```

### Step 7 — Generate metadata (first time only)

```bash
python preprocess/generate_metadata.py \
  --image-dir data/images \
  --output data/metadata.json

python preprocess/clean_json.py \
  --input data/metadata.json \
  --output data/metadata_cleaned.json
```

### Step 8 — Ingest images into Qdrant

```bash
python app/ingestion/ingest.py
# Connects to Qdrant, creates collection, embeds all images, upserts vectors
```

This will download CLIP and DINOv2 weights from HuggingFace on first run (~1–2 GB). Subsequent runs use the local cache.

### Step 9 — Run the API

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- API: http://localhost:8000
- Interactive docs (Swagger): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Step 10 — Run the Streamlit UI (optional)

```bash
export API_URL=http://localhost:8000
streamlit run streamlit_app.py --server.port 8501
```

Open http://localhost:8501 in your browser.

### Verify everything works

```bash
# Health check
curl http://localhost:8000/health

# Text search
curl -X POST http://localhost:8000/search-image \
  -F "query=gold necklace with pearls" \
  -F "top_k=5"

# Image search
curl -X POST http://localhost:8000/search-image \
  -F "file=@data/images/rings/diamond/some_ring.jpg" \
  -F "top_k=5"
```

---

## 9. Docker Setup

Use Docker Compose to run the full stack (Qdrant + FastAPI + Streamlit) with a single command. This is the recommended setup for demos and production-like testing.

### Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose plugin)
- At least 8 GB RAM available to Docker (models are large)

### Option A — Full Stack with Docker Compose

This starts Qdrant, the FastAPI backend, and the Streamlit UI together:

```bash
# From the project root (Task-3/)
docker-compose up --build
```

Services started:

| Service | URL | Description |
|---|---|---|
| Qdrant | http://localhost:6333 | Vector database + REST API |
| FastAPI | http://localhost:8000 | Search API |
| Streamlit | http://localhost:8501 | Search UI |

To run in the background:

```bash
docker-compose up --build -d
```

To stop all services:

```bash
docker-compose down
```

To stop and remove all volumes (full reset including stored vectors):

```bash
docker-compose down -v
```

### Option B — API Only (without Streamlit)

If you only need the FastAPI backend:

```bash
docker-compose up --build qdrant app
```

### Option C — Build and run the image manually

```bash
# Build the image
docker build -t image-search -f docker/Dockerfile .

# Run Qdrant first
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Run the API container, linked to Qdrant
docker run -d \
  --name jewellery-api \
  -p 8000:8000 \
  -e QDRANT_HOST=host.docker.internal \
  -e QDRANT_PORT=6333 \
  -e PYTHONPATH=/app \
  -e CLIP_MODEL_NAME=ViT-L-14 \
  -e CLIP_PRETRAINED=datacomp_xl_s13b_b90k \
  -e DINO_MODEL_NAME=facebook/dinov2-base \
  -v $(pwd)/data:/app/data \
  image-search
```

> **Note:** `host.docker.internal` is macOS/Windows Docker Desktop specific. On Linux, use `--network host` or create a Docker network instead.

### Ingest data inside Docker

After the containers are running, run ingestion from inside the API container:

```bash
docker exec -it jewellery-api python app/ingestion/ingest.py
```

Or mount your data and run ingestion as a one-off:

```bash
docker run --rm \
  -e QDRANT_HOST=host.docker.internal \
  -e QDRANT_PORT=6333 \
  -e PYTHONPATH=/app \
  -v $(pwd)/data:/app/data \
  image-search \
  python app/ingestion/ingest.py
```

### Viewing logs

```bash
# All services
docker-compose logs -f

# Just the API
docker-compose logs -f app

# Just Qdrant
docker-compose logs -f qdrant
```

---

## 10. Data Ingestion Pipeline

### Overview

The ingestion pipeline reads a JSON metadata file and bulk-upserts all items into Qdrant with three embedding vectors each.

### Metadata format (`metadata_cleaned.json`)

```json
[
  {
    "filename": "earring_gold_001.jpg",
    "filepath": "data/images/earring/gold/earring_gold_001.jpg",
    "category": "earring",
    "metal": "gold",
    "caption": "Gold hoop earrings with diamond pave setting",
    "is_product_image": true
  }
]
```

### Run ingestion

```bash
# With venv
python app/ingestion/ingest.py

# Inside Docker
docker exec -it jewellery-api python app/ingestion/ingest.py
```

The ingestor is idempotent at the collection level — if `jewellery-2` already exists, it skips creation. Individual points are upserted (insert or overwrite by ID).

### Index a single image via API

For adding items post-ingestion without re-running the full pipeline:

```bash
curl -X POST http://localhost:8000/index-image \
  -F "file=@new_product.jpg" \
  -F "caption=Sterling silver bracelet with turquoise stones" \
  -F "category=bracelet" \
  -F "metal=silver"
```

---

## 11. Preprocessing Pipeline

Run these scripts offline, before ingestion, to prepare your image dataset.

### Step 1 — Generate metadata

```bash
python preprocess/generate_metadata.py \
  --image-dir data/images \
  --output data/metadata.json

# Fast mode — no CLIP inference, folder path only
python preprocess/generate_metadata.py \
  --image-dir data/images \
  --output data/metadata.json \
  --skip-ai

# Dry run — prints output, saves nothing
python preprocess/generate_metadata.py \
  --image-dir data/images \
  --dry-run
```

The script derives `category`, `metal`, `gemstone`, `style`, and `audience` from the folder path and uses CLIP zero-shot classification only where the folder can't tell you (e.g. metal type when the subfolder is named after a gemstone). It also generates three caption variants per image.

### Step 2 — Clean and validate metadata

```bash
python preprocess/clean_json.py \
  --input data/metadata.json \
  --output data/metadata_cleaned.json
```

### Step 3 — Recaption (optional)

If you want to regenerate captions for an existing metadata file without re-scanning the image tree:

```bash
python preprocess/recaption.py \
  --input data/metadata_cleaned.json \
  --output data/metadata_recaptioned.json
```

---

## 12. Streamlit UI

The Streamlit app provides an interactive frontend for exploring the search system.

### Features

- Upload an image, type a query, or both simultaneously
- Adjust `top_k` and filter by category from the sidebar
- Browse paginated results with thumbnails, scores, and metadata
- Switch between search modes (text / image / hybrid) automatically

### Run locally

```bash
export API_URL=http://localhost:8000
streamlit run streamlit_app.py --server.port 8501
```

### Run with Docker Compose

The Streamlit service is included in `docker-compose.yml` and starts automatically with `docker-compose up`. It is pre-configured to call the API container by its Docker Compose service name (`http://app:8000`).

---

## 13. Testing

All test scripts live in `testing/`. They call the live API, so the service must be running.

### Retrieval quality evaluation

Measures Precision@K, category coherence, and score sanity by sampling images from your metadata and querying the API.

```bash
pip install httpx rich

python testing/retrieval_eval.py \
  --url http://localhost:8000 \
  --metadata data/metadata_cleaned.json \
  --samples 100 \
  --top-k 10

# Output: terminal table + testing/retrieval_eval_report.json
```

### Load / concurrency test

Tests API throughput and latency under concurrent load across `/health`, text search, and image search endpoints.

```bash
pip install httpx

python testing/load_test.py \
  --url http://localhost:8000 \
  --users 50 \
  --duration 30

# Output: terminal summary + testing/load_test_report.json
```

### Batch query runner

Runs a batch of predefined queries and saves results for analysis.

```bash
python testing/run_queries.py \
  --url http://localhost:8000 \
  --output testing/query_test_report.json
```

---

## 14. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `Connection refused` on startup | Qdrant not running | Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant` |
| `/health` returns `503 degraded` | Qdrant reachable but collection missing | Run ingestion: `python app/ingestion/ingest.py` |
| Slow first request (30–60s) | CLIP and DINOv2 loading on first call | Models load eagerly at Embedder init — this is a one-time cost per process start |
| `ModuleNotFoundError: app.config` | `PYTHONPATH` not set | `export PYTHONPATH=$(pwd)` from the `Task-3/` root |
| Empty search results | Collection empty or wrong collection name | Verify `COLLECTION_NAME` matches what was used during ingestion; re-run ingest |
| `415 Unsupported image type` | Wrong file format | API accepts only `image/jpeg`, `image/jpg`, `image/png`, `image/webp` |
| `413 File too large` | Image exceeds 10 MB | Resize before upload, or raise `MAX_FILE_MB` env var |
| HuggingFace download hangs | Network / proxy issue | Set `HF_HUB_OFFLINE=1` and pre-cache models, or check proxy settings |
| Docker: `host.docker.internal` not found | Linux Docker | Use `--network host` or a shared Docker network instead |
| DINOv2 `RuntimeError: CUDA out of memory` | GPU memory too small | Set `CUDA_VISIBLE_DEVICES=""` to force CPU |
| Qdrant `vector dimension mismatch` | Collection built with different model | Drop and recreate the collection, then re-ingest |

---

## 15. Onboarding Notes for New Developers

### Read these files first (in order)

1. `app/config.py` — all tunable parameters in one place
2. `app/ingestion/embedder.py` — how CLIP and DINOv2 are loaded and called
3. `app/services/search_service.py` — the full search + fusion logic
4. `app/api/routes/search.py` — how requests are validated and delegated
5. `app/api/main.py` — app setup, middleware, health endpoint

### Coding conventions

| Convention | Pattern |
|---|---|
| **Async for I/O** | Qdrant queries are blocking; wrapped with `asyncio.to_thread` so they don't block the event loop |
| **Concurrent queries** | Multiple Qdrant searches per request run in parallel via `asyncio.gather` |
| **Normalised embeddings** | Every embedding is L2-normalised before storage and before querying, so cosine similarity equals dot product |
| **Config via env vars** | No hardcoded values in source — everything flows through `app/config.py` |
| **Structured logging** | Every major step is logged with `logger.info` / `logger.debug`, including embed timing and result counts |
| **HTTP errors via `HTTPException`** | Validation failures return proper 4xx codes, not 500s |

### Extending the system

**Add a new embedding model:**
1. Add it to `Embedder.__init__()` in `app/ingestion/embedder.py`
2. Add a new named vector to the Qdrant collection schema in `app/ingestion/ingest.py`
3. Add a search method and fusion weight in `app/services/search_service.py`
4. Re-create the collection and re-ingest (named vectors can't be added to an existing collection)

**Add a new API endpoint:**
1. Define the endpoint in `app/api/routes/search.py` (or a new router file)
2. Register the router in `app/api/main.py`
3. Add business logic to `SearchService` if needed

**Change the collection name:**
Set `COLLECTION_NAME` env var. If the collection already has data under the old name, you'll need to re-ingest under the new name — Qdrant doesn't rename collections.

---

*Project: Jewellery Image Semantic Search | Stack: FastAPI + CLIP + DINOv2 + Qdrant | Python 3.10+*
