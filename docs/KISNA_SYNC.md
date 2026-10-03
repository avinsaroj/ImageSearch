# Kisna product-image sync and search

SQL Server (`T_ITEM_MST`, `T_COMMON_MASTER`, `T_IMAGE_MST`, read-only) -> download -> CLIP + DINOv2 -> Qdrant.

## Architecture

| Service | Role |
|---|---|
| `qdrant` | vector store (volume `qdrant_data`) |
| `app` (host :8005) | FastAPI: search, catalog, sync control. Never runs sync work itself |
| `worker` | single instance: daily scheduler + executes queued jobs (volume `sync_state`) |
| `streamlit` (host :8501) | main page (legacy search) + pages: Reference Image Search, Dashboard, Sync Admin |

Jobs are rows in SQLite (`SYNC_STATE_DB`, volume `sync_state`). API/UI only enqueue; the worker claims
and runs them, so a Streamlit/API restart never interrupts a sync, and a worker restart re-queues and resumes.

## Setup

```bash
cp .env.example .env        # fill SQLSERVER_* (read-only login) and ADMIN_API_TOKEN
docker compose up -d --build
```
Do not commit `.env`. The password is only read from the environment and is never logged.
For TLS keep `SQLSERVER_ENCRYPT=true`; set `SQLSERVER_TRUST_SERVER_CERTIFICATE=true` only if the server uses a
self-signed certificate and you accept that trade-off.

## Qdrant design

- Real collection `kisna-images-v1-<UTC stamp>`, served through alias `PRODUCT_COLLECTION` (`kisna-images-v1`).
  The legacy `jewellery-2` collection is untouched.
- Named vectors `clip_image` (768, cosine) and `dino_image` (768, cosine), both L2-normalised. Vectors from
  different models are never compared with each other; each space is searched with its own query vector.
- One point per image. Point id = `uuid5("kisna-img:<ImgID>")`, so re-upserts are idempotent.
- Payload: ItemID, ItemCode, CategoryCode, IsPlainGold/IsSolitaire/IsValid/IsFranchiseItem (normalised bool,
  raw value in `*_raw`), StatusRemark, CreatedOn, UpdatedOn, ImgID, ImageURL, ImgViewID, ContentHash,
  EmbeddingModelVersion, LastProcessedAt, `is_active`, plus `category_norm` / `item_code_norm`.
- Payload indexes: ItemID, ImgID, category_norm, item_code_norm, the four flags, is_active.
- Changing a model or `EMBEDDING_MODEL_VERSION` => next incremental run re-queues every image indexed under the old
  version. For a clean switch run a full rebuild: it fills a new collection and moves the alias only if
  pending == 0 and points >= indexed images. The old collection is kept; delete it manually once satisfied.

## Synchronization

Both kinds scan product master (keyset pages by ItemID) and image rows (metadata only), then process `pending`
images (download with bounded concurrency/timeouts/retries -> decode+validate -> embed -> upsert).

- New image rows -> pending. Changed (ItemID, URL, ImgViewID) -> pending. Unchanged -> untouched.
- Changed product fields (fingerprint over the master columns) -> payload refreshed on all that product's points,
  no re-embedding.
- Re-processed image whose bytes hash is unchanged (or HTTP 304 via ETag) -> no re-embedding.
- Failed images keep their error and attempt count; they are retried on later runs up to `SYNC_MAX_ATTEMPTS`
  (`POST /sync/retry-failed` resets the counter).
- Deletion policy (`SYNC_DELETION_POLICY`): `mark_invalid` (default) sets `is_active=false` so the image drops out
  of search but stays recoverable; `delete` removes the point. Images whose product left the master are treated alike.
- **Change detection deliberately does not depend on `ItemCngDt` or `ImgID` ordering.** Their behaviour in the
  live database has not been verified, so the daily run diffs fingerprints over the full (metadata-only) scan.
  If `ItemCngDt` proves reliable, the product scan can be narrowed to an overlapping time window.

Daily schedule: `SCHEDULER_DAILY_AT` (default 02:00) in `SCHEDULER_TIMEZONE` (default Asia/Kolkata); a missed
run is caught up when the worker starts. Job-level retries: 3 attempts, 30 s doubling backoff, capped at 10 min.

## API

Admin endpoints need header `X-Admin-Token: <ADMIN_API_TOKEN>` (disabled with 503 if the token is unset).

| Endpoint | Notes |
|---|---|
| `GET /health` | existing |
| `GET /categories` | live from SQL Server, 5 min cache |
| `POST /search/image` | multipart: `file`, `top_k`, `mode` (fusion/clip/dino), `category`, `plain_gold`/`solitaire`/`valid`/`franchise` (all/yes/no), `item_codes`, `min_score`, `clip_weight`, `dino_weight` |
| `GET /products/{item_id}`, `GET /products/{item_id}/images` | master record / images with index status |
| `GET /sync/status`, `/sync/history`, `/sync/failures` | admin |
| `POST /sync/full` `{"confirm": true}` | 409 if an index exists and `confirm` is false, or a job is active |
| `POST /sync/incremental`, `POST /sync/retry-failed` | admin, return 202 + job id |

Search score = cosine similarity (weighted mean of the selected spaces), not a probability. Products are ranked by
their best image. The weights (0.35 CLIP / 0.65 DINOv2) are inherited from the existing project and are **not**
tuned for this catalog - evaluate against a labelled set before relying on them.

## Operations

```bash
docker compose logs -f worker          # sync progress / errors
docker compose logs -f app
curl -H "X-Admin-Token: $ADMIN_API_TOKEN" http://localhost:8005/sync/status
curl -H "X-Admin-Token: $ADMIN_API_TOKEN" http://localhost:8005/sync/failures
```
Initial index: Streamlit "Sync Admin" -> Start full sync, or `POST /sync/full`.
If the Qdrant volume is lost but `sync_state` survives, run a full rebuild (state would otherwise think images are indexed).
