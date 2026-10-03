
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.search import router as search_router
from app.api.routes.products import router as products_router
from app.api.routes.sync import router as sync_router
from app.config import LOG_LEVEL, COLLECTION_NAME, PRODUCT_COLLECTION, QDRANT_HOST, QDRANT_PORT
from app.sync import index

# Logging setup
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("api")

# App 
app = FastAPI(
    title="Jewellery Semantic Search",
    description=(
        "Image + text semantic search powered by CLIP and DINOv2 embeddings "
        "stored in Qdrant."
    ),
    version="1.0.0",
)
app.include_router(search_router)
app.include_router(products_router)
app.include_router(sync_router)


# Request / response logging middleware 
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request and its response status + latency."""
    t0 = time.monotonic()
    try:
        response = await call_next(request)
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.error(
            "UNHANDLED  %s %s — %.0fms — %s",
            request.method, request.url.path, latency_ms, exc,
        )
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    latency_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "%s  %s %s — %dms",
        response.status_code,
        request.method,
        request.url.path,
        latency_ms,
    )
    return response


#  Health 
@app.get("/health", tags=["System"])
def health():
    """Healthy when Qdrant is reachable and the product index alias points at a collection."""
    try:
        c = index.client()
        real = index.resolve_alias(c)
        points = index.count_points(c, real) if real else None
    except Exception as exc:
        logger.warning("Health check failed — Qdrant at %s:%s unreachable: %s", QDRANT_HOST, QDRANT_PORT, exc)
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "qdrant": "unreachable",
                     "detail": f"Cannot reach Qdrant at {QDRANT_HOST}:{QDRANT_PORT}."},
        )

    if not real:
        logger.warning("Health check: alias '%s' does not point at a collection yet", PRODUCT_COLLECTION)
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "qdrant": "reachable", "alias": PRODUCT_COLLECTION,
                     "detail": f"No product index yet: alias '{PRODUCT_COLLECTION}' is not set. Run a full sync."},
        )
    return {"status": "ok", "qdrant": "reachable", "alias": PRODUCT_COLLECTION,
            "collection": real, "points": points}


#  Startup / shutdown events 
@app.on_event("startup")
async def on_startup():
    logger.info(
        "Jewellery Search API starting up (log level: %s, collection: %s)",
        LOG_LEVEL,
        COLLECTION_NAME,
    )


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Jewellery Search API shutting down")