import os
 
# Qdrant
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
 
# Collection name
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "jewellery-2")
 
CLIP_MODEL_NAME: str  = os.getenv("CLIP_MODEL_NAME", "ViT-L-14")
CLIP_PRETRAINED: str  = os.getenv("CLIP_PRETRAINED", "datacomp_xl_s13b_b90k")
DINO_MODEL_NAME: str  = os.getenv("DINO_MODEL_NAME", "facebook/dinov2-base")
 
CLIP_EMBED_DIM: int   = 768
DINO_EMBED_DIM: int   = 768
 
# API settings
MAX_FILE_MB: int      = int(os.getenv("MAX_FILE_MB", "10"))
DEFAULT_TOP_K: int    = int(os.getenv("DEFAULT_TOP_K", "10"))
MAX_TOP_K: int        = int(os.getenv("MAX_TOP_K", "500"))
DEFAULT_PAGE_SIZE: int = int(os.getenv("DEFAULT_PAGE_SIZE", "20"))
MAX_PAGE_SIZE: int     = int(os.getenv("MAX_PAGE_SIZE", "100"))
 
ALLOWED_CONTENT_TYPES: set = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
 
# Ingestion settings
INGEST_BATCH_SIZE: int = int(os.getenv("INGEST_BATCH_SIZE", "32"))
 
# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# SQL Server (read-only source). Credentials come from the environment only.
SQLSERVER_HOST: str = os.getenv("SQLSERVER_HOST", "")
SQLSERVER_PORT: int = int(os.getenv("SQLSERVER_PORT", "1433"))
SQLSERVER_DATABASE: str = os.getenv("SQLSERVER_DATABASE", "")
SQLSERVER_USER: str = os.getenv("SQLSERVER_USER", "")
SQLSERVER_PASSWORD: str = os.getenv("SQLSERVER_PASSWORD", "")
SQLSERVER_ENCRYPT: bool = _env_bool("SQLSERVER_ENCRYPT", True)
SQLSERVER_TRUST_SERVER_CERTIFICATE: bool = _env_bool("SQLSERVER_TRUST_SERVER_CERTIFICATE", False)
SQLSERVER_ODBC_DRIVER: str = os.getenv("SQLSERVER_ODBC_DRIVER", "ODBC Driver 18 for SQL Server")
IMAGE_BASE_URL: str = os.getenv("IMAGE_BASE_URL", "https://assets.kisna.com/")

# Product-image index (separate from the legacy COLLECTION_NAME so it is never overwritten)
PRODUCT_COLLECTION: str = os.getenv("PRODUCT_COLLECTION", "kisna-images-v1")
EMBEDDING_MODEL_VERSION: str = os.getenv(
    "EMBEDDING_MODEL_VERSION",
    f"clip={CLIP_MODEL_NAME}/{CLIP_PRETRAINED};dino={DINO_MODEL_NAME};v1",
)

# Synchronization
SYNC_STATE_DB: str = os.getenv("SYNC_STATE_DB", "/app/state/sync_state.db")
SYNC_PRODUCT_BATCH: int = int(os.getenv("SYNC_PRODUCT_BATCH", "200"))
SYNC_EMBED_BATCH: int = int(os.getenv("SYNC_EMBED_BATCH", "16"))
SYNC_DOWNLOAD_CONCURRENCY: int = int(os.getenv("SYNC_DOWNLOAD_CONCURRENCY", "8"))
SYNC_DOWNLOAD_TIMEOUT_S: float = float(os.getenv("SYNC_DOWNLOAD_TIMEOUT_S", "20"))
SYNC_DOWNLOAD_RETRIES: int = int(os.getenv("SYNC_DOWNLOAD_RETRIES", "3"))
SYNC_MAX_IMAGE_MB: int = int(os.getenv("SYNC_MAX_IMAGE_MB", "20"))
SYNC_MAX_ATTEMPTS: int = int(os.getenv("SYNC_MAX_ATTEMPTS", "5"))
SYNC_DELETION_POLICY: str = os.getenv("SYNC_DELETION_POLICY", "mark_invalid")  # mark_invalid | delete

# Scheduler
SCHEDULER_TIMEZONE: str = os.getenv("SCHEDULER_TIMEZONE", "Asia/Kolkata")
SCHEDULER_DAILY_AT: str = os.getenv("SCHEDULER_DAILY_AT", "02:00")  # HH:MM local to the timezone

# Admin protection for /sync/* endpoints. Empty token => admin endpoints refuse all calls.
ADMIN_API_TOKEN: str = os.getenv("ADMIN_API_TOKEN", "")

# Fusion weights for product image search (CLIP-only / DINO-only: set the other to 0)
SEARCH_WEIGHT_CLIP: float = float(os.getenv("SEARCH_WEIGHT_CLIP", "0.35"))
SEARCH_WEIGHT_DINO: float = float(os.getenv("SEARCH_WEIGHT_DINO", "0.65"))