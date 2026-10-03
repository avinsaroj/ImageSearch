"""Read-only SQL Server access for product, category and image master data.

All queries are parameterized and only SELECT. Credentials come from app.config
(environment variables) and are never logged.
"""

import logging
from contextlib import contextmanager
from typing import Iterator, Optional

import pyodbc

from app import config

logger = logging.getLogger(__name__)

_PRODUCT_COLUMNS = """
    i.ItemID            AS ItemID,
    i.ItemCd            AS ItemCode,
    cm.MstCd            AS CategoryCode,
    i.ItemPlainGold     AS IsPlainGold,
    i.ItemSoliterSts    AS IsSolitaire,
    i.ItemValidSts      AS IsValid,
    i.ItemFranchiseSts  AS IsFranchiseItem,
    i.ItemStatusRemark  AS StatusRemark,
    i.ItemEntDt         AS CreatedOn,
    i.ItemCngDt         AS UpdatedOn
"""
_PRODUCT_FROM = """
    FROM dbo.T_ITEM_MST AS i
    INNER JOIN dbo.T_COMMON_MASTER AS cm ON i.ItemCtgCommonID = cm.MstID
"""
_IMAGE_COLUMNS = "img.ImgID, img.ImgTagRefID AS ItemID, img.ImgPath, img.ImgViewID"


def _connection_string() -> str:
    missing = [
        n for n, v in (
            ("SQLSERVER_HOST", config.SQLSERVER_HOST),
            ("SQLSERVER_DATABASE", config.SQLSERVER_DATABASE),
            ("SQLSERVER_USER", config.SQLSERVER_USER),
            ("SQLSERVER_PASSWORD", config.SQLSERVER_PASSWORD),
        ) if not v
    ]
    if missing:
        raise RuntimeError(f"SQL Server settings missing: {', '.join(missing)}")
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    return (
        f"DRIVER={{{config.SQLSERVER_ODBC_DRIVER}}};"
        f"SERVER={config.SQLSERVER_HOST},{config.SQLSERVER_PORT};"
        f"DATABASE={config.SQLSERVER_DATABASE};"
        f"UID={config.SQLSERVER_USER};PWD={{{config.SQLSERVER_PASSWORD}}};"
        f"Encrypt={yn(config.SQLSERVER_ENCRYPT)};"
        f"TrustServerCertificate={yn(config.SQLSERVER_TRUST_SERVER_CERTIFICATE)};"
        "ApplicationIntent=ReadOnly;Connection Timeout=15;"
    )


@contextmanager
def connect() -> Iterator[pyodbc.Connection]:
    conn = pyodbc.connect(_connection_string(), readonly=True, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


def _rows(cur: pyodbc.Cursor) -> list[dict]:
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def image_url(img_path: Optional[str]) -> Optional[str]:
    """Same construction as the reference query: base + ImgPath."""
    if not img_path or not str(img_path).strip():
        return None
    return config.IMAGE_BASE_URL.rstrip("/") + "/" + str(img_path).strip().lstrip("/")


def ping() -> bool:
    try:
        with connect() as conn:
            conn.cursor().execute("SELECT 1").fetchone()
        return True
    except Exception as exc:
        logger.warning("SQL Server ping failed: %s", type(exc).__name__)
        return False


def fetch_categories() -> list[str]:
    sql = (
        "SELECT DISTINCT cm.MstCd AS CategoryCode FROM dbo.T_COMMON_MASTER AS cm "
        "INNER JOIN dbo.T_ITEM_MST AS i ON i.ItemCtgCommonID = cm.MstID ORDER BY cm.MstCd"
    )
    with connect() as conn:
        return [r["CategoryCode"] for r in _rows(conn.cursor().execute(sql)) if r["CategoryCode"]]


def count_products() -> int:
    with connect() as conn:
        return conn.cursor().execute("SELECT COUNT(*) " + _PRODUCT_FROM).fetchone()[0]


def count_images() -> int:
    with connect() as conn:
        return conn.cursor().execute(
            "SELECT COUNT(*) FROM dbo.T_IMAGE_MST WHERE ImgTagRefID IS NOT NULL AND ImgPath IS NOT NULL"
        ).fetchone()[0]


def iter_products(batch_size: int, after_item_id: int = 0) -> Iterator[list[dict]]:
    """Keyset-paginated product batches ordered by ItemID (resumable from a checkpoint)."""
    last = after_item_id
    sql = f"SELECT TOP (?) {_PRODUCT_COLUMNS} {_PRODUCT_FROM} WHERE i.ItemID > ? ORDER BY i.ItemID"
    with connect() as conn:
        while True:
            batch = _rows(conn.cursor().execute(sql, batch_size, last))
            if not batch:
                return
            yield batch
            last = batch[-1]["ItemID"]


def fetch_products_by_ids(item_ids: list[int]) -> list[dict]:
    if not item_ids:
        return []
    marks = ",".join("?" * len(item_ids))
    sql = f"SELECT {_PRODUCT_COLUMNS} {_PRODUCT_FROM} WHERE i.ItemID IN ({marks})"
    with connect() as conn:
        return _rows(conn.cursor().execute(sql, *item_ids))


def fetch_images_for_items(item_ids: list[int]) -> list[dict]:
    """Batched form of the reference image query (one round trip per product batch)."""
    if not item_ids:
        return []
    marks = ",".join("?" * len(item_ids))
    sql = (
        f"SELECT {_IMAGE_COLUMNS} FROM dbo.T_IMAGE_MST AS img "
        f"WHERE img.ImgTagRefID IN ({marks}) ORDER BY img.ImgID ASC"
    )
    with connect() as conn:
        rows = _rows(conn.cursor().execute(sql, *item_ids))
    for r in rows:
        r["ImageURL"] = image_url(r.pop("ImgPath"))
    return rows


def iter_all_image_rows(batch_size: int, after_img_id: int = 0) -> Iterator[list[dict]]:
    """Metadata-only scan of every image row (no downloads) used to detect new/changed images."""
    last = after_img_id
    sql = (
        f"SELECT TOP (?) {_IMAGE_COLUMNS} FROM dbo.T_IMAGE_MST AS img "
        "WHERE img.ImgID > ? AND img.ImgTagRefID IS NOT NULL ORDER BY img.ImgID"
    )
    with connect() as conn:
        while True:
            batch = _rows(conn.cursor().execute(sql, batch_size, last))
            if not batch:
                return
            for r in batch:
                r["ImageURL"] = image_url(r.pop("ImgPath"))
            yield batch
            last = batch[-1]["ImgID"]
