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


_MAX_ID = 9223372036854775807


def scope_defaults() -> dict:
    """Scope used when a job carries none: the SYNC_* environment settings."""
    return {"item_id_max": config.SYNC_ITEM_ID_MAX or None, "valid": True if config.SYNC_ONLY_VALID else None}


def _scope_where(scope: Optional[dict]) -> tuple[str, list]:
    """WHERE clause (on aliases i = T_ITEM_MST, cm = T_COMMON_MASTER) and params for a sync scope.

    scope keys: categories, plain_gold / solitaire / valid / franchise (True, False or None = any),
    search_text (substring of item code or status remark); item_id_min / item_id_max come from the env
    defaults only. None means the environment defaults."""
    s = scope_defaults() if scope is None else scope
    clauses = ["i.ItemID >= ?", "i.ItemID <= ?"]
    args: list = [s.get("item_id_min") or 0, s.get("item_id_max") or _MAX_ID]
    cats = s.get("categories") or []
    if cats:
        clauses.append(f"cm.MstCd IN ({','.join('?' * len(cats))})")
        args += cats
    valid = s.get("valid")
    if valid is None and s.get("only_valid"):  # scopes saved before the yes/no filters existed
        valid = True
    for flag, col in ((s.get("plain_gold"), "ItemPlainGold"), (s.get("solitaire"), "ItemSoliterSts"),
                      (valid, "ItemValidSts"), (s.get("franchise"), "ItemFranchiseSts")):
        if flag is not None:
            clauses.append(f"i.{col} = '{'Y' if flag else 'N'}'")
    text = (s.get("search_text") or "").strip()
    if text:
        like = "%" + text.replace("[", "[[]").replace("%", "[%]").replace("_", "[_]") + "%"
        clauses.append("(i.ItemCd LIKE ? OR i.ItemStatusRemark LIKE ?)")
        args += [like, like]
    return " AND ".join(clauses), args


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


def iter_products(batch_size: int, scope: Optional[dict] = None, after_item_id: int = 0) -> Iterator[list[dict]]:
    """Keyset-paginated product batches ordered by ItemID (resumable from a checkpoint).

    Honours scope filters and scope["max_products"] (stop after N products, for test runs)."""
    last = after_item_id
    where, args = _scope_where(scope)
    remaining = (scope or {}).get("max_products") or None
    sql = f"SELECT TOP (?) {_PRODUCT_COLUMNS} {_PRODUCT_FROM} WHERE i.ItemID > ? AND {where} ORDER BY i.ItemID"
    with connect() as conn:
        while True:
            size = min(batch_size, remaining) if remaining else batch_size
            batch = _rows(conn.cursor().execute(sql, size, last, *args))
            if not batch:
                return
            yield batch
            last = batch[-1]["ItemID"]
            if remaining:
                remaining -= len(batch)
                if remaining <= 0:
                    return


def count_scope(scope: Optional[dict]) -> dict:
    """How much a sync scope would cover: {products, images, files} (files = distinct image paths)."""
    where, args = _scope_where(scope)
    top = (scope or {}).get("max_products") or None
    items = (f"SELECT {'TOP (?) ' if top else ''}i.ItemID {_PRODUCT_FROM} WHERE {where}"
             + (" ORDER BY i.ItemID" if top else ""))
    params = ([top] if top else []) + args
    with connect() as conn:
        cur = conn.cursor()
        products = cur.execute(f"SELECT COUNT(*) FROM ({items}) t", *params).fetchone()[0]
        images, files = cur.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT img.ImgPath) FROM dbo.T_IMAGE_MST AS img "
            f"WHERE img.ImgPath IS NOT NULL AND img.ImgTagRefID IN ({items})", *params).fetchone()
    return {"products": products, "images": images, "files": files}


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


def iter_all_image_rows(batch_size: int, item_id_min: int = 0, item_id_max: Optional[int] = None,
                        after_img_id: int = 0) -> Iterator[list[dict]]:
    """Metadata-only scan of image rows (no downloads) used to detect new/changed images.

    Limited to the ItemID window of the products being synced."""
    last = after_img_id
    sql = (
        f"SELECT TOP (?) {_IMAGE_COLUMNS} FROM dbo.T_IMAGE_MST AS img "
        "WHERE img.ImgID > ? AND img.ImgTagRefID >= ? AND img.ImgTagRefID <= ? ORDER BY img.ImgID"
    )
    with connect() as conn:
        while True:
            batch = _rows(conn.cursor().execute(sql, batch_size, last, item_id_min, item_id_max or _MAX_ID))
            if not batch:
                return
            for r in batch:
                r["ImageURL"] = image_url(r.pop("ImgPath"))
            yield batch
            last = batch[-1]["ImgID"]
