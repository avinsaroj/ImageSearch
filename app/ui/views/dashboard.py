import streamlit as st

from app.ui import theme
from app.ui.client import call


def render() -> None:
    theme.head("Index dashboard", "Products, images and vectors are different counts: a product has several "
               "images, and each indexed image is one Qdrant point.")
    if st.button("Refresh"):
        st.rerun()
    s, err = call("GET", "/sync/status", admin=True)
    if err:
        st.error(err)
        return

    sql, imgs, q, job = s["sql_server"], s["images"], s["qdrant"], s["current_job"]
    fmt = lambda v: f"{v:,}" if isinstance(v, int) else "n/a"  # noqa: E731
    c = st.columns(5)
    theme.kpi(c[0], "Products", fmt(sql["products"]), "SQL Server")
    theme.kpi(c[1], "Images", fmt(sql["images"]), "SQL Server")
    theme.kpi(c[2], "Indexed", fmt(imgs["indexed"]), "images embedded")
    theme.kpi(c[3], "Pending", fmt(imgs["pending"]), "waiting to process")
    theme.kpi(c[4], "Failed", fmt(imgs["failed"]), "see Sync admin")
    if sql.get("error"):
        st.warning(f"SQL Server totals unavailable ({sql['error']}).")

    total = sql["images"] or 0
    if total:
        st.write("")
        st.progress(min(imgs["indexed"] / total, 1.0), text=f"{imgs['indexed']:,} of {total:,} images indexed")

    st.write("")
    c = st.columns(3)
    theme.kpi(c[0], "Qdrant", q["status"], f"{fmt(q.get('points'))} vectors")
    theme.kpi(c[1], "Sync", f"{job['kind']} running" if job else "Idle", job["stage"] if job else "")
    ls = s["last_success"]
    theme.kpi(c[2], "Last successful sync", (ls["finished_at"][:16].replace("T", " ") + " UTC") if ls else "Never",
              ls["kind"] if ls else "")

    st.write("")
    with st.expander("Technical details"):
        st.markdown(f"- **Model version:** `{s['model_version']}`\n"
                    f"- **Collection:** `{q.get('collection', '-')}` (alias `{q.get('alias')}`)\n"
                    f"- **Daily schedule:** {s['scheduler']['daily_at']} {s['scheduler']['timezone']} "
                    f"(last fired: {s['scheduler']['last_fired_date'] or 'never'})")
