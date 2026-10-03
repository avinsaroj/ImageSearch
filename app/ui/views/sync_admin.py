import streamlit as st

from app.ui import theme
from app.ui.client import call


@st.fragment(run_every=3)
def _live_progress() -> None:
    s, err = call("GET", "/sync/status", admin=True)
    if err:
        st.error(err)
        return
    job = s["current_job"]
    if not job:
        st.success("No sync job running.")
        return
    c = job["counters"]
    st.markdown(f"**{job['kind'].title()} sync** · `{job['job_id'][:8]}` · {job['status']} · _{job['stage']}_")
    total, done = c.get("to_process") or 0, c.get("processed", 0)
    st.progress(min(done / total, 1.0) if total else 0.0, text=f"Images processed {done:,} / {total:,}")
    cols = st.columns(5)
    theme.kpi(cols[0], "Indexed", c.get("indexed", 0))
    theme.kpi(cols[1], "Skipped", c.get("skipped", 0), "unchanged")
    theme.kpi(cols[2], "Failed", c.get("failed", 0))
    theme.kpi(cols[3], "Products scanned", f"{c.get('products_scanned', 0):,}")
    theme.kpi(cols[4], "Images scanned", f"{c.get('images_scanned', 0):,}")


def _result(res, err, ok_msg: str) -> None:
    if err:
        st.error(err)
    else:
        st.success(ok_msg.format(**res))


def _scope_form() -> dict:
    """Filters for a full sync; returns the `scope` body for /sync/full and /sync/preview."""
    cats, err = call("GET", "/categories")
    options = [c["code"] for c in cats["categories"]] if cats else []
    if err:
        st.warning(f"Category list unavailable: {err}")
    chosen = st.multiselect("Categories", options, placeholder="All categories")
    lo, hi = st.columns(2)
    id_min = lo.number_input("ItemID from", min_value=0, value=0, step=1, help="0 = no lower bound")
    id_max = hi.number_input("ItemID to", min_value=0, value=0, step=1, help="0 = no upper bound")
    only_valid = st.checkbox("Only valid products (ItemValidSts = Y)")
    limit = st.number_input("Test sample: first N products only", min_value=0, value=0, step=10,
                            help="0 = all matching products. Use e.g. 20 to try the pipeline quickly.")
    return {"categories": chosen, "item_id_min": int(id_min) or None, "item_id_max": int(id_max) or None,
            "only_valid": only_valid, "max_products": int(limit) or None}


def render() -> None:
    theme.head("Sync admin", "Run and monitor synchronization from SQL Server. Jobs run in the worker, "
               "so closing this page never interrupts them.")
    _live_progress()
    st.write("")

    run_tab, fail_tab, hist_tab = st.tabs(["Run sync", "Failed images", "History"])

    with run_tab:
        a, b = st.columns(2)
        with a, st.container(border=True):
            st.subheader("Incremental")
            st.caption("New and changed products/images only. Also runs automatically every day.")
            cur, _err = call("GET", "/sync/scope", admin=True)
            if cur:
                st.caption(f"Scope ({cur['source'].replace('_', ' ')}): `{cur['scope'] or 'everything'}`")
            if st.button("Run incremental now", type="primary"):
                _result(*call("POST", "/sync/incremental", admin=True), "Queued job {job_id}")
        with b, st.container(border=True):
            st.subheader("Full sync / rebuild")
            st.caption("Embeds the selected images into a new collection and switches search to it only "
                       "after validation. The current index keeps serving meanwhile.")
            scope = _scope_form()
            sc_cols = st.columns(2)
            if sc_cols[0].button("Preview size"):
                res, err = call("POST", "/sync/preview", admin=True, json=scope, timeout=120)
                if err:
                    st.error(err)
                else:
                    st.info(f"{res['products']:,} products · {res['images']:,} image rows · "
                            f"{res['files']:,} distinct files to embed")
            ok = st.checkbox("I understand this rebuilds the index with the selection above")
            if sc_cols[1].button("Start full sync", type="primary", disabled=not ok):
                _result(*call("POST", "/sync/full", admin=True, json={"confirm": True, "scope": scope}),
                        "Queued job {job_id}")

    with fail_tab:
        fails, err = call("GET", "/sync/failures", admin=True, params={"limit": 200})
        if err:
            st.error(err)
        elif not fails["failures"]:
            st.success("No failed images.")
        else:
            st.dataframe(fails["failures"], use_container_width=True, hide_index=True)
            if st.button("Retry all failed"):
                _result(*call("POST", "/sync/retry-failed", admin=True),
                        "Re-queued {requeued_images} images (job {job_id})")

    with hist_tab:
        hist, err = call("GET", "/sync/history", admin=True, params={"limit": 50})
        if err:
            st.error(err)
        else:
            rows = [{k: j[k] for k in ("started_at", "finished_at", "kind", "trigger", "status", "error")}
                    | {k: j["counters"].get(k) for k in ("indexed", "skipped", "failed")} for j in hist["jobs"]]
            st.dataframe(rows, use_container_width=True, hide_index=True)
