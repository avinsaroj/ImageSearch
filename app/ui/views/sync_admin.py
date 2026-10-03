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
            if st.button("Run incremental now", type="primary"):
                _result(*call("POST", "/sync/incremental", admin=True), "Queued job {job_id}")
        with b, st.container(border=True):
            st.subheader("Full sync / rebuild")
            st.caption("Embeds every image into a new collection and switches search to it only after "
                       "validation. The current index keeps serving meanwhile.")
            ok = st.checkbox("I understand this re-embeds all images and can take hours")
            if st.button("Start full sync", disabled=not ok):
                _result(*call("POST", "/sync/full", admin=True, json={"confirm": True}), "Queued job {job_id}")

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
