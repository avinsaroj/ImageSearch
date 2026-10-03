import html
import io

import streamlit as st
from PIL import Image

from app.ui import theme
from app.ui.client import call

COLS = 4


@st.dialog("Product details", width="large")
def product_dialog(r: dict) -> None:
    best = r["best_image"]
    left, right = st.columns([3, 2])
    with left:
        st.image(best["ImageURL"], use_container_width=True)
    with right:
        st.markdown(f"### {r['item_code']}")
        st.markdown(
            theme.chip("Plain gold", r["is_plain_gold"]) + theme.chip("Solitaire", r["is_solitaire"])
            + theme.chip("Valid", r["is_valid"]) + theme.chip("Franchise", r["is_franchise_item"]),
            unsafe_allow_html=True)
        st.write(f"**Category:** {r['category_code']}")
        st.write(f"**ItemID:** {r['item_id']}")
        st.write(f"**Status remark:** {r['status_remark'] or '-'}")
        st.write(f"**Cosine similarity:** {r['score']:.4f}")
        st.caption("Per vector space: " + ", ".join(f"{k} {v:.3f}" for k, v in r["score_per_space"].items()))
        st.caption(f"Best match: ImgID {best['ImgID']} · view {best['ImgViewID']} · "
                   f"{r['matched_images']} image(s) of this product matched")
        st.link_button("Open original image", best["ImageURL"])
    imgs = r["images"] or [best]
    st.markdown(f"**All images ({len(imgs)})**")
    cols = st.columns(min(len(imgs), 5))
    for i, img in enumerate(imgs):
        star = " ★ best match" if img["ImgID"] == best["ImgID"] else ""
        with cols[i % len(cols)]:
            st.image(img["ImageURL"], caption=f"Img {img['ImgID']} · view {img['ImgViewID']}{star}")


def _card(r: dict, key: str) -> None:
    with st.container(border=True):
        st.image(r["best_image"]["ImageURL"], use_container_width=True)
        st.markdown(
            f'<div class="pcode">{html.escape(str(r["item_code"]))}'
            f'<span class="score">{r["score"]:.3f}</span></div>'
            f'<div class="pmeta">#{r["rank"]} · {html.escape(str(r["category_code"]))} · '
            f'{len(r["images"])} image(s)</div>'
            + theme.chip("Solitaire", r["is_solitaire"]) + theme.chip("Valid", r["is_valid"]),
            unsafe_allow_html=True)
        if st.button("Details", key=key, use_container_width=True):
            product_dialog(r)


def render() -> None:
    theme.head("Find similar jewellery", "Upload a reference photo. Each product is shown once, "
               "ranked by its best matching image.")

    cats, err = call("GET", "/categories")
    options = ["All"] + [c["code"] for c in (cats or {}).get("categories", [])]
    if err:
        st.warning(f"Category list unavailable ({err}). Search still works without a category filter.")

    up_col, prev_col = st.columns([3, 1])
    with up_col:
        upload = st.file_uploader("Reference image", type=["jpg", "jpeg", "png", "webp"],
                                  label_visibility="collapsed")
    with prev_col:
        if upload:
            st.image(Image.open(upload), caption=upload.name, use_container_width=True)

    with st.expander("Filters & ranking", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        category = c1.selectbox("Category", options)
        plain_gold = c2.segmented_control("Plain gold", ["All", "Yes", "No"], default="All")
        solitaire = c3.segmented_control("Solitaire", ["All", "Yes", "No"], default="All")
        valid = c4.segmented_control("Valid", ["All", "Yes", "No"], default="All")
        d1, d2, d3, d4 = st.columns(4)
        franchise = d1.segmented_control("Franchise item", ["All", "Yes", "No"], default="All")
        item_codes = d2.text_input("Item code(s)", placeholder="comma separated")
        top_k = d3.slider("Products to return", 4, 60, 12, step=4)
        min_score = d4.slider("Min cosine similarity", 0.0, 1.0, 0.0, 0.01)
        mode = st.radio("Ranking model", ["fusion", "clip", "dino"], horizontal=True,
                        format_func={"fusion": "CLIP + DINOv2 (fusion)", "clip": "CLIP only",
                                     "dino": "DINOv2 only"}.get)

    if st.button("Search", type="primary", disabled=upload is None):
        data = {"top_k": top_k, "mode": mode, "plain_gold": (plain_gold or "All").lower(),
                "solitaire": (solitaire or "All").lower(), "valid": (valid or "All").lower(),
                "franchise": (franchise or "All").lower()}
        if category != "All":
            data["category"] = category
        if item_codes.strip():
            data["item_codes"] = item_codes
        if min_score > 0:
            data["min_score"] = min_score
        upload.seek(0)
        files = {"file": (upload.name, io.BytesIO(upload.read()), upload.type or "image/jpeg")}
        with st.spinner("Searching…"):
            res, err = call("POST", "/search/image", data=data, files=files, timeout=120)
        if err:
            st.error(err)
            st.session_state.pop("search_res", None)
        else:
            st.session_state["search_res"] = res

    res = st.session_state.get("search_res")
    if not res:
        if upload is None:
            st.info("Upload a jewellery image to begin.")
        return
    st.divider()
    st.markdown(f"**{res['total_returned']} products** &nbsp;·&nbsp; "
                f"<span style='color:#9a8d7d'>score: {html.escape(res['score_type'])}</span>",
                unsafe_allow_html=True)
    if not res["results"]:
        st.info("No products matched the filters / similarity threshold. Try lowering the threshold.")
        return
    for start in range(0, len(res["results"]), COLS):
        cols = st.columns(COLS)
        for col, r in zip(cols, res["results"][start:start + COLS]):
            with col:
                _card(r, key=f"d_{r['item_id']}")
