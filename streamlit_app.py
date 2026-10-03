"""Kisna Jewellery Search - Streamlit entry point (router + shared sidebar).

Run: streamlit run streamlit_app.py     (API_URL env var points at the FastAPI service)
"""

import os

import streamlit as st

from app.ui import theme
from app.ui.client import DEFAULT_API_URL, api_online
from app.ui.views import dashboard, search, sync_admin

st.set_page_config(page_title="Jewellery Search", page_icon="💎", layout="wide")
theme.inject()

pages = [
    st.Page(search.render, title="Search", icon="🔍", url_path="search", default=True),
    st.Page(dashboard.render, title="Dashboard", icon="📊", url_path="dashboard"),
    st.Page(sync_admin.render, title="Sync admin", icon="🔄", url_path="sync"),
]

with st.sidebar:
    st.markdown('<div class="brand">Jewellery Search<small>Visual similarity</small></div>',
                unsafe_allow_html=True)
    st.write("")

nav = st.navigation(pages)

with st.sidebar:
    st.divider()
    online = api_online()
    color = "#4caf50" if online else "#e57373"
    st.markdown(f'<span class="state-dot" style="background:{color}"></span>'
                f'{"API connected" if online else "API unreachable"}', unsafe_allow_html=True)
    with st.expander("Connection"):
        st.text_input("API base URL", value=DEFAULT_API_URL, key="api_url")
        if not os.getenv("ADMIN_API_TOKEN"):
            st.text_input("Admin token", type="password", key="admin_token",
                          help="Needed for Dashboard and Sync admin.")

nav.run()
