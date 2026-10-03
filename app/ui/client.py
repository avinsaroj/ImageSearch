"""Thin HTTP client shared by the Streamlit views."""

import os

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")


def api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL).rstrip("/")


def admin_headers() -> dict:
    token = os.getenv("ADMIN_API_TOKEN") or st.session_state.get("admin_token", "")
    return {"X-Admin-Token": token} if token else {}


def call(method: str, path: str, admin: bool = False, **kw):
    """Return (json|None, error|None). Surfaces the API's `detail` message on errors."""
    try:
        r = requests.request(method, f"{api_url()}{path}", headers=admin_headers() if admin else None,
                             timeout=kw.pop("timeout", 30), **kw)
    except requests.RequestException as exc:
        return None, f"Cannot reach API: {type(exc).__name__}"
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"{r.status_code}: {detail}"
    return r.json(), None


def api_online() -> bool:
    try:
        return requests.get(f"{api_url()}/openapi.json", timeout=3).status_code == 200
    except requests.RequestException:
        return False
