import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@500;600&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; letter-spacing: .01em; }
.block-container { padding-top: 2rem; max-width: 1280px; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(201,169,110,.18); }

.brand { font-family: 'Playfair Display', serif; font-size: 1.5rem; color: #c9a96e; line-height: 1.1; }
.brand small { display:block; font-family:'Inter',sans-serif; font-size:.65rem; letter-spacing:.2em;
  text-transform:uppercase; color:#8a7c6a; margin-top:.25rem; }

.page-head h1 { margin: 0 0 .15rem 0; font-size: 2rem; }
.page-head p { margin: 0 0 1.2rem 0; color: #9a8d7d; font-size: .92rem; }

.kpi { background: rgba(201,169,110,.06); border: 1px solid rgba(201,169,110,.18);
  border-radius: 12px; padding: 1rem 1.1rem; height: 100%; }
.kpi .v { font-family:'Playfair Display',serif; font-size: 1.9rem; color:#e8d9bd; line-height:1.15; }
.kpi .l { font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; color:#8a7c6a; }
.kpi .s { font-size:.75rem; color:#9a8d7d; margin-top:.2rem; }

.chip { display:inline-block; padding:.12rem .55rem; margin:0 .3rem .3rem 0; border-radius:999px;
  font-size:.7rem; border:1px solid rgba(201,169,110,.3); color:#d8c7a6; }
.chip.yes { background: rgba(76,175,80,.14); border-color: rgba(76,175,80,.4); color:#9fd8a2; }
.chip.no  { background: rgba(229,115,115,.10); border-color: rgba(229,115,115,.35); color:#e8a0a0; }
.chip.na  { opacity:.55; }

.pcode { font-weight:600; font-size:1rem; margin:.4rem 0 .1rem 0; }
.pmeta { color:#9a8d7d; font-size:.75rem; margin-bottom:.4rem; }
.score { float:right; font-size:.75rem; padding:.1rem .5rem; border-radius:6px;
  background: rgba(201,169,110,.15); color:#e8d9bd; }
.state-dot { display:inline-block; width:.55rem; height:.55rem; border-radius:50%; margin-right:.4rem; }
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def head(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-head"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def kpi(col, label: str, value, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="l">{html.escape(label)}</div>'
                 f'<div class="v">{html.escape(str(value))}</div>'
                 f'<div class="s">{html.escape(sub)}</div></div>', unsafe_allow_html=True)


def chip(label: str, value) -> str:
    cls, txt = ("yes", "Yes") if value is True else ("no", "No") if value is False else ("na", "n/a")
    return f'<span class="chip {cls}">{html.escape(label)}: {txt}</span>'
