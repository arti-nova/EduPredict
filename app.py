import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from batch_analysis import render_batch_page, apply_theme_fixes

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="EduPredict",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TEAL = "#14b8a6"
TEAL_DARK = "#0f766e"
RED = "#ef4444"

PAGES = [
    "Home",
    "Dashboard",
    "Predict Performance",
    "Batch Analysis",
    "Data Insights",
    "Prediction History",
    "Settings",
]

NAV_ICONS = ["🏠", "📊", "🔮", "🧮", "📈", "🕘", "⚙️"]

# ============================================================
# Theme and styling
# ============================================================
st.session_state.setdefault("dark_mode", False)
st.session_state.setdefault("prediction_history", [])
st.session_state.setdefault("selected_page", "Home")

LIGHT = {
    "bg": "#f3f7fb",
    "card": "#ffffff",
    "border": "#e3eaf3",
    "text": "#0b1f3a",
    "muted": "#5b6b82",
    "track": "#eef2f7",
    "alert": "#f1f9f8",
    "nav_bg": "rgba(255,255,255,.86)",
    "sidebar": "#0b1f3a",
    "sidebar2": "#12305a",
    "side_hover": "rgba(255,255,255,.10)",
    "hero": "linear-gradient(135deg, #e6f7f5 0%, #f3f7fb 45%, #e3edfb 100%)",
    "shadow": "rgba(15, 30, 61, .08)",
    "shadow_hover": "rgba(15, 30, 61, .20)",
}
DARK = {
    "bg": "#0b1220",
    "card": "#131c2e",
    "border": "#24324d",
    "text": "#e6ecf7",
    "muted": "#9fb0c9",
    "track": "#1d2a44",
    "alert": "#10263a",
    "nav_bg": "rgba(19,28,46,.88)",
    "sidebar": "#070d1a",
    "sidebar2": "#0d1a33",
    "side_hover": "rgba(255,255,255,.08)",
    "hero": "linear-gradient(135deg, #0f2b2a 0%, #0b1220 50%, #121a3a 100%)",
    "shadow": "rgba(0, 0, 0, .40)",
    "shadow_hover": "rgba(20, 184, 166, .25)",
}
THEME = DARK if st.session_state.dark_mode else LIGHT
theme_css_vars = ":root{" + "".join(f"--{key}:{value};" for key, value in THEME.items()) + "}"

CSS = """
/* ---------- keyframes ---------- */
@keyframes fadeUp { from {opacity:0; transform:translateY(16px);} to {opacity:1; transform:translateY(0);} }
@keyframes navDrop { from {opacity:0; transform:translateY(-18px);} to {opacity:1; transform:translateY(0);} }
@keyframes popIn { 0% {opacity:0; transform:scale(.92);} 60% {transform:scale(1.03);} 100% {opacity:1; transform:scale(1);} }
@keyframes gradientShift { 0% {background-position:0% 50%;} 50% {background-position:100% 50%;} 100% {background-position:0% 50%;} }
@keyframes floaty { 0%,100% {transform:translate(0,0);} 50% {transform:translate(-18px,16px);} }
@keyframes pulseDot { 0% {box-shadow:0 0 0 0 rgba(20,184,166,.6);} 70% {box-shadow:0 0 0 8px rgba(20,184,166,0);} 100% {box-shadow:0 0 0 0 rgba(20,184,166,0);} }
@keyframes logoFloat { 0%,100% {transform:translateY(0);} 50% {transform:translateY(-3px);} }

/* ---------- base ---------- */
html { scroll-behavior:smooth; }
.stApp { background: var(--bg); transition: background-color .4s ease, color .4s ease; }
header[data-testid="stHeader"], [data-testid="stToolbar"] { display:none !important; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"] { display:none !important; }
.block-container { padding-top:1rem; padding-bottom:3rem; max-width:1450px; }
.stApp, .stApp p, .stApp label, .stApp span, .stApp li,
[data-testid="stWidgetLabel"] *, [data-testid="stMarkdownContainer"] { color: var(--text); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: var(--text); }
[data-testid="stCaptionContainer"] * { color: var(--muted); }
::-webkit-scrollbar { width:9px; height:9px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:99px; }
::-webkit-scrollbar-thumb:hover { background: #14b8a6; }

/* ---------- top navbar ---------- */
/* the wrapper around the navbar is what must be sticky (its parent spans the whole page) */
div:has(> .st-key-navbar) { position:sticky; top:.5rem; z-index:1000; }
.st-key-navbar {
    background:var(--nav_bg); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
    border:1px solid var(--border); border-radius:18px; padding:8px 16px; margin-bottom:24px;
    box-shadow:0 10px 30px var(--shadow);
    animation:navDrop .6s cubic-bezier(.22,1,.36,1) both;
}
.st-key-navbar [data-testid="stHorizontalBlock"] { align-items:center; gap:.75rem; flex-wrap:nowrap; }
.st-key-navbar [data-testid="stRadioGroup"] {
    gap:4px; flex-wrap:nowrap; overflow-x:auto; scrollbar-width:none; justify-content:center; padding:2px;
}
.st-key-navbar [data-testid="stRadioGroup"]::-webkit-scrollbar { display:none; }
.st-key-navbar [data-testid="stRadioGroup"] > div { width:auto; flex:0 0 auto; }
.st-key-navbar [data-testid="stRadioOption"] {
    position:relative; cursor:pointer; white-space:nowrap; padding:7px 10px; border-radius:12px;
    background:transparent;
    transition:background .3s cubic-bezier(.4,0,.2,1), transform .3s cubic-bezier(.4,0,.2,1), box-shadow .3s ease;
}
.st-key-navbar [data-testid="stRadioOption"] > div > div:first-child { display:none; }   /* radio circle */
.st-key-navbar [data-testid="stRadioOption"] p { margin:0; font-size:.84rem; font-weight:600; color:var(--text); white-space:nowrap; }
.st-key-navbar [data-testid="stRadioOption"]:hover { background:var(--track); transform:translateY(-2px); }
.st-key-navbar [data-testid="stRadioOption"][data-selected="true"] {
    background:linear-gradient(135deg,#0f766e 0%,#14b8a6 100%);
    box-shadow:0 8px 18px rgba(20,184,166,.38); transform:translateY(-1px);
}
.st-key-navbar [data-testid="stRadioOption"][data-selected="true"] p { color:#fff !important; font-weight:700; }
.st-key-navbar [data-testid="stToggle"] { justify-content:flex-end; }
.brand-wrap { display:flex; align-items:center; gap:10px; margin:0; white-space:nowrap; }
.brand-logo {
    width:40px; height:40px; border-radius:12px; display:flex; align-items:center; justify-content:center;
    font-size:1.3rem; background:linear-gradient(135deg,#14b8a6,#0f766e);
    box-shadow:0 8px 20px rgba(20,184,166,.4); animation: logoFloat 4s ease-in-out infinite;
}
.brand { font-size:1.2rem; font-weight:800; line-height:1.1; color:var(--text) !important; }
.brand-sub { display:none; }
.status-row { display:flex; align-items:center; gap:9px; font-size:.85rem; margin:6px 0; }
.dot { width:9px; height:9px; border-radius:50%; background:#14b8a6; animation:pulseDot 2s infinite; }
.dot.off { background:#ef4444; animation:none; }

/* ---------- headings ---------- */
.page-title { font-size:2rem; font-weight:800; margin:0; color:var(--text) !important; animation: fadeUp .5s ease both; }
.page-sub { color:var(--muted) !important; margin:3px 0 22px; animation: fadeUp .6s ease both; }

/* ---------- hero ---------- */
.hero {
    position:relative; overflow:hidden; border:1px solid var(--border); border-radius:24px;
    padding:44px 40px; min-height:310px; background: var(--hero); background-size:220% 220%;
    box-shadow: 0 14px 40px var(--shadow); animation: gradientShift 14s ease infinite, fadeUp .7s ease both;
}
.hero::before, .hero::after {
    content:""; position:absolute; border-radius:50%; pointer-events:none;
    background: radial-gradient(circle, rgba(20,184,166,.32), transparent 70%);
}
.hero::before { width:300px; height:300px; right:-70px; top:-90px; animation: floaty 8s ease-in-out infinite; }
.hero::after { width:220px; height:220px; left:35%; bottom:-110px; animation: floaty 10s ease-in-out infinite reverse; }
.hero > * { position:relative; z-index:1; }
.hero h1 { font-size:clamp(2rem,4vw,3.1rem); line-height:1.08; margin:0 0 16px; font-weight:850; color:var(--text) !important; }
.hero p { color:var(--muted) !important; font-size:1.04rem; max-width:680px; line-height:1.65; }
.pill {
    display:inline-block; padding:6px 12px; border-radius:999px; margin-bottom:15px;
    background:rgba(20,184,166,.15); color:#0f766e !important; font-size:.78rem; font-weight:700;
    border:1px solid rgba(20,184,166,.3);
}

/* ---------- cards ---------- */
.feature-card, .stat-card {
    position:relative; overflow:hidden; background:var(--card); border:1px solid var(--border);
    border-radius:16px; padding:19px; box-shadow: 0 4px 16px var(--shadow);
    transition: transform .3s cubic-bezier(.4,0,.2,1), box-shadow .3s cubic-bezier(.4,0,.2,1), border-color .3s ease;
    animation: fadeUp .6s ease both;
}
.feature-card { min-height:125px; }
.stat-card { min-height:128px; }
.feature-card:hover, .stat-card:hover {
    transform: translateY(-6px); box-shadow: 0 18px 38px var(--shadow_hover); border-color:#14b8a6;
}
.stat-card::before {
    content:""; position:absolute; left:0; top:0; right:0; height:4px;
    background: linear-gradient(90deg,#14b8a6,#3b82f6,#8b5cf6); background-size:200% 100%;
    animation: gradientShift 6s linear infinite;
}
.stat-top { display:flex; align-items:center; gap:10px; }
.stat-icon {
    width:38px; height:38px; border-radius:11px; display:flex; align-items:center; justify-content:center;
    background:rgba(20,184,166,.14); font-size:1.15rem; transition: transform .3s ease;
}
.stat-card:hover .stat-icon { transform: rotate(-8deg) scale(1.12); }
.stat-label { color:var(--muted) !important; font-size:.84rem; font-weight:650; }
.stat-value { color:var(--text) !important; font-size:1.85rem; font-weight:850; line-height:1.4; margin-top:4px; }
.stat-note { color:var(--muted) !important; font-size:.77rem; }
.feature-icon { font-size:1.55rem; margin-bottom:8px; display:inline-block; transition: transform .3s ease; }
.feature-card:hover .feature-icon { transform: scale(1.18) rotate(-6deg); }
.feature-title { font-size:1rem; font-weight:750; margin-bottom:5px; }
.feature-desc { font-size:.86rem; color:var(--muted) !important; line-height:1.45; }

[data-testid="stVerticalBlockBorderWrapper"] {
    background:var(--card); border-radius:16px; border:1px solid var(--border);
    box-shadow:0 4px 16px var(--shadow); animation: fadeUp .6s ease both;
    transition: box-shadow .3s ease, border-color .3s ease, transform .3s ease, background-color .4s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover { box-shadow:0 14px 34px var(--shadow_hover); border-color:rgba(20,184,166,.55); }
[data-testid="stColumn"]:nth-child(2) [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stColumn"]:nth-child(2) .stat-card { animation-delay:.08s; }
[data-testid="stColumn"]:nth-child(3) .stat-card { animation-delay:.16s; }
[data-testid="stColumn"]:nth-child(4) .stat-card { animation-delay:.24s; }
.card-title { font-weight:750; font-size:1.04rem; margin-bottom:8px; color:var(--text) !important; }
.insight-text { font-size:.9rem; line-height:1.55; color:var(--muted) !important; }
.small-muted { font-size:.82rem; color:var(--muted) !important; }

/* ---------- result ---------- */
.result-banner { border-radius:14px; padding:23px; text-align:center; font-size:1.3rem; font-weight:800;
    color:#fff !important; animation: popIn .6s cubic-bezier(.22,1,.36,1) both; }
.result-banner * { color:#fff !important; }
.result-pass { background:linear-gradient(135deg,#0f766e,#14b8a6); box-shadow:0 12px 28px rgba(20,184,166,.4); }
.result-risk { background:linear-gradient(135deg,#b91c1c,#ef4444); box-shadow:0 12px 28px rgba(239,68,68,.4); }

/* ---------- buttons ---------- */
div.stButton > button, div.stDownloadButton > button {
    border-radius:12px; font-weight:700; min-height:44px; background:var(--card); border:1px solid var(--border);
    box-shadow:0 2px 8px var(--shadow);
    transition: transform .25s cubic-bezier(.4,0,.2,1), box-shadow .25s ease, background .25s ease, border-color .25s ease;
}
div.stButton > button p, div.stDownloadButton > button p { color:var(--text) !important; }
div.stButton > button:hover, div.stDownloadButton > button:hover {
    transform:translateY(-3px); box-shadow:0 10px 22px var(--shadow_hover); border-color:#14b8a6;
}
div.stButton > button:active, div.stDownloadButton > button:active { transform:translateY(0) scale(.98); }
div.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#0f766e,#14b8a6); border:none; box-shadow:0 8px 20px rgba(20,184,166,.35);
}
div.stButton > button[kind="primary"] p { color:#fff !important; }
div.stButton > button[kind="primary"]:hover { box-shadow:0 14px 28px rgba(20,184,166,.5); }

/* ---------- inputs (older Streamlit / BaseWeb; newer widgets are handled by apply_theme_fixes) ---------- */
[data-baseweb="select"] > div { background:var(--card); border-color:var(--border); transition:border-color .25s ease, box-shadow .25s ease; }
[data-baseweb="select"] > div:hover { border-color:#14b8a6; }
[data-baseweb="select"] * { color:var(--text) !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background:var(--card) !important; }
[data-baseweb="popover"] li { color:var(--text) !important; transition:background .2s ease; }
[data-baseweb="popover"] li:hover { background:var(--track) !important; }
[data-testid="stSlider"] [role="slider"] { background:#14b8a6 !important; box-shadow:0 0 0 4px rgba(20,184,166,.2); }
[data-testid="stProgress"] > div > div { background:var(--track) !important; }
[data-testid="stProgress"] > div > div > div { background:linear-gradient(90deg,#0f766e,#14b8a6) !important; transition:width .9s cubic-bezier(.22,1,.36,1); }
hr { border-color:var(--border); }

/* ---------- metrics / alerts / expanders / tabs ---------- */
[data-testid="stMetric"] {
    background:var(--card); border:1px solid var(--border); border-radius:14px; padding:14px 16px;
    box-shadow:0 3px 12px var(--shadow); transition:transform .25s ease, box-shadow .25s ease;
}
[data-testid="stMetric"]:hover { transform:translateY(-3px); box-shadow:0 12px 24px var(--shadow_hover); }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color:var(--text) !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color:var(--muted) !important; }
[data-testid="stAlert"] > div, [data-baseweb="notification"] {
    background:var(--alert) !important; border:1px solid var(--border) !important;
    border-left:4px solid #14b8a6 !important; border-radius:12px !important;
}
[data-testid="stAlert"] *, [data-baseweb="notification"] * { color:var(--text) !important; }
[data-testid="stExpander"] { background:var(--card); border:1px solid var(--border) !important; border-radius:14px; }
[data-testid="stExpander"] summary * { color:var(--text) !important; }
[data-testid="stCode"] pre, [data-testid="stCode"] code { background:var(--track) !important; color:var(--text) !important; }
[data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid var(--border); }
[data-baseweb="tab"] { background:transparent !important; border-radius:10px 10px 0 0; transition:background .25s ease; }
[data-baseweb="tab"] p { color:var(--muted) !important; font-weight:600; transition:color .25s ease; }
[data-baseweb="tab"]:hover { background:var(--track) !important; }
[data-baseweb="tab"][aria-selected="true"] p { color:var(--text) !important; }
[data-baseweb="tab-highlight"] { background:#14b8a6 !important; height:3px !important; border-radius:3px; }
[data-baseweb="tab-border"] { background:transparent !important; }

/* ---------- html tables (theme-aware replacement for st.dataframe) ---------- */
.tbl-wrap { max-height:420px; overflow:auto; border-radius:12px; border:1px solid var(--border); animation: fadeUp .5s ease both; }
table.tbl { width:100%; border-collapse:collapse; font-size:.84rem; }
table.tbl th { position:sticky; top:0; text-align:left; padding:10px 12px; background:var(--track); color:var(--text); white-space:nowrap; }
table.tbl td { padding:9px 12px; border-top:1px solid var(--border); color:var(--text); white-space:nowrap; }
table.tbl tbody tr { transition:background .2s ease; }
table.tbl tbody tr:hover { background:rgba(20,184,166,.10); }

@media (max-width: 900px) {
    div:has(> .st-key-navbar) { position:relative; }
    .st-key-navbar [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation:none !important; transition:none !important; }
}
"""

# Navbar icons: one emoji in front of each menu item
NAV_CSS = "".join(
    f'.st-key-navbar [data-testid="stRadioGroup"] > div:nth-of-type({i}) [data-testid="stMarkdownContainer"] p::before '
    f'{{ content:"{icon}"; margin-right:7px; display:inline-block; transition:transform .3s ease; }}'
    f'.st-key-navbar [data-testid="stRadioGroup"] > div:nth-of-type({i}) [data-testid="stRadioOption"]:hover '
    f'[data-testid="stMarkdownContainer"] p::before {{ transform:scale(1.25) rotate(-8deg); }}'
    for i, icon in enumerate(NAV_ICONS, start=1)
)

st.markdown(f"<style>{theme_css_vars}{CSS}{NAV_CSS}</style>", unsafe_allow_html=True)
apply_theme_fixes()   # dropdowns, tabs, alerts, file uploader and chart text for newer Streamlit versions

# ============================================================
# Model and dataset loading
# ============================================================
MODEL_PATH = "student_model.pkl"
SCALER_PATH = "feature_scaler.pkl"
DATASET_PATHS = [
    "student_performance_dataset.csv",
    os.path.join("data", "student_performance_dataset.csv"),
]

# These seven features must be in exactly the same order as model training.
FEATURE_NAMES = [
    "Study_Hours_Weekly",
    "Attendance_Pct",
    "Past_GPA",
    "Sleep_Hours_Daily",
    "Extracurricular",
    "Family_Relations",
    "Social_Engagement",
]


@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_dataset():
    for path in DATASET_PATHS:
        if os.path.isfile(path):
            raw_df = pd.read_csv(path)
            cleaned_df = raw_df.drop_duplicates().reset_index(drop=True)
            cleaned_df.attrs["raw_rows"] = len(raw_df)
            cleaned_df.attrs["source_path"] = path
            return cleaned_df
    return None


model = None
scaler = None
asset_error = None
try:
    model, scaler = load_model_assets()
except Exception as exc:
    asset_error = str(exc)

df = None
dataset_error = None
try:
    df = load_dataset()
except Exception as exc:
    dataset_error = str(exc)


# ============================================================
# Helpers
# ============================================================
def navigate(page_name):
    st.session_state.selected_page = page_name


def page_header(title, subtitle):
    st.markdown(
        f"<div class='page-title'>{title}</div><div class='page-sub'>{subtitle}</div>",
        unsafe_allow_html=True,
    )


def stat_card(icon, label, value, note):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-top"><div class="stat-icon">{icon}</div><div class="stat-label">{label}</div></div>
            <div class="stat-value">{value}</div>
            <div class="stat-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_table(frame):
    """Theme-aware table (st.dataframe ignores custom CSS and stays white in dark mode)."""
    html = frame.to_html(
        index=False,
        classes="tbl",
        border=0,
        float_format=lambda v: f"{v:,.2f}",
    )
    st.markdown(f"<div class='tbl-wrap'>{html}</div>", unsafe_allow_html=True)


def themed_chart(fig, height=320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=THEME["text"],
        legend_font_color=THEME["text"],
        hoverlabel=dict(bgcolor=THEME["card"], font_color=THEME["text"], bordercolor=TEAL),
        colorway=[TEAL, "#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444"],
        margin=dict(l=10, r=10, t=35, b=10),
        height=height,
        transition=dict(duration=500),
    )
    fig.update_xaxes(gridcolor=THEME["border"], zerolinecolor=THEME["border"])
    fig.update_yaxes(gridcolor=THEME["border"], zerolinecolor=THEME["border"])
    return fig


def find_column(frame, possible_names):
    """Find a column without depending on capitalization or spaces."""
    normalized = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in frame.columns
    }
    for name in possible_names:
        key = name.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def show_dataset_warning():
    if dataset_error:
        st.warning(f"Could not read the dataset: {dataset_error}")
    else:
        st.info(
            "Dataset not found. Add `student_performance_dataset.csv` beside "
            "`app.py` or inside the `data` folder to enable dataset charts."
        )


def show_history_table(records):
    if not records:
        st.caption("No predictions have been made in this session yet.")
        return
    render_table(pd.DataFrame(records[::-1]))


def correlation_chart(frame):
    numeric_df = frame.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        st.info("At least two numeric columns are needed for a correlation chart.")
        return
    corr = numeric_df.corr(numeric_only=True)
    fig = px.imshow(
        corr.round(2),
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Teal",
        title="Correlation between numeric features",
    )
    st.plotly_chart(themed_chart(fig, 450), use_container_width=True)


def status_row(ok, text):
    cls = "dot" if ok else "dot off"
    st.markdown(f"<div class='status-row'><span class='{cls}'></span>{text}</div>", unsafe_allow_html=True)


# ============================================================
# Top navigation bar (replaces the sidebar)
# ============================================================
def render_navbar():
    with st.container(key="navbar"):
        brand_col, nav_col, theme_col = st.columns([1.15, 7.0, 1.35], vertical_alignment="center")
        with brand_col:
            st.markdown(
                "<div class='brand-wrap'><div class='brand-logo'>🎓</div>"
                "<div class='brand'>EduPredict</div></div>",
                unsafe_allow_html=True,
            )
        with nav_col:
            st.radio(
                "Navigation",
                PAGES,
                key="selected_page",
                horizontal=True,
                label_visibility="collapsed",
            )
        with theme_col:
            st.toggle("🌙 Dark mode", key="dark_mode")


render_navbar()
page = st.session_state.selected_page

if asset_error:
    st.error(
        "The model or scaler could not be loaded. Ensure `student_model.pkl` "
        "and `feature_scaler.pkl` are in the same folder as `app.py`."
    )
    with st.expander("Technical details"):
        st.code(asset_error)


# ============================================================
# HOME / LANDING PAGE
# ============================================================
if page == "Home":
    left, right = st.columns([1.7, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="hero">
                <div class="pill">STUDENT SUCCESS • DATA-DRIVEN INSIGHTS</div>
                <h1>Understand performance.<br>Support potential.</h1>
                <p>
                    EduPredict uses a machine learning model to estimate whether
                    a student is on track or may need additional support. Explore
                    academic habits, review dataset patterns, and turn insights into
                    practical next steps.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        action1, action2 = st.columns(2)
        action1.button(
            "🔮  Start a Prediction",
            type="primary",
            use_container_width=True,
            on_click=navigate,
            args=("Predict Performance",),
        )
        action2.button(
            "📊  Open Dashboard",
            use_container_width=True,
            on_click=navigate,
            args=("Dashboard",),
        )

    with right:
        st.write("")
        st.write("")
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Performance Prediction</div>
                <div class="feature-desc">Estimate passing and at-risk probabilities from seven student indicators.</div>
            </div>
            <br>
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Data Analytics</div>
                <div class="feature-desc">Explore available records, distributions, and relationships between numeric features.</div>
            </div>
            <br>
            <div class="feature-card">
                <div class="feature-icon">💡</div>
                <div class="feature-title">Actionable Suggestions</div>
                <div class="feature-desc">Review simple, rule-based suggestions based on the entered student details.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### A clearer view of student success")
    f1, f2, f3 = st.columns(3)
    features = [
        (f1, "📚", "Academic Habits", "Study time, attendance, past score, and sleep."),
        (f2, "🤝", "Student Environment", "Family relations, social engagement, and activities."),
        (f3, "🧭", "Early Support", "Use predictions as a starting point for a supportive conversation."),
    ]
    for col, icon, title, description in features:
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption(
        "Note: Predictions are estimates from the trained model, not a definitive "
        "judgment of a student's ability."
    )


# ============================================================
# DASHBOARD
# ============================================================
elif page == "Dashboard":
    page_header("Dashboard", "A snapshot of student records and prediction activity.")

    if df is not None and not df.empty:
        total_records = len(df)
        numeric_df = df.select_dtypes(include=np.number)

        pass_col = find_column(df, ["Pass_Fail", "Pass Status", "Result", "Status"])
        score_col = find_column(
            df,
            ["Final_Exam_Score", "Final Score", "G3", "Performance_Score", "Score"],
        )
        study_col = find_column(
            df,
            ["Study_Hours_Weekly", "Study_Hours_per_Week", "Study Hours", "Weekly_Study_Hours"],
        )

        pass_rate_value = None
        at_risk_count = None
        if pass_col:
            result_text = df[pass_col].astype(str).str.strip().str.lower()
            pass_mask = result_text.isin(["pass", "passed", "passing", "on-track", "on track"])
            fail_mask = result_text.isin(["fail", "failed", "at-risk", "at risk"])
            if (pass_mask | fail_mask).any():
                pass_rate_value = pass_mask[pass_mask | fail_mask].mean() * 100
                at_risk_count = int(fail_mask.sum())

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            stat_card("👥", "Student Records", f"{total_records:,}", "After removing duplicates")
        with k2:
            if score_col:
                stat_card(
                    "🎓",
                    "Average Score",
                    f"{pd.to_numeric(df[score_col], errors='coerce').mean():.1f}",
                    f"Column: {score_col}",
                )
            else:
                stat_card("📊", "Numeric Features", f"{numeric_df.shape[1]}", "Available numeric columns")
        with k3:
            if pass_rate_value is not None:
                stat_card("✅", "Pass Rate", f"{pass_rate_value:.1f}%", "Based on the dataset result column")
            else:
                stat_card("🔮", "Predictions", f"{len(st.session_state.prediction_history)}", "Made this session")
        with k4:
            if at_risk_count is not None:
                stat_card("⚠️", "At-Risk / Failed", f"{at_risk_count:,}", "Based on the dataset result column")
            else:
                stat_card("🧠", "Model", "Ready" if model is not None else "Unavailable", "Seven-feature prediction model")

        st.write("")
        chart1, chart2 = st.columns(2, gap="large")

        with chart1:
            with st.container(border=True):
                st.markdown("<div class='card-title'>Numeric Feature Distribution</div>", unsafe_allow_html=True)
                if score_col:
                    values = pd.to_numeric(df[score_col], errors="coerce").dropna()
                    if not values.empty:
                        fig = px.histogram(values, x=score_col, nbins=20, color_discrete_sequence=[TEAL])
                        fig.update_layout(xaxis_title=score_col, yaxis_title="Number of students")
                        st.plotly_chart(themed_chart(fig), use_container_width=True)
                    else:
                        st.info("The score column contains no numeric values.")
                elif not numeric_df.empty:
                    selected_num = st.selectbox("Choose a numeric feature", list(numeric_df.columns), key="dashboard_num")
                    fig = px.histogram(df, x=selected_num, nbins=20, color_discrete_sequence=[TEAL])
                    st.plotly_chart(themed_chart(fig), use_container_width=True)
                else:
                    st.info("No numeric columns are available for charting.")

        with chart2:
            with st.container(border=True):
                st.markdown("<div class='card-title'>Study Hours vs Score</div>", unsafe_allow_html=True)
                if study_col and score_col:
                    plot_df = df[[study_col, score_col]].copy()
                    plot_df[study_col] = pd.to_numeric(plot_df[study_col], errors="coerce")
                    plot_df[score_col] = pd.to_numeric(plot_df[score_col], errors="coerce")
                    plot_df = plot_df.dropna()
                    if not plot_df.empty:
                        fig = px.scatter(
                            plot_df,
                            x=study_col,
                            y=score_col,
                            opacity=0.75,
                            color_discrete_sequence=[TEAL],
                        )
                        st.plotly_chart(themed_chart(fig), use_container_width=True)
                    else:
                        st.info("There are no usable numeric values for these columns.")
                elif not numeric_df.empty:
                    st.markdown(
                        "<div class='insight-text'>Your dataset does not contain recognized "
                        "study-hour and final-score columns. Explore correlations in Data Insights.</div>",
                        unsafe_allow_html=True,
                    )
                    st.metric("Numeric columns available", len(numeric_df.columns))
                else:
                    st.info("No numeric columns are available for charting.")

        c1, c2 = st.columns([1.2, 1], gap="large")
        with c1:
            with st.container(border=True):
                st.markdown("<div class='card-title'>Recent Predictions</div>", unsafe_allow_html=True)
                show_history_table(st.session_state.prediction_history[-5:])
        with c2:
            with st.container(border=True):
                st.markdown("<div class='card-title'>Quick Actions</div>", unsafe_allow_html=True)
                st.button(
                    "🔮 Predict Performance",
                    type="primary",
                    use_container_width=True,
                    on_click=navigate,
                    args=("Predict Performance",),
                )
                st.button(
                    "📈 Explore Data Insights",
                    use_container_width=True,
                    on_click=navigate,
                    args=("Data Insights",),
                )
    else:
        show_dataset_warning()
        st.write("")
        st.markdown("### Session overview")
        k1, k2 = st.columns(2)
        with k1:
            stat_card("🔮", "Predictions this session", len(st.session_state.prediction_history), "History is kept during this session")
        with k2:
            stat_card("🧠", "Model status", "Ready" if model is not None else "Unavailable", "Model and scaler files are required")


# ============================================================
# PREDICTION PAGE — seven features matching original app.py
# ============================================================
elif page == "Predict Performance":
    page_header(
        "Predict Student Performance",
        "Enter all seven indicators in the same feature order used to train your model.",
    )

    if model is None or scaler is None:
        st.error("Prediction is unavailable until the model and scaler load successfully.")
    else:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Student Information</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='small-muted'>These values are passed to the saved scaler and model. "
                "Keep the training feature order unchanged.</div>",
                unsafe_allow_html=True,
            )
            st.write("")
            col1, col2 = st.columns(2, gap="large")

            with col1:
                study_hours = st.slider(
                    "Weekly Study Hours",
                    min_value=2.0,
                    max_value=35.0,
                    value=14.0,
                    step=0.5,
                    help="Total hours spent studying in a typical week.",
                )
                attendance = st.slider(
                    "Attendance Percentage",
                    min_value=40.0,
                    max_value=100.0,
                    value=82.0,
                    step=0.5,
                )
                past_gpa = st.slider(
                    "Past GPA / Score",
                    min_value=3.0,
                    max_value=10.0,
                    value=7.2,
                    step=0.1,
                )
                sleep_hours = st.slider(
                    "Daily Sleep Hours",
                    min_value=4.0,
                    max_value=10.0,
                    value=7.0,
                    step=0.5,
                )

            with col2:
                extracurricular_choice = st.radio(
                    "Extracurricular Activities",
                    ["No", "Yes"],
                    horizontal=True,
                )
                family_relations = st.slider(
                    "Family Relations Quality",
                    min_value=1,
                    max_value=5,
                    value=4,
                    help="1 = Poor, 5 = Excellent",
                )
                social_engagement = st.slider(
                    "Social Engagement Level",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="1 = Low, 5 = High",
                )
                st.caption("The last two ratings use a 1–5 scale.")

            predict_clicked = st.button(
                "🔍 Predict Performance",
                type="primary",
                use_container_width=True,
            )

        if predict_clicked:
            input_data = np.array(
                [[
                    study_hours,
                    attendance,
                    past_gpa,
                    sleep_hours,
                    1 if extracurricular_choice == "Yes" else 0,
                    family_relations,
                    social_engagement,
                ]],
                dtype=float,
            )

            try:
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_scaled)[0]
                    classes = list(getattr(model, "classes_", [0, 1]))
                    probability_map = {
                        cls: float(prob)
                        for cls, prob in zip(classes, probabilities)
                    }
                    # The existing app treats class 1 as passing and class 0 as at-risk.
                    pass_probability = probability_map.get(1, 0.0) * 100
                    risk_probability = probability_map.get(0, 0.0) * 100
                else:
                    pass_probability = 100.0 if prediction == 1 else 0.0
                    risk_probability = 100.0 - pass_probability

                is_passing = str(prediction).strip().lower() in ["1", "pass", "passed", "true"]
                status = "On-Track" if is_passing else "At-Risk"

                record = {
                    "Date": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                    "ID": f"{len(st.session_state.prediction_history) + 1:03d}",
                    "Study Hours": study_hours,
                    "Attendance (%)": attendance,
                    "Past GPA / Score": past_gpa,
                    "Sleep Hours": sleep_hours,
                    "Extracurricular": extracurricular_choice,
                    "Family Relations": family_relations,
                    "Social Engagement": social_engagement,
                    "Status": status,
                    "At-Risk Probability (%)": round(risk_probability, 1),
                    "Passing Probability (%)": round(pass_probability, 1),
                }
                st.session_state.prediction_history.append(record)

                st.write("")
                result_col, confidence_col = st.columns(2, gap="large")
                with result_col:
                    with st.container(border=True):
                        st.markdown("<div class='card-title'>Prediction Result</div>", unsafe_allow_html=True)
                        if is_passing:
                            st.markdown(
                                "<div class='result-banner result-pass'>✅ On-Track (Passing)</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                "<div class='result-banner result-risk'>⚠️ At-Risk — Needs Support</div>",
                                unsafe_allow_html=True,
                            )
                        st.write("")
                        st.markdown(
                            "<div class='small-muted'>Use this result as a screening signal, "
                            "not as a definitive judgment of student ability.</div>",
                            unsafe_allow_html=True,
                        )

                with confidence_col:
                    with st.container(border=True):
                        st.markdown("<div class='card-title'>Model Probabilities</div>", unsafe_allow_html=True)
                        st.write(f"**Passing probability:** {pass_probability:.1f}%")
                        st.progress(max(0, min(100, int(round(pass_probability)))))
                        st.write(f"**At-risk probability:** {risk_probability:.1f}%")
                        st.progress(max(0, min(100, int(round(risk_probability)))))

                suggestions = []
                if attendance < 75:
                    suggestions.append("📅 Attendance is below 75%. Consider support to improve regular attendance.")
                if study_hours < 10:
                    suggestions.append("📚 Study time is relatively low. A consistent weekly study plan may help.")
                if sleep_hours < 6:
                    suggestions.append("😴 Sleep is below 6 hours per day. Encourage a healthier rest routine.")
                if past_gpa < 6:
                    suggestions.append("🎯 Consider tutoring, mentoring, or reviewing foundational concepts.")
                if family_relations <= 2:
                    suggestions.append("🤝 Consider a supportive check-in and access to student support services.")
                if social_engagement <= 2:
                    suggestions.append("🌱 Look for comfortable, optional opportunities to connect with peers.")
                with st.container(border=True):
                    st.markdown("<div class='card-title'>Suggested Next Steps</div>", unsafe_allow_html=True)
                    for suggestion in suggestions or [
                        "🌟 The entered indicators look generally positive. Continue monitoring progress and maintaining healthy routines."
                    ]:
                        st.write(suggestion)

            except Exception as exc:
                st.error(
                    "Prediction failed. Confirm that the scaler and model were trained "
                    "with exactly these seven features, in the same order."
                )
                with st.expander("Technical details"):
                    st.code(str(exc))


# ============================================================
# BATCH ANALYSIS (UI unchanged — lives in batch_analysis.py)
# ============================================================
elif page == "Batch Analysis":
    render_batch_page(model, scaler, FEATURE_NAMES, page_header, stat_card, render_table, themed_chart)


# ============================================================
# DATA INSIGHTS
# ============================================================
elif page == "Data Insights":
    page_header("Data Insights", "Explore distributions, correlations, and the available dataset.")

    if df is None or df.empty:
        show_dataset_warning()
    else:
        numeric_df = df.select_dtypes(include=np.number)
        tab_corr, tab_feature, tab_preview = st.tabs(
            ["Correlation Analysis", "Feature Explorer", "Dataset Preview"]
        )

        with tab_corr:
            with st.container(border=True):
                st.markdown("<div class='card-title'>Correlation Heatmap</div>", unsafe_allow_html=True)
                correlation_chart(df)
                st.caption("Correlation describes association, not proof of cause and effect.")

        with tab_feature:
            with st.container(border=True):
                st.markdown("<div class='card-title'>Explore a Feature</div>", unsafe_allow_html=True)
                selected_column = st.selectbox("Choose a dataset column", list(df.columns))
                series = df[selected_column]
                if pd.api.types.is_numeric_dtype(series):
                    chart_data = pd.to_numeric(series, errors="coerce").dropna()
                    if not chart_data.empty:
                        fig = px.histogram(
                            chart_data,
                            x=selected_column,
                            nbins=25,
                            color_discrete_sequence=[TEAL],
                        )
                        fig.update_layout(xaxis_title=selected_column, yaxis_title="Count")
                        st.plotly_chart(themed_chart(fig), use_container_width=True)
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Mean", f"{chart_data.mean():.2f}")
                        m2.metric("Median", f"{chart_data.median():.2f}")
                        m3.metric("Minimum", f"{chart_data.min():.2f}")
                        m4.metric("Maximum", f"{chart_data.max():.2f}")
                    else:
                        st.info("No numeric values are available in this column.")
                else:
                    counts = series.fillna("Missing").astype(str).value_counts().reset_index()
                    counts.columns = [selected_column, "Count"]
                    fig = px.bar(
                        counts,
                        x=selected_column,
                        y="Count",
                        color_discrete_sequence=[TEAL],
                    )
                    fig.update_layout(xaxis_title=selected_column, yaxis_title="Count")
                    st.plotly_chart(themed_chart(fig), use_container_width=True)
                    render_table(counts)

        with tab_preview:
            with st.container(border=True):
                st.markdown("<div class='card-title'>Dataset Preview</div>", unsafe_allow_html=True)
                st.write(f"Showing the first 50 of **{len(df):,}** unique rows.")
                render_table(df.head(50))
                st.write("")
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download cleaned dataset (CSV)",
                    data=csv_bytes,
                    file_name="student_performance_cleaned.csv",
                    mime="text/csv",
                )


# ============================================================
# PREDICTION HISTORY
# ============================================================
elif page == "Prediction History":
    page_header("Prediction History", "Review predictions made during this app session.")

    with st.container(border=True):
        st.markdown("<div class='card-title'>Previous Predictions</div>", unsafe_allow_html=True)
        if st.session_state.prediction_history:
            history_df = pd.DataFrame(st.session_state.prediction_history[::-1])
            render_table(history_df)
            st.write("")
            st.download_button(
                "⬇️ Export prediction history (CSV)",
                data=history_df.to_csv(index=False).encode("utf-8"),
                file_name="student_prediction_history.csv",
                mime="text/csv",
            )
            if st.button("🗑️ Clear prediction history"):
                st.session_state.prediction_history = []
                st.success("Prediction history cleared.")
                st.rerun()
        else:
            st.info("No predictions yet. Open Predict Performance to create your first prediction.")


# ============================================================
# SETTINGS
# ============================================================
elif page == "Settings":
    page_header("Settings", "Check application assets and manage session preferences.")

    with st.container(border=True):
        st.markdown("<div class='card-title'>System Status</div>", unsafe_allow_html=True)
        status_row(
            model is not None and scaler is not None,
            "Prediction model loaded" if model is not None and scaler is not None else "Prediction model unavailable",
        )
        status_row(df is not None, f"{len(df):,} unique dataset rows" if df is not None else "Dataset not loaded")

    with st.container(border=True):
        st.markdown("<div class='card-title'>Model Files</div>", unsafe_allow_html=True)
        model_col, scaler_col = st.columns(2)
        with model_col:
            if os.path.isfile(MODEL_PATH):
                st.success(f"Found `{MODEL_PATH}`")
            else:
                st.error(f"Missing `{MODEL_PATH}`")
        with scaler_col:
            if os.path.isfile(SCALER_PATH):
                st.success(f"Found `{SCALER_PATH}`")
            else:
                st.error(f"Missing `{SCALER_PATH}`")

        st.write("")
        st.markdown("<div class='card-title'>Dataset</div>", unsafe_allow_html=True)
        if df is None:
            st.warning("Dataset not found.")
        else:
            raw_rows = df.attrs.get("raw_rows", len(df))
            st.success(f"Loaded {len(df):,} unique rows and {len(df.columns)} columns.")
            st.caption(f"Source: {df.attrs.get('source_path', 'CSV file')}")
            st.caption(f"Duplicate rows removed: {raw_rows - len(df):,}")

        st.write("")
        st.markdown("<div class='card-title'>Session</div>", unsafe_allow_html=True)
        st.write(f"Predictions made this session: **{len(st.session_state.prediction_history)}**")
        if st.button("Clear prediction history", use_container_width=False):
            st.session_state.prediction_history = []
            st.success("Prediction history cleared.")

    with st.container(border=True):
        st.markdown("<div class='card-title'>About EduPredict</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="insight-text">
                EduPredict is a student performance analytics demonstration.
                It combines a saved machine learning model with visual summaries of
                the available dataset. Predictions and recommendations should be
                reviewed alongside real academic context and student support needs.
            </div>
            """,
            unsafe_allow_html=True,
        )