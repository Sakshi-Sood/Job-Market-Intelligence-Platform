"""
Streamlit interactive dashboard for the AI-Powered Job Market Analytics Platform.
"""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sqlalchemy import text
from datetime import datetime

from backend.database import engine
from analytics.metrics import (
    get_employment_distribution,
    get_top_cities,
    get_remote_distribution,
    get_data_quality_metrics,
    get_top_companies,
)
from data_pipeline.skill_extractor import get_top_skills
from data_pipeline.nlp_pipeline import get_role_distribution, get_skills_by_role

# ================================================================
#  PAGE CONFIG — must be the first Streamlit call
# ================================================================

st.set_page_config(
    page_title="Job Market Intelligence",
    page_icon="JMI",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# ================================================================
#  THEME & CUSTOM CSS
# ================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --bg-base:    #06060f;
    --bg-card:    #0d0d1a;
    --bg-raised:  #141425;
    --border:     #1e1e35;
    --accent:     #7c6aff;
    --accent2:    #56c596;
    --accent3:    #ff6b6b;
    --accent4:    #3dc1d3;
    --accent5:    #ffd700;
    --text-hi:    #eeeef5;
    --text-mid:   #8a8aa8;
    --text-lo:    #45455a;
    --font-mono:  'Space Mono', monospace;
    --font-body:  'Inter', 'DM Sans', sans-serif;
    --glow-purple: rgba(124, 106, 255, 0.15);
    --glow-green:  rgba(86, 197, 150, 0.15);
    --glow-red:    rgba(255, 107, 107, 0.15);
    --glow-cyan:   rgba(61, 193, 211, 0.15);
}

/* ── Global reset ── */
html, body, .stApp {
    background-color: var(--bg-base) !important;
    color: var(--text-hi) !important;
    font-family: var(--font-body) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2.5rem 2rem !important; max-width: 1440px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a18 0%, #0d0d1a 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-hi) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: var(--text-mid) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: var(--font-mono) !important;
}

/* ── Metric cards with unique glow accents ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(13, 13, 26, 0.85) 0%, rgba(20, 20, 37, 0.65) 100%) !important;
    backdrop-filter: blur(16px);
    border: 1px solid rgba(124, 106, 255, 0.12) !important;
    border-radius: 14px;
    padding: 1.2rem 1.3rem;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.03);
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent4));
    opacity: 0.6;
    transition: opacity 0.3s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    border-color: rgba(124, 106, 255, 0.5) !important;
    box-shadow: 0 12px 35px rgba(124, 106, 255, 0.15), 0 4px 15px rgba(0, 0, 0, 0.3);
}
[data-testid="stMetric"]:hover::before {
    opacity: 1;
}
[data-testid="stMetric"] label {
    color: var(--text-mid) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: var(--font-mono) !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-hi) !important;
    font-family: var(--font-mono) !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #eeeef5 0%, #c0c0e0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Section dividers ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; opacity: 0.5; }

/* ── Plotly chart wrappers ── */
.js-plotly-plot {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(30, 30, 53, 0.8) !important;
    transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}
.js-plotly-plot:hover {
    border-color: rgba(124, 106, 255, 0.25) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4), 0 0 40px rgba(124, 106, 255, 0.05);
    transform: translateY(-2px);
}

/* ── Premium Tab Toggles ── */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(135deg, rgba(13, 13, 26, 0.8) 0%, rgba(20, 20, 37, 0.6) 100%) !important;
    backdrop-filter: blur(12px);
    border-radius: 12px;
    padding: 5px 6px;
    gap: 6px;
    border: 1px solid var(--border) !important;
    margin-bottom: 1.5rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-mid) !important;
    border-radius: 9px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 9px 24px !important;
    transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    border: none !important;
    position: relative;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-hi) !important;
    background: rgba(124, 106, 255, 0.08) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c6aff 0%, #9b8bff 100%) !important;
    color: white !important;
    box-shadow: 0 4px 18px rgba(124, 106, 255, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Select / multiselect ── */
[data-baseweb="select"] > div {
    background: var(--bg-raised) !important;
    border-color: var(--border) !important;
    color: var(--text-hi) !important;
    border-radius: 10px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
    background: var(--accent) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* ── Primary Button ── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent) 0%, #9b8bff 50%, #b4a7ff 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    color: white !important;
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.06em !important;
    transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    box-shadow: 0 4px 18px rgba(124, 106, 255, 0.35) !important;
}
div.stButton > button[kind="primary"]:hover {
    transform: scale(1.04) translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124, 106, 255, 0.55) !important;
}

/* ── Live Pulse Indicator ── */
@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(86, 197, 150, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(86, 197, 150, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(86, 197, 150, 0); }
}
.pulse-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background-color: #56c596;
    border-radius: 50%;
    animation: pulse 2.0s infinite;
}

/* ── Shimmer Animation for header ── */
@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}
.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(
        120deg,
        #eeeef5 0%, #eeeef5 40%,
        #7c6aff 50%,
        #eeeef5 60%, #eeeef5 100%
    );
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 6s ease-in-out infinite;
}

/* ── Animated border for section headers ── */
@keyframes borderGlow {
    0%, 100% { border-image-source: linear-gradient(90deg, #7c6aff, #56c596, #3dc1d3); }
    50% { border-image-source: linear-gradient(90deg, #3dc1d3, #7c6aff, #56c596); }
}

/* ── Info/empty state cards ── */
.empty-state-card {
    background: linear-gradient(135deg, rgba(13, 13, 26, 0.9) 0%, rgba(20, 20, 37, 0.7) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(124, 106, 255, 0.15);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}
.empty-state-card .es-icon {
    margin-bottom: 0.75rem;
    display: flex;
    justify-content: center;
}
.empty-state-card .title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #eeeef5;
    margin-bottom: 0.5rem;
}
.empty-state-card .description {
    font-size: 0.85rem;
    color: #8a8aa8;
    line-height: 1.6;
}

/* ── Section badge ── */
.section-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 8px;
    margin-bottom: 1rem;
}
.section-badge svg {
    flex-shrink: 0;
}

/* ── Sidebar section icon ── */
.sidebar-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 7px;
    flex-shrink: 0;
}

/* ── Footer ── */
.dashboard-footer {
    margin-top: 3rem;
    padding: 1.5rem 0;
    border-top: 1px solid var(--border);
    text-align: center;
}
.dashboard-footer p {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-lo);
    letter-spacing: 0.08em;
    margin: 0;
}
.dashboard-footer .accent {
    color: var(--accent);
}
</style>
""", unsafe_allow_html=True)


# ================================================================
#  PLOTLY THEME HELPER
# ================================================================

BG_BASE  = "#06060f"
BG_CARD  = "#0d0d1a"
TEXT_HI  = "#eeeef5"
TEXT_MID = "#8a8aa8"
GRID     = "#1e1e35"
ACCENT   = "#7c6aff"

PALETTE = [
    "#7c6aff", "#56c596", "#ff6b6b", "#ffd700",
    "#3dc1d3", "#ff8e72", "#b388ff", "#ea80fc",
    "#a3de83", "#ffb347",
]

def dark_layout(fig, title="", xaxis_title="", yaxis_title="", height=380):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color=TEXT_HI,
                   family="Space Mono, monospace"), x=0.02),
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(family="Inter, DM Sans, sans-serif", color=TEXT_HI),
        xaxis=dict(title=xaxis_title, showgrid=False,
                   linecolor=GRID, tickfont=dict(size=11, color=TEXT_MID)),
        yaxis=dict(title=yaxis_title, gridcolor=GRID, gridwidth=0.5,
                   linecolor=GRID, tickfont=dict(size=11, color=TEXT_MID)),
        margin=dict(l=50, r=30, t=55, b=50),
        height=height,
        hoverlabel=dict(bgcolor="#141425", font_size=12,
                        font_color=TEXT_HI, bordercolor=ACCENT),
    )
    return fig

# ================================================================
#  DATA LOADING — cached for performance
# ================================================================

@st.cache_data(ttl=300)
def load_data(search_query: str = ""):
    try:
        if search_query.strip():
            df = pd.read_sql(
                text("SELECT * FROM jobs WHERE LOWER(search_query) LIKE :q ORDER BY last_updated DESC"),
                engine,
                params={"q": f"%{search_query.strip().lower().split()[0]}%"}
            )
            if df.empty:
                df = pd.read_sql(text("SELECT * FROM jobs ORDER BY last_updated DESC"), engine)
        else:
            df = pd.read_sql(text("SELECT * FROM jobs ORDER BY last_updated DESC"), engine)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


def apply_filters(df, cities, emp_types, remote_only):
    """Apply sidebar filters to the main DataFrame."""
    filtered = df.copy()

    if cities:
        filtered = filtered[filtered["job_city"].isin(cities)]

    if emp_types:
        filtered = filtered[filtered["job_employment_type"].isin(emp_types)]

    if remote_only:
        filtered = filtered[filtered["job_is_remote"].fillna(False) == True]

    return filtered

# ================================================================
#  HEADER
# ================================================================

st.markdown("""
<div style="
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 0.2rem;
">
    <span class="header-title">Job Market Intelligence</span>
    <span style="
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: #7c6aff;
        background: linear-gradient(135deg, #1a1830 0%, #1e1640 100%);
        border: 1px solid #7c6aff33;
        border-radius: 6px;
        padding: 4px 12px;
        letter-spacing: 0.1em;
        font-weight: 600;
    ">ML + DEVOPS</span>
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: #56c596;
        background: linear-gradient(135deg, #0a1f18 0%, #0f251d 100%);
        border: 1px solid #56c59625;
        border-radius: 6px;
        padding: 4px 12px;
        letter-spacing: 0.1em;
        font-weight: 600;
    ">
        <span class="pulse-dot"></span> LIVE DATA
    </span>
</div>
<p style="color: #8a8aa8; font-size: 0.85rem; margin-bottom: 1.5rem; font-family: 'Inter', 'DM Sans', sans-serif; line-height: 1.5;">
    Real-time analytics on tech job listings — skills, roles, and hiring trends across the market.
</p>
""", unsafe_allow_html=True)

# ================================================================
#  LOAD DATA
# ================================================================

df_raw, load_error = load_data(st.session_state.last_query)

if load_error:
    st.error(f"Database connection failed: {load_error}")
    st.info("Make sure PostgreSQL is running and your .env is configured correctly.")
    st.stop()

if df_raw.empty:
    st.warning("No data found. Run the ETL pipeline first: `python -m data_pipeline.fetch_jobs`")
    st.stop()

# ================================================================
#  SIDEBAR — FETCH NEW DATA + FILTERS
# ================================================================

with st.sidebar:

    # ── Section: Fetch New Data ──────────────────────────────────
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.92rem;
                font-weight:700; color:#eeeef5; margin-bottom:1rem;
                padding-bottom:0.75rem; border-bottom:1px solid #1e1e35;
                display:flex; align-items:center; gap:10px;">
        <span class="sidebar-icon" style="background:rgba(124,106,255,0.12);">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7c6aff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
        </span> Fetch New Data
    </div>
    """, unsafe_allow_html=True)

    search_query = st.text_input(
        "Job Search Query",
        value="Data Scientist in India",
        placeholder="e.g. ML Engineer in USA",
        help="Enter any job title, role, or keyword + location",
    )

    num_pages = st.slider(
        "Pages to fetch",
        min_value=1,
        max_value=5,
        value=1,
        help="Each page returns ~10 jobs. More pages = more data but slower.",
    )

    custom_skills_input = st.text_input(
        "Custom Skills (comma-separated)",
        value="",
        placeholder="e.g. dax, power automate, looker studio",
        help="Add extra skills to track beyond the built-in 70+ skills",
    )

    # Parse custom skills
    custom_skills_list = None
    if custom_skills_input.strip():
        custom_skills_list = [
            s.strip() for s in custom_skills_input.split(",") if s.strip()
        ]

    fetch_clicked = st.button(
        "Fetch Jobs",
        use_container_width=True,
        type="primary",
    )

    if fetch_clicked:
        if not search_query.strip():
            st.error("Please enter a search query.")
        else:
            from data_pipeline.fetch_jobs import run_pipeline
            from data_pipeline.skill_extractor import run_skill_extraction
            from data_pipeline.nlp_pipeline import run_nlp_pipeline

            progress = st.progress(0, text="Starting pipeline...")

            try:
                progress.progress(10, text="[1/3] Fetching jobs from API...")
                run_pipeline(query=search_query.strip(), num_pages=num_pages)

                progress.progress(45, text="[2/3] Extracting skills...")
                run_skill_extraction(custom_skills=custom_skills_list)

                progress.progress(75, text="[3/3] Running NLP pipeline...")
                run_nlp_pipeline()

                progress.progress(100, text="Pipeline complete!")
                st.success(
                    f"Fetched jobs for **\"{search_query.strip()}\"** "
                    f"({num_pages} page{'s' if num_pages > 1 else ''})."
                )

                # Clear cached data so the dashboard refreshes
                st.session_state.last_query = search_query.strip()
                load_data.clear()
                st.rerun()

            except Exception as e:
                progress.progress(100, text="Pipeline failed")
                st.error(f"Pipeline error: {e}")

    st.divider()

    # ── Section: Filters ────────────────────────────────────────
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.92rem;
                font-weight:700; color:#eeeef5; margin-bottom:1rem;
                padding-bottom:0.75rem; border-bottom:1px solid #1e1e35;
                display:flex; align-items:center; gap:10px;">
        <span class="sidebar-icon" style="background:rgba(86,197,150,0.12);">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#56c596" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        </span> Filters
    </div>
    """, unsafe_allow_html=True)

    # City filter
    valid_cities = sorted(
        df_raw[df_raw["job_city"] != "Not Specified"]["job_city"].dropna().unique()
    )
    selected_cities = st.multiselect(
        "City", valid_cities, placeholder="All cities"
    )

    # Employment type
    valid_emp = sorted(
        df_raw[df_raw["job_employment_type"] != "Not Specified"]["job_employment_type"].dropna().unique()
    )
    selected_emp = st.multiselect(
        "Employment Type", valid_emp, placeholder="All types"
    )

    # Toggles
    remote_only = st.toggle("Remote jobs only", value=False)

    st.divider()

    # Dataset info
    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace; font-size:0.68rem; color:#45455a;
                text-transform:uppercase; letter-spacing:0.1em;">
        DATASET<br>
        <span style="color:#8a8aa8; font-size:0.78rem; font-weight:600;">{len(df_raw):,} total records</span>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
#  APPLY FILTERS
# ================================================================

df = apply_filters(df_raw, selected_cities, selected_emp, remote_only)

if df.empty:
    st.markdown("""
    <div class="empty-state-card">
        <div class="es-icon">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#7c6aff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        </div>
        <div class="title">No Results Found</div>
        <div class="description">No jobs match the current filters. Try adjusting your city, employment type, or remote settings.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ================================================================
#  TABS
# ================================================================

tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Skills & Roles",
    "Job Explorer",
])

# ──────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW (Home Page)
# ──────────────────────────────────────────────────────────────────

with tab1:

    # ── KPI Metrics Row ──────────────────────────────────────────
    remote_count   = int(df["job_is_remote"].fillna(False).sum())
    remote_pct     = round(remote_count / len(df) * 100, 1) if len(df) > 0 else 0

    nlp_available = (
        "role_category" in df.columns
        and df["role_category"].notna().any()
    )
    unique_roles  = df["role_category"].nunique() if nlp_available else 0

    skill_jobs = (
        (df["skill_count"] > 0).sum()
        if "skill_count" in df.columns else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Jobs", f"{len(df):,}")
    with col2:
        st.metric("Remote Jobs", f"{remote_count:,}", f"{remote_pct}%")
    with col3:
        st.metric("Roles Identified", str(unique_roles) if nlp_available else "Run NLP")
    with col4:
        st.metric("Jobs with Skills", f"{skill_jobs:,}")

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # ── Section Header ───────────────────────────────────────────
    st.markdown("""
    <div class="section-badge" style="color:#7c6aff; background:rgba(124,106,255,0.08); border:1px solid rgba(124,106,255,0.15);">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#7c6aff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
        MARKET DISTRIBUTION
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    # ── Top Hiring Cities ────────────────────────────────────────
    with col_left:
        top_cities = get_top_cities(df, top_n=10)

        if not top_cities.empty:
            fig = go.Figure(go.Bar(
                x=top_cities.values[::-1],
                y=top_cities.index[::-1],
                orientation="h",
                marker=dict(
                    color=PALETTE[:len(top_cities)][::-1],
                    opacity=0.9,
                    line=dict(width=0),
                ),
                text=[f"{v:,}" for v in top_cities.values[::-1]],
                textposition="outside",
                textfont=dict(size=11, color=TEXT_HI, family="Space Mono, monospace"),
                hovertemplate="<b>%{y}</b><br>%{x:,} jobs<extra></extra>",
            ))
            dark_layout(fig, "Top Hiring Cities", "Number of Jobs", height=380)
            fig.update_layout(
                xaxis=dict(range=[0, top_cities.max() * 1.22]),
                margin=dict(l=120, r=60, t=55, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Top Companies ────────────────────────────────────────────
    with col_right:
        top_companies = get_top_companies(df, top_n=10)

        if not top_companies.empty:
            fig = go.Figure(go.Bar(
                x=top_companies.index,
                y=top_companies.values,
                marker=dict(
                    color=PALETTE[:len(top_companies)],
                    opacity=0.9,
                    line=dict(width=0),
                ),
                text=[f"{v:,}" for v in top_companies.values],
                textposition="outside",
                textfont=dict(size=11, color=TEXT_HI, family="Space Mono, monospace"),
                hovertemplate="<b>%{x}</b><br>%{y:,} listings<extra></extra>",
            ))
            dark_layout(fig, "Top Hiring Companies", height=380)
            fig.update_layout(
                xaxis_tickangle=-30,
                yaxis=dict(range=[0, top_companies.max() * 1.22]),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Employment Type + Remote Distribution ────────────────────
    st.markdown("""
    <div class="section-badge" style="color:#56c596; background:rgba(86,197,150,0.08); border:1px solid rgba(86,197,150,0.15);">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#56c596" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>
        EMPLOYMENT BREAKDOWN
    </div>
    """, unsafe_allow_html=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        emp_dist = get_employment_distribution(df)

        if not emp_dist.empty:
            fig = go.Figure(go.Pie(
                labels=emp_dist.index,
                values=emp_dist.values,
                hole=0.55,
                marker=dict(
                    colors=PALETTE[:len(emp_dist)],
                    line=dict(color=BG_BASE, width=2.5),
                ),
                textinfo="label+percent",
                textfont=dict(size=11, family="Inter, sans-serif"),
                hovertemplate="<b>%{label}</b><br>%{value:,} jobs<br>%{percent}<extra></extra>",
            ))
            fig.add_annotation(
                text=f"<b>{emp_dist.sum():,}</b><br><span style='font-size:10px;color:{TEXT_MID}'>Jobs</span>",
                x=0.5, y=0.5, font=dict(size=20, color=TEXT_HI),
                showarrow=False, xref="paper", yref="paper",
            )
            dark_layout(fig, "Employment Type", height=370)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_right2:
        remote_dist = get_remote_distribution(df)
        label_map   = {True: "Remote", False: "On-site", "True": "Remote", "False": "On-site"}
        labels      = [label_map.get(l, str(l)) for l in remote_dist.index]

        remote_colors = ["#56c596" if "Remote" in str(l) else "#ff6b6b" for l in labels]

        fig = go.Figure(go.Bar(
            x=labels,
            y=remote_dist.values,
            marker=dict(
                color=remote_colors,
                opacity=0.85,
                line=dict(width=0),
            ),
            text=[
                f"{v:,}<br>({v/remote_dist.sum()*100:.1f}%)"
                for v in remote_dist.values
            ],
            textposition="outside",
            textfont=dict(size=12, color=TEXT_HI, family="Space Mono, monospace"),
            hovertemplate="<b>%{x}</b><br>%{y:,} jobs<extra></extra>",
        ))
        dark_layout(fig, "Remote vs On-site", height=370)
        fig.update_layout(
            yaxis=dict(range=[0, remote_dist.max() * 1.28]),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# TAB 2 — SKILLS & ROLES
# ──────────────────────────────────────────────────────────────────

with tab2:

    nlp_available_tab2 = (
        "role_category" in df.columns
        and df["role_category"].notna().any()
    )

    if not nlp_available_tab2:
        st.markdown("""
        <div class="empty-state-card">
            <div class="es-icon">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#3dc1d3" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>
            </div>
            <div class="title">NLP Pipeline Required</div>
            <div class="description">
                Skills and role data haven't been generated yet. Run the NLP pipeline to unlock these insights:
                <div style="margin-top: 1rem; text-align: left; display: inline-block;">
                    <code style="background:#141425; padding:8px 14px; border-radius:8px; font-size:0.82rem; color:#56c596; display:block; margin-bottom:6px;">
                        python -m data_pipeline.skill_extractor
                    </code>
                    <code style="background:#141425; padding:8px 14px; border-radius:8px; font-size:0.82rem; color:#56c596; display:block;">
                        python -m data_pipeline.nlp_pipeline
                    </code>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Section Header ───────────────────────────────────────
        st.markdown("""
        <div class="section-badge" style="color:#3dc1d3; background:rgba(61,193,211,0.08); border:1px solid rgba(61,193,211,0.15);">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3dc1d3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            IN-DEMAND SKILLS & ROLE MAPPING
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2])

        # ── Top Skills ───────────────────────────────────────────
        with col_left:
            top_skills = get_top_skills(df, top_n=20)

            if not top_skills.empty:
                skills = top_skills.index.tolist()[::-1]
                counts = top_skills.values.tolist()[::-1]

                # Vibrant multi-color palette for skills
                colors_cycle = (PALETTE * 3)[:len(skills)][::-1]

                fig = go.Figure(go.Bar(
                    x=counts,
                    y=skills,
                    orientation="h",
                    marker=dict(color=colors_cycle, opacity=0.9, line=dict(width=0)),
                    text=[f"{v:,}" for v in counts],
                    textposition="outside",
                    textfont=dict(size=10, color=TEXT_HI, family="Space Mono, monospace"),
                    hovertemplate="<b>%{y}</b><br>%{x:,} jobs<extra></extra>",
                ))
                dark_layout(fig, "Most In-Demand Skills",
                            "Job listings mentioning skill", height=580)
                fig.update_layout(
                    xaxis=dict(range=[0, max(counts) * 1.18]),
                    margin=dict(l=140, r=60, t=55, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Role Distribution ────────────────────────────────────
        with col_right:
            role_dist = get_role_distribution(df)

            if not role_dist.empty:
                fig = go.Figure(go.Pie(
                    labels=role_dist.index,
                    values=role_dist.values,
                    hole=0.52,
                    marker=dict(
                        colors=PALETTE[:len(role_dist)],
                        line=dict(color=BG_BASE, width=2.5),
                    ),
                    pull=[0.05 if v == role_dist.max() else 0 for v in role_dist.values],
                    textinfo="label+percent",
                    textfont=dict(size=10),
                    hovertemplate="<b>%{label}</b><br>%{value:,} jobs<br>%{percent}<extra></extra>",
                ))
                fig.add_annotation(
                    text=f"<b>{role_dist.sum():,}</b><br><span style='font-size:10px;color:{TEXT_MID}'>Mapped</span>",
                    x=0.5, y=0.5, font=dict(size=20, color=TEXT_HI),
                    showarrow=False, xref="paper", yref="paper",
                )
                dark_layout(fig, "Role Distribution", height=360)
                fig.update_layout(
                    showlegend=True,
                    legend=dict(
                        orientation="v", x=1.02, y=0.5,
                        font=dict(size=10, color=TEXT_MID),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Skills by Role ───────────────────────────────────────
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-badge" style="color:#ffd700; background:rgba(255,215,0,0.06); border:1px solid rgba(255,215,0,0.15);">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ffd700" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
            TOP SKILLS BY ROLE
        </div>
        """, unsafe_allow_html=True)

        skills_by_role = get_skills_by_role(df)

        if not skills_by_role.empty:
            roles = skills_by_role["role_category"].unique().tolist()
            cols  = st.columns(min(len(roles), 4))

            for i, role in enumerate(roles[:8]):
                role_skills = (
                    skills_by_role[skills_by_role["role_category"] == role]
                    .sort_values("count", ascending=False)
                    .head(5)
                )
                with cols[i % 4]:
                    color = PALETTE[i % len(PALETTE)]
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(13,13,26,0.9) 0%, rgba(20,20,37,0.7) 100%);
                        border: 1px solid {color}22;
                        border-top: 2px solid {color};
                        border-radius: 12px;
                        padding: 1rem 1.1rem;
                        margin-bottom: 0.75rem;
                        backdrop-filter: blur(8px);
                        transition: all 0.3s ease;
                    ">
                        <div style="font-family:'Space Mono',monospace;
                                    font-size:0.7rem; color:{color};
                                    letter-spacing:0.1em; margin-bottom:0.6rem;
                                    font-weight:700;">
                            {role.upper()}
                        </div>
                        {''.join(
                            f'<div style="display:flex; justify-content:space-between;'
                            f'padding:4px 0; border-bottom:1px solid rgba(30,30,53,0.5);">'
                            f'<span style="font-size:0.82rem; color:#eeeef5; font-family:Inter,sans-serif;">{row["skill"]}</span>'
                            f'<span style="font-size:0.75rem; color:{color}; font-family:Space Mono,monospace; font-weight:700;">{row["count"]}</span>'
                            f'</div>'
                            for _, row in role_skills.iterrows()
                        )}
                    </div>
                    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# TAB 3 — JOB EXPLORER (formerly Raw Data)
# ──────────────────────────────────────────────────────────────────

with tab3:

    # ── Section Header ───────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem;">
        <div class="section-badge" style="color:#ff8e72; background:rgba(255,142,114,0.08); border:1px solid rgba(255,142,114,0.15); margin-bottom:0;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ff8e72" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
            BROWSE ALL LISTINGS
        </div>
        <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#8a8aa8;">
            <span style="color:#56c596; font-weight:700;">{len(df):,}</span> records after filters
        </div>
    </div>
    """, unsafe_allow_html=True)

    display_cols = [
        c for c in [
            "job_title", "employer_name", "job_city", "job_country",
            "job_employment_type", "job_is_remote",
            "role_category", "extracted_skills", "job_posted_at",
        ] if c and c in df.columns
    ]

    display_df = df[display_cols].copy()

    # Search
    search = st.text_input(
        "Search job titles, companies, or skills...",
        placeholder="e.g. Data Scientist, Google, Python...",
        label_visibility="collapsed",
    )
    if search:
        search_mask = pd.Series([False] * len(display_df), index=display_df.index)
        for col in ["job_title", "employer_name", "extracted_skills"]:
            if col in display_df.columns:
                search_mask |= display_df[col].astype(str).str.contains(search, case=False, na=False)
        display_df = display_df[search_mask]

    if display_df.empty:
        st.markdown("""
        <div class="empty-state-card">
            <div class="es-icon">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#ff8e72" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
            </div>
            <div class="title">No Matches</div>
            <div class="description">No job listings match your search. Try a different keyword or clear the search field.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Rename columns for better display
        col_renames = {
            "job_title": "Job Title",
            "employer_name": "Company",
            "job_city": "City",
            "job_country": "Country",
            "job_employment_type": "Type",
            "job_is_remote": "Remote",
            "role_category": "Role",
            "extracted_skills": "Skills",
            "job_posted_at": "Posted",
        }
        display_df = display_df.rename(columns={k: v for k, v in col_renames.items() if k in display_df.columns})

        st.dataframe(
            display_df,
            use_container_width=True,
            height=550,
            hide_index=True,
        )

# ================================================================
#  FOOTER
# ================================================================

st.markdown(f"""
<div class="dashboard-footer">
    <p>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#7c6aff" stroke-width="2" style="vertical-align: middle; margin-right: 6px;"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
        Job Market Intelligence Platform&ensp;·&ensp;
        Built with Streamlit + Plotly&ensp;·&ensp;
        Last refreshed: <span class="accent">{datetime.now().strftime("%b %d, %Y at %I:%M %p")}</span>
    </p>
</div>
""", unsafe_allow_html=True)