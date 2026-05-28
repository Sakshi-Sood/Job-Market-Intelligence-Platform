"""
Streamlit interactive dashboard for the AI-Powered Job Market Analytics Platform.
"""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from backend.database import engine
from analytics.metrics import (
    get_employment_distribution,
    get_top_cities,
    get_remote_distribution,
    get_salary_metrics,
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
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
#  THEME & CUSTOM CSS
# ================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --bg-base:    #090912;
    --bg-card:    #11111e;
    --bg-raised:  #181828;
    --border:     #2a2a40;
    --accent:     #7c6aff;
    --accent2:    #56c596;
    --accent3:    #ff6b6b;
    --text-hi:    #eeeef5;
    --text-mid:   #9898b0;
    --text-lo:    #55556a;
    --font-mono:  'Space Mono', monospace;
    --font-body:  'DM Sans', sans-serif;
}

/* ── Global reset ── */
html, body, .stApp {
    background-color: var(--bg-base) !important;
    color: var(--text-hi) !important;
    font-family: var(--font-body) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-hi) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: var(--text-mid) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetric"] label {
    color: var(--text-mid) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: var(--font-mono) !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-hi) !important;
    font-family: var(--font-mono) !important;
    font-size: 1.9rem !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Section dividers ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Plotly chart backgrounds ── */
.js-plotly-plot { border-radius: 10px; overflow: hidden; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-mid) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    padding: 6px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* ── Select / multiselect ── */
[data-baseweb="select"] > div {
    background: var(--bg-raised) !important;
    border-color: var(--border) !important;
    color: var(--text-hi) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
    background: var(--accent) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
#  PLOTLY THEME HELPER
# ================================================================

BG_BASE  = "#090912"
BG_CARD  = "#11111e"
TEXT_HI  = "#eeeef5"
TEXT_MID = "#9898b0"
GRID     = "#2a2a40"
ACCENT   = "#7c6aff"

PALETTE = [
    "#7c6aff", "#56c596", "#ff6b6b", "#ffd700",
    "#3dc1d3", "#ff8e72", "#b388ff", "#ea80fc",
    "#a3de83", "#ffb347",
]

def dark_layout(fig, title="", xaxis_title="", yaxis_title="", height=380):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, color=TEXT_HI,
                   family="Space Mono, monospace"), x=0.02),
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(family="DM Sans, sans-serif", color=TEXT_HI),
        xaxis=dict(title=xaxis_title, showgrid=False,
                   linecolor=GRID, tickfont=dict(size=11, color=TEXT_MID)),
        yaxis=dict(title=yaxis_title, gridcolor=GRID, gridwidth=0.5,
                   linecolor=GRID, tickfont=dict(size=11, color=TEXT_MID)),
        margin=dict(l=50, r=30, t=55, b=50),
        height=height,
        hoverlabel=dict(bgcolor="#1e1e30", font_size=12,
                        font_color=TEXT_HI, bordercolor=ACCENT),
    )
    return fig

# ================================================================
#  DATA LOADING — cached for performance
# ================================================================

@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_data():
    try:
        df = pd.read_sql(text("SELECT * FROM jobs"), engine)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


def apply_filters(df, cities, emp_types, remote_only, salary_only):
    """Apply sidebar filters to the main DataFrame."""
    filtered = df.copy()

    if cities:
        filtered = filtered[filtered["job_city"].isin(cities)]

    if emp_types:
        filtered = filtered[filtered["job_employment_type"].isin(emp_types)]

    if remote_only:
        filtered = filtered[filtered["job_is_remote"].fillna(False) == True]

    if salary_only:
        filtered = filtered[filtered["salary_available"].fillna(False) == True]

    return filtered

# ================================================================
#  HEADER
# ================================================================

st.markdown("""
<div style="
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 0.25rem;
">
    <span style="
        font-family: 'Space Mono', monospace;
        font-size: 1.7rem;
        font-weight: 700;
        color: #eeeef5;
        letter-spacing: -0.02em;
    ">Job Market Intelligence</span>
    <span style="
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        color: #7c6aff;
        background: #1a1830;
        border: 1px solid #7c6aff44;
        border-radius: 4px;
        padding: 3px 10px;
        letter-spacing: 0.08em;
    ">ML + DEVOPS</span>
</div>
<p style="color: #9898b0; font-size: 0.88rem; margin-bottom: 1.5rem; font-family: 'DM Sans';">
    Real-time analytics on tech job listings — skills, roles, salaries, and hiring trends.
</p>
""", unsafe_allow_html=True)

# ================================================================
#  LOAD DATA
# ================================================================

df_raw, load_error = load_data()

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
    <div style="font-family:'Space Mono',monospace; font-size:0.95rem;
                font-weight:700; color:#eeeef5; margin-bottom:1rem;
                padding-bottom:0.75rem; border-bottom:1px solid #2a2a40;">
        🔄 Fetch New Data
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
        "🚀 Fetch Jobs",
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

                progress.progress(100, text="✅ Pipeline complete!")
                st.success(
                    f"Fetched jobs for **\"{search_query.strip()}\"** "
                    f"({num_pages} page{'s' if num_pages > 1 else ''})."
                )

                # Clear cached data so the dashboard refreshes
                load_data.clear()
                st.rerun()

            except Exception as e:
                progress.progress(100, text="❌ Pipeline failed")
                st.error(f"Pipeline error: {e}")

    st.divider()

    # ── Section: Filters ────────────────────────────────────────
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.95rem;
                font-weight:700; color:#eeeef5; margin-bottom:1rem;
                padding-bottom:0.75rem; border-bottom:1px solid #2a2a40;">
        ⚙ Filters
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
    salary_only = st.toggle("Has salary data", value=False)

    st.divider()

    # Dataset info
    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#55556a;">
        DATASET<br>
        <span style="color:#9898b0;">{len(df_raw):,} total records</span>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
#  APPLY FILTERS
# ================================================================

df = apply_filters(df_raw, selected_cities, selected_emp, remote_only, salary_only)

if df.empty:
    st.warning("No results match the current filters. Try adjusting them.")
    st.stop()

# ================================================================
#  KPI METRICS ROW
# ================================================================

salary_metrics = get_salary_metrics(df)
remote_count   = int(df["job_is_remote"].fillna(False).sum())
remote_pct     = round(remote_count / len(df) * 100, 1) if len(df) > 0 else 0

# NLP metrics
nlp_available = (
    "role_category" in df.columns
    and df["role_category"].notna().any()
)
unique_roles  = df["role_category"].nunique() if nlp_available else 0

skill_jobs = (
    (df["skill_count"] > 0).sum()
    if "skill_count" in df.columns else 0
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Jobs", f"{len(df):,}")
with col2:
    st.metric("Remote Jobs", f"{remote_count:,}", f"{remote_pct}%")
with col3:
    avg_sal = salary_metrics.get("average_salary")
    st.metric(
        "Avg Salary",
        f"${avg_sal:,.0f}" if avg_sal else "N/A",
        f"{salary_metrics.get('count', 0)} with data"
    )
with col4:
    st.metric("Roles Identified", str(unique_roles) if nlp_available else "Run NLP")
with col5:
    st.metric("Jobs with Skills", f"{skill_jobs:,}")

st.markdown("<hr>", unsafe_allow_html=True)

# ================================================================
#  TABS
# ================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📍  Market Overview",
    "🧠  Skills & Roles",
    "💰  Salary Analysis",
    "🔎  Raw Data",
])

# ──────────────────────────────────────────────────────────────────
# TAB 1 — MARKET OVERVIEW
# ──────────────────────────────────────────────────────────────────

with tab1:

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
                textfont=dict(size=11, color=TEXT_HI),
                hovertemplate="<b>%{y}</b><br>%{x:,} jobs<extra></extra>",
            ))
            dark_layout(fig, "Top Hiring Cities", "Number of Jobs", height=380)
            fig.update_layout(
                xaxis=dict(range=[0, top_cities.max() * 1.2]),
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
                textfont=dict(size=11, color=TEXT_HI),
                hovertemplate="<b>%{x}</b><br>%{y:,} listings<extra></extra>",
            ))
            dark_layout(fig, "Top Hiring Companies", height=380)
            fig.update_layout(
                xaxis_tickangle=-30,
                yaxis=dict(range=[0, top_companies.max() * 1.2]),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Employment Type + Remote Distribution ────────────────────
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        emp_dist = get_employment_distribution(df)

        if not emp_dist.empty:
            fig = go.Figure(go.Pie(
                labels=emp_dist.index,
                values=emp_dist.values,
                hole=0.52,
                marker=dict(
                    colors=PALETTE[:len(emp_dist)],
                    line=dict(color=BG_BASE, width=2),
                ),
                textinfo="label+percent",
                textfont=dict(size=11),
                hovertemplate="<b>%{label}</b><br>%{value:,} jobs<br>%{percent}<extra></extra>",
            ))
            fig.add_annotation(
                text=f"<b>{emp_dist.sum():,}</b><br><span style='font-size:11px;color:{TEXT_MID}'>Jobs</span>",
                x=0.5, y=0.5, font=dict(size=20, color=TEXT_HI),
                showarrow=False, xref="paper", yref="paper",
            )
            dark_layout(fig, "Employment Type", height=350)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_right2:
        remote_dist = get_remote_distribution(df)
        label_map   = {True: "Remote", False: "On-site", "True": "Remote", "False": "On-site"}
        labels      = [label_map.get(l, str(l)) for l in remote_dist.index]

        fig = go.Figure(go.Bar(
            x=labels,
            y=remote_dist.values,
            marker=dict(
                color=["#56c596", "#ff6b6b"],
                opacity=0.9,
                line=dict(width=0),
            ),
            text=[
                f"{v:,}<br>({v/remote_dist.sum()*100:.1f}%)"
                for v in remote_dist.values
            ],
            textposition="outside",
            textfont=dict(size=12, color=TEXT_HI),
            hovertemplate="<b>%{x}</b><br>%{y:,} jobs<extra></extra>",
        ))
        dark_layout(fig, "Remote vs On-site", height=350)
        fig.update_layout(
            yaxis=dict(range=[0, remote_dist.max() * 1.25]),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# TAB 2 — SKILLS & ROLES
# ──────────────────────────────────────────────────────────────────

with tab2:

    if not nlp_available:
        st.info(
            "NLP data not found. Run the NLP pipeline first:\n\n"
            "```bash\npython -m data_pipeline.skill_extractor\n"
            "python -m data_pipeline.nlp_pipeline\n```"
        )
    else:
        col_left, col_right = st.columns([3, 2])

        # ── Top Skills ───────────────────────────────────────────
        with col_left:
            top_skills = get_top_skills(df, top_n=20)

            if not top_skills.empty:
                skills = top_skills.index.tolist()[::-1]
                counts = top_skills.values.tolist()[::-1]
                colors = (PALETTE * 3)[:len(skills)][::-1]

                fig = go.Figure(go.Bar(
                    x=counts,
                    y=skills,
                    orientation="h",
                    marker=dict(color=colors, opacity=0.9, line=dict(width=0)),
                    text=[f"{v:,}" for v in counts],
                    textposition="outside",
                    textfont=dict(size=10, color=TEXT_HI),
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
                    hole=0.50,
                    marker=dict(
                        colors=PALETTE[:len(role_dist)],
                        line=dict(color=BG_BASE, width=2),
                    ),
                    pull=[0.05 if v == role_dist.max() else 0 for v in role_dist.values],
                    textinfo="label+percent",
                    textfont=dict(size=10),
                    hovertemplate="<b>%{label}</b><br>%{value:,} jobs<br>%{percent}<extra></extra>",
                ))
                fig.add_annotation(
                    text=f"<b>{role_dist.sum():,}</b><br><span style='font-size:11px;color:{TEXT_MID}'>Mapped</span>",
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
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-family:Space Mono,monospace; font-size:0.8rem;"
            "color:#9898b0; letter-spacing:0.08em;'>TOP SKILLS BY ROLE</p>",
            unsafe_allow_html=True
        )

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
                        background: var(--bg-card);
                        border: 1px solid {color}33;
                        border-top: 2px solid {color};
                        border-radius: 8px;
                        padding: 0.9rem 1rem;
                        margin-bottom: 0.75rem;
                    ">
                        <div style="font-family:'Space Mono',monospace;
                                    font-size:0.72rem; color:{color};
                                    letter-spacing:0.08em; margin-bottom:0.5rem;">
                            {role.upper()}
                        </div>
                        {''.join(
                            f'<div style="display:flex; justify-content:space-between;'
                            f'padding:3px 0; border-bottom:1px solid #2a2a4022;">'
                            f'<span style="font-size:0.82rem; color:#eeeef5;">{row["skill"]}</span>'
                            f'<span style="font-size:0.78rem; color:{color}; font-family:Space Mono,monospace;">{row["count"]}</span>'
                            f'</div>'
                            for _, row in role_skills.iterrows()
                        )}
                    </div>
                    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# TAB 3 — SALARY ANALYSIS
# ──────────────────────────────────────────────────────────────────

with tab3:

    salary_df = df[df["salary_available"].fillna(False) == True].copy()

    if salary_df.empty:
        st.info("No salary data available in the current filtered dataset.")
    else:
        salary_df["avg_salary"] = (
            salary_df["job_min_salary"] + salary_df["job_max_salary"]
        ) / 2

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jobs with Salary", f"{len(salary_df):,}")
        with col2:
            st.metric("Average Salary", f"${salary_df['avg_salary'].mean():,.0f}")
        with col3:
            st.metric("Median Salary", f"${salary_df['avg_salary'].median():,.0f}")

        st.markdown("<hr>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        # ── Salary Distribution Box ──────────────────────────────
        with col_left:
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=salary_df["avg_salary"],
                name="Salary Range",
                marker_color=ACCENT,
                boxmean="sd",
                fillcolor="#7c6aff22",
                line=dict(color=ACCENT, width=2),
                jitter=0.4,
                pointpos=-1.5,
                boxpoints="all",
                marker=dict(size=4, opacity=0.4, color="#b388ff"),
            ))
            dark_layout(fig, "Salary Distribution", "Avg Salary (USD)", height=400)
            fig.update_layout(
                yaxis=dict(tickprefix="$", tickformat=","),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Salary by Employment Type ────────────────────────────
        with col_right:
            if "job_employment_type" in salary_df.columns:
                sal_by_emp = (
                    salary_df.groupby("job_employment_type")["avg_salary"]
                    .mean()
                    .sort_values(ascending=False)
                )
                fig = go.Figure(go.Bar(
                    x=sal_by_emp.index,
                    y=sal_by_emp.values,
                    marker=dict(
                        color=PALETTE[:len(sal_by_emp)],
                        opacity=0.9, line=dict(width=0),
                    ),
                    text=[f"${v:,.0f}" for v in sal_by_emp.values],
                    textposition="outside",
                    textfont=dict(size=11, color=TEXT_HI),
                    hovertemplate="<b>%{x}</b><br>Avg: $%{y:,.0f}<extra></extra>",
                ))
                dark_layout(fig, "Avg Salary by Employment Type", height=400)
                fig.update_layout(
                    yaxis=dict(
                        tickprefix="$", tickformat=",",
                        range=[0, sal_by_emp.max() * 1.2],
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Salary by Role ───────────────────────────────────────
        if nlp_available and "role_category" in salary_df.columns:
            sal_by_role = (
                salary_df.groupby("role_category")["avg_salary"]
                .agg(["mean", "count"])
                .sort_values("mean", ascending=True)
                .reset_index()
            )
            sal_by_role.columns = ["Role", "Avg Salary", "Count"]

            fig = go.Figure(go.Bar(
                x=sal_by_role["Avg Salary"],
                y=sal_by_role["Role"],
                orientation="h",
                marker=dict(
                    color=PALETTE[:len(sal_by_role)],
                    opacity=0.9, line=dict(width=0),
                ),
                text=[f"${v:,.0f}" for v in sal_by_role["Avg Salary"]],
                textposition="outside",
                textfont=dict(size=11, color=TEXT_HI),
                hovertemplate=(
                    "<b>%{y}</b><br>Avg Salary: $%{x:,.0f}<extra></extra>"
                ),
            ))
            dark_layout(fig, "Average Salary by Role",
                        "Average Salary (USD)", height=350)
            fig.update_layout(
                xaxis=dict(
                    tickprefix="$", tickformat=",",
                    range=[0, sal_by_role["Avg Salary"].max() * 1.2],
                ),
                margin=dict(l=150, r=80, t=55, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# TAB 4 — RAW DATA
# ──────────────────────────────────────────────────────────────────

with tab4:

    display_cols = [
        c for c in [
            "job_title", "employer_name", "job_city", "job_country",
            "job_employment_type", "job_is_remote", "salary_available",
            "avg_salary" if "avg_salary" in df.columns else None,
            "role_category", "extracted_skills", "job_posted_at",
        ] if c and c in df.columns
    ]

    # Add avg_salary column for display if salary data exists
    display_df = df[display_cols].copy()
    if "salary_available" in df.columns:
        mask = df["salary_available"].fillna(False) == True
        if mask.any() and "avg_salary" not in display_df.columns:
            display_df.loc[mask, "avg_salary"] = (
                df.loc[mask, "job_min_salary"] + df.loc[mask, "job_max_salary"]
            ) / 2

    st.markdown(
        f"<p style='color:{TEXT_MID}; font-size:0.82rem; font-family:Space Mono,monospace;'>"
        f"Showing {len(display_df):,} records after filters</p>",
        unsafe_allow_html=True
    )

    # Search
    search = st.text_input(
        "Search job titles",
        placeholder="e.g. Data Scientist, ML Engineer...",
        label_visibility="collapsed",
    )
    if search:
        display_df = display_df[
            display_df["job_title"].str.contains(search, case=False, na=False)
        ]

    st.dataframe(
        display_df,
        use_container_width=True,
        height=500,
        hide_index=True,
    )