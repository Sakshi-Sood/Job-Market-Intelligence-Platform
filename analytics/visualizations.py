"""
analytics/visualizations.py
============================
Production-ready visualization module for the Job Market Intelligence Platform.

Generates premium dark-themed charts using Matplotlib and Plotly.
All outputs are saved to ``outputs/visualizations/`` as PNG (Matplotlib)
or interactive HTML (Plotly) files.

Designed for reuse in Streamlit dashboards, FastAPI endpoints, Docker
containers, and CI/CD pipelines.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
#  Logging configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ==============================================
#  OUTPUT DIRECTORY
# ==============================================

_OUTPUT_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "visualizations",
)
os.makedirs(_OUTPUT_DIR, exist_ok=True)


# ==============================================
#  SHARED STYLE CONSTANTS
# ==============================================

# Dark background palette
_BG_DARK: str = "#0f0f1a"
_BG_CARD: str = "#1a1a2e"
_TEXT_PRIMARY: str = "#e0e0e0"
_TEXT_SECONDARY: str = "#9e9e9e"
_GRID_COLOR: str = "#2a2a40"
_ACCENT: str = "#6c63ff"

# Gradient colour ramp (10 colours – warm → cool)
_GRADIENT_PALETTE: List[str] = [
    "#ff6b6b", "#ff8e72", "#ffb347", "#ffd700",
    "#a3de83", "#56c596", "#3dc1d3", "#6c63ff",
    "#b388ff", "#ea80fc",
]

# Plotly template base
_PLOTLY_TEMPLATE: str = "plotly_dark"

# Common filler words to exclude from keyword analysis
_STOP_WORDS: set = {
    "a", "an", "the", "and", "or", "of", "to", "in", "for",
    "with", "at", "by", "on", "is", "it", "as", "from",
    "that", "this", "are", "was", "be", "has", "have",
    "will", "can", "not", "but", "we", "our", "you", "your",
    "-", "/", "&", "|", "i", "ii", "iii", "iv", "v",
}


# ==============================================
#  HELPER UTILITIES
# ==============================================

def _apply_dark_theme(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply a consistent dark theme to a matplotlib axes/figure."""
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_CARD)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_GRID_COLOR)
    ax.spines["bottom"].set_color(_GRID_COLOR)

    ax.tick_params(colors=_TEXT_SECONDARY, labelsize=10)
    ax.xaxis.label.set_color(_TEXT_PRIMARY)
    ax.yaxis.label.set_color(_TEXT_PRIMARY)
    ax.title.set_color(_TEXT_PRIMARY)

    ax.yaxis.grid(
        True, linestyle="--",
        linewidth=0.4, color=_GRID_COLOR, alpha=0.6,
    )
    ax.set_axisbelow(True)


def _plotly_dark_layout(
    fig: go.Figure,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
) -> go.Figure:
    """Apply a premium dark layout to a Plotly figure."""
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(
                size=22,
                color="#ffffff",
                family="Segoe UI, Roboto, sans-serif",
            ),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor=_BG_DARK,
        plot_bgcolor=_BG_CARD,
        font=dict(
            family="Segoe UI, Roboto, sans-serif",
            color=_TEXT_PRIMARY,
        ),
        xaxis=dict(
            title=xaxis_title,
            showgrid=False,
            linecolor=_GRID_COLOR,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor=_GRID_COLOR,
            gridwidth=0.4,
            linecolor=_GRID_COLOR,
            tickfont=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=80, b=60),
        hoverlabel=dict(
            bgcolor="#1e1e30",
            font_size=13,
            font_color="#ffffff",
            bordercolor="#6c63ff",
        ),
    )
    return fig


def _save_matplotlib(fig: plt.Figure, filename: str) -> str:
    """Save a matplotlib figure to the outputs directory.

    Returns:
        Absolute path to the saved PNG file.
    """
    path = os.path.join(_OUTPUT_DIR, filename)
    try:
        fig.savefig(
            path, dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        logger.info("Saved matplotlib chart -> %s", path)
    except Exception:
        logger.error("Failed to save matplotlib chart: %s", path, exc_info=True)
    return path


def _save_plotly(fig: go.Figure, filename: str) -> str:
    """Save a Plotly figure to the outputs directory as HTML.

    ``auto_open`` is **disabled** to support headless / Docker / CI
    environments.

    Returns:
        Absolute path to the saved HTML file.
    """
    path = os.path.join(_OUTPUT_DIR, filename)
    try:
        fig.write_html(path, auto_open=False)
        logger.info("Saved plotly chart   -> %s", path)
    except Exception:
        logger.error("Failed to save plotly chart: %s", path, exc_info=True)
    return path


# ==============================================
#  1. TOP HIRING CITIES  (Matplotlib – Horizontal)
# ==============================================

def plot_top_cities(top_cities: pd.Series) -> str:
    """Horizontal bar chart with gradient colouring & value labels.

    Args:
        top_cities: A ``pd.Series`` mapping city names to job counts.

    Returns:
        File path to the saved PNG chart.
    """
    cities = top_cities.index.tolist()[::-1]
    counts = top_cities.values.tolist()[::-1]
    n = len(cities)

    colors = _GRADIENT_PALETTE[:n][::-1]

    fig, ax = plt.subplots(figsize=(11, 6))
    _apply_dark_theme(ax, fig)

    bars = ax.barh(
        cities, counts,
        color=colors,
        edgecolor="none",
        height=0.65,
        zorder=3,
    )

    # Glow shadow behind bars
    for bar, color in zip(bars, colors):
        ax.barh(
            bar.get_y() + bar.get_height() / 2,
            bar.get_width(),
            height=bar.get_height() * 1.15,
            color=color, alpha=0.12,
            zorder=2,
        )

    # Value labels
    max_val = max(counts) if counts else 1
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_width() + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center", ha="left",
            fontsize=10, fontweight="bold",
            color=_TEXT_PRIMARY,
            path_effects=[
                pe.withStroke(linewidth=2, foreground=_BG_CARD)
            ],
        )

    ax.set_title(
        "Top Hiring Cities",
        fontsize=18, fontweight="bold", pad=18,
    )
    ax.set_xlabel("Number of Jobs", fontsize=12, labelpad=10)
    ax.set_xlim(0, max_val * 1.15)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4,
                  color=_GRID_COLOR, alpha=0.6)
    ax.yaxis.grid(False)

    plt.tight_layout()
    return _save_matplotlib(fig, "top_hiring_cities.png")


# ==============================================
#  2. TOP HIRING COMPANIES  (Plotly – Animated Bar)
# ==============================================

def plot_top_companies(top_companies: pd.Series) -> str:
    """Interactive vertical bar chart with hover details & gradient fill.

    Args:
        top_companies: A ``pd.Series`` mapping company names to job counts.

    Returns:
        File path to the saved HTML chart.
    """
    companies = top_companies.index.tolist()
    counts = top_companies.values.tolist()
    n = len(companies)

    colors = _GRADIENT_PALETTE[:n]

    fig = go.Figure(
        go.Bar(
            x=companies,
            y=counts,
            marker=dict(
                color=colors,
                line=dict(width=0),
                opacity=0.92,
            ),
            text=[f"{v:,}" for v in counts],
            textposition="outside",
            textfont=dict(size=12, color=_TEXT_PRIMARY),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Open Positions: %{y:,}<extra></extra>"
            ),
        )
    )

    _plotly_dark_layout(
        fig,
        title="Top Hiring Companies",
        xaxis_title="Company",
        yaxis_title="Number of Jobs",
    )

    fig.update_layout(
        xaxis_tickangle=-30,
        yaxis=dict(
            range=[0, max(counts) * 1.20] if counts else [0, 1],
            dtick=1,
        ),
    )

    return _save_plotly(fig, "top_hiring_companies.html")


# ==============================================
#  3. EMPLOYMENT TYPE DISTRIBUTION  (Plotly – Donut)
# ==============================================

def plot_employment_distribution(employment_distribution: pd.Series) -> str:
    """Donut chart with pull effect & custom hover.

    Args:
        employment_distribution: A ``pd.Series`` mapping employment types
            (e.g. Full-time, Contract) to counts.

    Returns:
        File path to the saved HTML chart.
    """
    labels = employment_distribution.index.tolist()
    values = employment_distribution.values.tolist()
    n = len(labels)

    donut_colors = [
        "#6c63ff", "#ff6b6b", "#56c596",
        "#ffd700", "#3dc1d3", "#ea80fc",
        "#ff8e72", "#b388ff", "#a3de83", "#ffb347",
    ][:n]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.52,
            marker=dict(
                colors=donut_colors,
                line=dict(color=_BG_DARK, width=2.5),
            ),
            pull=[0.04] * n,
            textinfo="label+percent",
            textfont=dict(size=12, color="#ffffff"),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Jobs: %{value:,}<br>"
                "Share: %{percent}<extra></extra>"
            ),
        )
    )

    # Centre annotation
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:12px;"
             f"color:{_TEXT_SECONDARY}'>Total Jobs</span>",
        x=0.5, y=0.5,
        font=dict(size=22, color="#ffffff"),
        showarrow=False,
        xref="paper", yref="paper",
    )

    _plotly_dark_layout(
        fig,
        title="Employment Type Distribution",
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.15,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
    )

    return _save_plotly(fig, "employment_type_distribution.html")


# ==============================================
#  4. REMOTE JOB DISTRIBUTION  (Matplotlib – Styled)
# ==============================================

def plot_remote_distribution(remote_distribution: pd.Series) -> str:
    """Dual-colour bar chart for remote vs on-site breakdown.

    Args:
        remote_distribution: A ``pd.Series`` indexed by True/False mapping
            to job counts.

    Returns:
        File path to the saved PNG chart.
    """
    labels_raw = remote_distribution.index.tolist()
    values = remote_distribution.values.tolist()

    # Pretty labels
    label_map = {
        True: "Remote",
        False: "On-site",
        "True": "Remote",
        "False": "On-site",
    }
    labels = [label_map.get(l, str(l)) for l in labels_raw]

    bar_colors = ["#56c596", "#ff6b6b"][:len(labels)]

    fig, ax = plt.subplots(figsize=(7, 5))
    _apply_dark_theme(ax, fig)

    bars = ax.bar(
        labels, values,
        color=bar_colors,
        edgecolor="none",
        width=0.5,
        zorder=3,
    )

    # Glow
    for bar, c in zip(bars, bar_colors):
        ax.bar(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            width=bar.get_width() * 1.15,
            color=c, alpha=0.10,
            zorder=2,
        )

    total = sum(values)
    for bar, val in zip(bars, values):
        pct = val / total * 100 if total else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            color=_TEXT_PRIMARY,
            path_effects=[
                pe.withStroke(linewidth=2, foreground=_BG_CARD)
            ],
        )

    ax.set_title(
        "Remote vs On-site Jobs",
        fontsize=18, fontweight="bold", pad=18,
    )
    ax.set_ylabel("Number of Jobs", fontsize=12, labelpad=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(0, max(values) * 1.25 if values else 1)

    plt.tight_layout()
    return _save_matplotlib(fig, "remote_vs_onsite.png")


# ==============================================
#  5. SALARY DISTRIBUTION  (Plotly – Box + Strip)
# ==============================================

def plot_salary_distribution(df: pd.DataFrame) -> Optional[str]:
    """Combined box-and-strip plot showing salary spread.

    Args:
        df: Full jobs DataFrame; must contain ``salary_available``,
            ``job_min_salary``, and ``job_max_salary`` columns.

    Returns:
        File path to the saved HTML chart, or ``None`` if salary data
        is unavailable.
    """
    salary_df = df[
        df["salary_available"].fillna(False) == True
    ].copy()

    if salary_df.empty:
        logger.warning("No salary data available -- skipping salary chart.")
        return None

    salary_df["avg_salary"] = (
        salary_df["job_min_salary"] +
        salary_df["job_max_salary"]
    ) / 2

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=salary_df["avg_salary"],
            name="Salary Range",
            marker_color=_ACCENT,
            boxmean="sd",
            fillcolor="rgba(108, 99, 255, 0.25)",
            line=dict(color=_ACCENT, width=2),
            jitter=0.4,
            pointpos=-1.5,
            boxpoints="all",
            marker=dict(
                size=4,
                opacity=0.5,
                color="#b388ff",
            ),
        )
    )

    _plotly_dark_layout(
        fig,
        title="Salary Distribution",
        yaxis_title="Average Salary (USD)",
    )

    fig.update_layout(
        yaxis=dict(
            tickprefix="$",
            tickformat=",",
        ),
        showlegend=False,
        height=500,
    )

    return _save_plotly(fig, "salary_distribution.html")


# ==============================================
#  6. DATA QUALITY HEATMAP  (Matplotlib)
# ==============================================

def plot_data_quality(missing_pct: pd.Series) -> Optional[str]:
    """Horizontal bar chart showing columns with missing data %.

    Args:
        missing_pct: A ``pd.Series`` mapping column names to their
            percentage of missing values (only columns with > 0 %).

    Returns:
        File path to the saved PNG chart, or ``None`` if there is no
        missing data.
    """
    if missing_pct.empty:
        logger.info("No missing data detected -- skipping data quality chart.")
        return None

    cols = missing_pct.index.tolist()[::-1]
    pcts = missing_pct.values.tolist()[::-1]

    # Colour mapping: low → green, high → red
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=0, vmax=100)
    colors = [cmap(norm(p)) for p in pcts]

    fig, ax = plt.subplots(figsize=(10, max(4, len(cols) * 0.45)))
    _apply_dark_theme(ax, fig)

    bars = ax.barh(
        cols, pcts,
        color=colors,
        edgecolor="none",
        height=0.6,
        zorder=3,
    )

    for bar, val in zip(bars, pcts):
        ax.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", ha="left",
            fontsize=10, fontweight="bold",
            color=_TEXT_PRIMARY,
            path_effects=[
                pe.withStroke(linewidth=2, foreground=_BG_CARD)
            ],
        )

    ax.set_title(
        "Data Quality - Missing Values by Column",
        fontsize=16, fontweight="bold", pad=18,
    )
    ax.set_xlabel("Missing (%)", fontsize=12, labelpad=10)
    ax.set_xlim(0, min(max(pcts) * 1.25, 105) if pcts else 100)

    ax.xaxis.grid(True, linestyle="--", linewidth=0.4,
                  color=_GRID_COLOR, alpha=0.6)
    ax.yaxis.grid(False)

    plt.tight_layout()
    return _save_matplotlib(fig, "data_quality.png")


# ==============================================
#  7. JOB TITLE KEYWORD ANALYSIS  (Matplotlib)
# ==============================================

def plot_title_keywords(
    df: pd.DataFrame,
    top_n: int = 15,
) -> Optional[str]:
    """Horizontal bar chart of the most frequent keywords in job titles.

    Tokenises every ``job_title`` value, strips punctuation and stop words,
    then plots the *top_n* most common keywords.

    Args:
        df: Jobs DataFrame containing a ``job_title`` column.
        top_n: Number of keywords to display.

    Returns:
        File path to the saved PNG chart, or ``None`` if no usable
        keyword data exists.
    """
    if "job_title" not in df.columns or df["job_title"].dropna().empty:
        logger.warning("No job title data available -- skipping keyword chart.")
        return None

    # Tokenise and count
    word_counts: Counter = Counter()
    for title in df["job_title"].dropna():
        tokens = str(title).lower().split()
        for tok in tokens:
            tok = tok.strip("(),.-:;!?\"'")
            if len(tok) > 1 and tok not in _STOP_WORDS:
                word_counts[tok] += 1

    if not word_counts:
        logger.warning("No keywords extracted from job titles.")
        return None

    most_common = word_counts.most_common(top_n)
    keywords = [w for w, _ in most_common][::-1]
    counts = [c for _, c in most_common][::-1]
    n = len(keywords)

    # Build a purple-to-cyan gradient
    cmap = plt.cm.cool
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.42)))
    _apply_dark_theme(ax, fig)

    bars = ax.barh(
        keywords, counts,
        color=colors,
        edgecolor="none",
        height=0.65,
        zorder=3,
    )

    # Glow effect
    for bar, color in zip(bars, colors):
        ax.barh(
            bar.get_y() + bar.get_height() / 2,
            bar.get_width(),
            height=bar.get_height() * 1.15,
            color=color, alpha=0.12,
            zorder=2,
        )

    # Value labels
    max_val = max(counts) if counts else 1
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_width() + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val}",
            va="center", ha="left",
            fontsize=10, fontweight="bold",
            color=_TEXT_PRIMARY,
            path_effects=[
                pe.withStroke(linewidth=2, foreground=_BG_CARD)
            ],
        )

    ax.set_title(
        "Top Keywords in Job Titles",
        fontsize=18, fontweight="bold", pad=18,
    )
    ax.set_xlabel("Frequency", fontsize=12, labelpad=10)
    ax.set_xlim(0, max_val * 1.18)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4,
                  color=_GRID_COLOR, alpha=0.6)
    ax.yaxis.grid(False)

    plt.tight_layout()
    return _save_matplotlib(fig, "title_keywords.png")

# ==============================================
#  8. TOP SKILLS DEMAND  (Plotly – Horizontal Bar)
# ==============================================
 
def plot_top_skills(top_skills: pd.Series) -> Optional[str]:
    """
    Horizontal bar chart showing the most in-demand skills across all jobs.
 
    Parameters
    ----------
    top_skills : pd.Series  skill -> count, from skill_extractor.get_top_skills()
 
    Returns
    -------
    File path to saved HTML, or None if no skill data.
    """
    if top_skills is None or top_skills.empty:
        logger.warning("No skill data available — skipping skills chart.")
        return None
 
    skills = top_skills.index.tolist()[::-1]   # reverse for bottom-to-top display
    counts = top_skills.values.tolist()[::-1]
    n      = len(skills)
 
    colors = (_GRADIENT_PALETTE * 3)[:n][::-1]  # cycle palette if > 10 skills
 
    fig = go.Figure(
        go.Bar(
            x=counts,
            y=skills,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
                opacity=0.92,
            ),
            text=[f"{v:,}" for v in counts],
            textposition="outside",
            textfont=dict(size=11, color=_TEXT_PRIMARY),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Job listings: %{x:,}<extra></extra>"
            ),
        )
    )
 
    _plotly_dark_layout(
        fig,
        title="Most In-Demand Skills",
        xaxis_title="Number of Jobs Mentioning Skill",
        yaxis_title="",
    )
 
    fig.update_layout(
        height=max(400, n * 38),
        xaxis=dict(range=[0, max(counts) * 1.18] if counts else [0, 1]),
        yaxis=dict(tickfont=dict(size=12)),
        margin=dict(l=160, r=60, t=80, b=60),
    )
 
    return _save_plotly(fig, "top_skills_demand.html")
 
 
# ==============================================
#  9. ROLE DISTRIBUTION  (Plotly – Donut)
# ==============================================
 
def plot_role_distribution(role_distribution: pd.Series) -> Optional[str]:
    """
    Donut chart showing the breakdown of job roles across the dataset.
 
    Parameters
    ----------
    role_distribution : pd.Series  role -> count,
                        from nlp_pipeline.get_role_distribution()
 
    Returns
    -------
    File path to saved HTML, or None if no data.
    """
    if role_distribution is None or role_distribution.empty:
        logger.warning("No role data available — skipping role chart.")
        return None
 
    labels = role_distribution.index.tolist()
    values = role_distribution.values.tolist()
    n      = len(labels)
 
    role_colors = [
        "#6c63ff", "#56c596", "#ff6b6b",
        "#ffd700", "#3dc1d3", "#ff8e72",
        "#b388ff", "#ea80fc", "#a3de83", "#ffb347",
    ][:n]
 
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.52,
            marker=dict(
                colors=role_colors,
                line=dict(color=_BG_DARK, width=2.5),
            ),
            pull=[0.04 if v == max(values) else 0 for v in values],  # pull largest
            textinfo="label+percent",
            textfont=dict(size=11, color="#ffffff"),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Jobs: %{value:,}<br>"
                "Share: %{percent}<extra></extra>"
            ),
        )
    )
 
    total = sum(values)
    fig.add_annotation(
        text=(
            f"<b>{total:,}</b><br>"
            f"<span style='font-size:12px;color:{_TEXT_SECONDARY}'>Roles Mapped</span>"
        ),
        x=0.5, y=0.5,
        font=dict(size=22, color="#ffffff"),
        showarrow=False,
        xref="paper", yref="paper",
    )
 
    _plotly_dark_layout(fig, title="Job Role Distribution")
 
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.20,
            xanchor="center", x=0.5,
            font=dict(size=11),
        ),
    )
 
    return _save_plotly(fig, "role_distribution.html")

# ==============================================
#  ORCHESTRATOR
# ==============================================

def generate_all_visualizations(
    df: pd.DataFrame,
    top_cities: pd.Series,
    top_companies: pd.Series,
    employment_distribution: pd.Series,
    remote_distribution: pd.Series,
    missing_pct: pd.Series,
    top_skills: pd.Series = None,           
    role_distribution: pd.Series = None,
) -> Dict[str, Optional[str]]:
    """Generate every chart sequentially and return their file paths.

    This single entry-point is designed for integration with Streamlit,
    FastAPI, or any other caller that needs a one-call interface.

    Args:
        df: The cleaned jobs DataFrame (used for salary & keyword charts).
        top_cities: Series from ``get_top_cities()``.
        top_companies: Series from ``get_top_companies()``.
        employment_distribution: Series from ``get_employment_distribution()``.
        remote_distribution: Series from ``get_remote_distribution()``.
        missing_pct: Series from ``get_data_quality_metrics()``.

    Returns:
        Dictionary mapping chart names to their saved file paths.
        A value of ``None`` means that chart was skipped (e.g. no salary
        data).
    """
    logger.info("Starting visualization generation...")

    paths: Dict[str, Optional[str]] = {
        "top_cities": plot_top_cities(top_cities),
        "top_companies": plot_top_companies(top_companies),
        "employment_distribution": plot_employment_distribution(
            employment_distribution
        ),
        "remote_distribution": plot_remote_distribution(remote_distribution),
        "salary_distribution": plot_salary_distribution(df),
        "data_quality": plot_data_quality(missing_pct),
        "title_keywords": plot_title_keywords(df),
        "top_skills": plot_top_skills(top_skills),             
        "role_distribution": plot_role_distribution(role_distribution),
    }

    generated = sum(1 for v in paths.values() if v is not None)
    logger.info(
        "Visualization generation complete: %d/%d charts saved to %s",
        generated, len(paths), _OUTPUT_DIR,
    )

    return paths