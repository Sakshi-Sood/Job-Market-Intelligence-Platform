"""
data_pipeline/skill_extractor.py
==================================
Extracts tech skills from job descriptions using keyword matching + regex.

Designed to run after clean_data.py. Adds two new columns to your DataFrame:
  - extracted_skills  : comma-separated string of matched skills
  - skill_count       : how many distinct skills were found

Run standalone or import extract_skills() into your pipeline.
"""

import re
import logging
import pandas as pd
from sqlalchemy import text
from backend.database import engine

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

# ==============================================================
#  MASTER SKILL LIST
#  Add/remove skills here — lowercase, order doesn't matter.
#  Grouped by category for readability only.
# ==============================================================

DEFAULT_SKILLS = {
    # --- Languages ---
    "python", "sql", "r", "java", "scala", "julia", "c++", "go", "rust",

    # --- ML / AI ---
    "machine learning", "deep learning", "nlp", "computer vision",
    "reinforcement learning", "llm", "generative ai", "transformer",
    "neural network", "xgboost", "lightgbm", "catboost", "random forest",
    "scikit-learn", "sklearn",

    # --- Frameworks ---
    "tensorflow", "pytorch", "keras", "hugging face", "langchain",
    "fastapi", "flask", "django", "streamlit",

    # --- Data Engineering ---
    "spark", "hadoop", "kafka", "airflow", "dbt", "flink",
    "pandas", "numpy", "polars",

    # --- Databases ---
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "bigquery", "snowflake", "redshift", "databricks",

    # --- Cloud ---
    "aws", "azure", "gcp", "google cloud",
    "ec2", "s3", "lambda", "sagemaker",

    # --- DevOps ---
    "docker", "kubernetes", "jenkins", "github actions", "ci/cd",
    "terraform", "ansible", "helm", "prometheus", "grafana",

    # --- Visualisation ---
    "tableau", "power bi", "plotly", "matplotlib", "seaborn", "looker",

    # --- Version Control ---
    "git", "github", "gitlab",

    # --- Other ---
    "excel", "linux", "bash", "rest api", "graphql",
}

# Pre-compile regex patterns once at import time for the default set.
_DEFAULT_PATTERNS: dict[str, re.Pattern] = {
    skill: re.compile(rf"\b{re.escape(skill)}\b", re.IGNORECASE)
    for skill in DEFAULT_SKILLS
}


def build_patterns(
    custom_skills: set[str] | list[str] | None = None,
) -> dict[str, re.Pattern]:
    """
    Return compiled regex patterns for the full skill set.

    If custom_skills is provided, merge them with DEFAULT_SKILLS
    and compile patterns for the new entries only.
    """
    if not custom_skills:
        return _DEFAULT_PATTERNS

    merged = DEFAULT_SKILLS | {s.strip().lower() for s in custom_skills if s.strip()}
    new_skills = merged - DEFAULT_SKILLS

    if not new_skills:
        return _DEFAULT_PATTERNS

    # Start from default patterns, add new ones
    patterns = dict(_DEFAULT_PATTERNS)
    for skill in new_skills:
        patterns[skill] = re.compile(rf"\b{re.escape(skill)}\b", re.IGNORECASE)

    return patterns


# ==============================================================
#  CORE EXTRACTION FUNCTION
# ==============================================================

def extract_skills_from_text(
    text: str,
    custom_skills: set[str] | list[str] | None = None,
) -> list[str]:
    """
    Return a sorted list of skills found in `text`.

    Parameters
    ----------
    text          : str   Raw or cleaned job description string.
    custom_skills : set | list | None
        Extra skills to look for on top of the built-in list.

    Returns
    -------
    list[str]
        Alphabetically sorted list of matched skill names.
        Empty list if nothing matches or text is blank.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    patterns = build_patterns(custom_skills)

    found = [
        skill
        for skill, pattern in patterns.items()
        if pattern.search(text)
    ]

    return sorted(found)


def extract_skills(
    df: pd.DataFrame,
    custom_skills: set[str] | list[str] | None = None,
) -> pd.DataFrame:
    """
    Add `extracted_skills` and `skill_count` columns to a jobs DataFrame.

    Parameters
    ----------
    df            : pd.DataFrame
        Must contain a `job_description` column (lowercase text preferred).
    custom_skills : set | list | None
        Extra skills to look for on top of the built-in list.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with two new columns appended.
    """
    if "job_description" not in df.columns:
        raise ValueError("DataFrame must contain a 'job_description' column.")

    logger.info("Extracting skills from %d job descriptions...", len(df))
    if custom_skills:
        logger.info("Including %d custom skills: %s", len(custom_skills), custom_skills)

    skill_lists = df["job_description"].apply(
        lambda desc: extract_skills_from_text(desc, custom_skills)
    )

    df = df.copy()
    df["extracted_skills"] = skill_lists.apply(lambda s: ", ".join(s))
    df["skill_count"]       = skill_lists.apply(len)

    total_with_skills = (df["skill_count"] > 0).sum()
    logger.info(
        "Done. %d / %d jobs had at least one recognised skill.",
        total_with_skills, len(df),
    )

    return df


# ==============================================================
#  ANALYTICS HELPERS
# ==============================================================

def get_top_skills(df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    """
    Count how often each skill appears across all jobs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `extracted_skills` (comma-separated string).
    top_n : int
        Return the N most common skills.

    Returns
    -------
    pd.Series
        Skill → count, sorted descending.
    """
    all_skills: list[str] = []

    for cell in df["extracted_skills"].dropna():
        if cell.strip():
            all_skills.extend(s.strip() for s in cell.split(","))

    return (
        pd.Series(all_skills)
        .value_counts()
        .head(top_n)
    )


def get_skill_cooccurrence(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Build a simple co-occurrence count: for each pair of skills,
    how many jobs mention both?

    Useful for understanding which skills are typically bundled together.

    Returns
    -------
    pd.DataFrame
        Columns: skill_a, skill_b, count — sorted by count desc.
    """
    from itertools import combinations
    from collections import Counter

    pair_counts: Counter = Counter()

    for cell in df["extracted_skills"].dropna():
        skills = [s.strip() for s in cell.split(",") if s.strip()]
        for a, b in combinations(sorted(skills), 2):
            pair_counts[(a, b)] += 1

    top_pairs = pair_counts.most_common(top_n)

    return pd.DataFrame(
        [(a, b, c) for (a, b), c in top_pairs],
        columns=["skill_a", "skill_b", "count"],
    )


# ==============================================================
#  STANDALONE RUNNER
#  python -m data_pipeline.skill_extractor
# ==============================================================

def run_skill_extraction(
    custom_skills: set[str] | list[str] | None = None,
) -> None:
    """
    Load jobs from PostgreSQL → extract skills → write results back.

    Adds columns `extracted_skills` and `skill_count` to the jobs table
    using an UPDATE statement per row (safe for incremental runs).

    Parameters
    ----------
    custom_skills : set | list | None
        Extra skills to look for on top of the built-in list.
    """
    logger.info("Loading jobs from PostgreSQL...")

    df = pd.read_sql(text("SELECT job_id, job_description FROM jobs"), engine)
    logger.info("Loaded %d rows.", len(df))

    df = extract_skills(df, custom_skills)

    logger.info("Writing skill results back to PostgreSQL...")

    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text(
                    """
                    UPDATE jobs
                    SET    extracted_skills = :skills,
                           skill_count      = :count
                    WHERE  job_id = :job_id
                    """
                ),
                {
                    "skills": row["extracted_skills"],
                    "count":  int(row["skill_count"]),
                    "job_id": row["job_id"],
                },
            )

    logger.info("Skill extraction complete.")

    # Print a quick summary
    top = get_top_skills(df, top_n=10)
    print("\nTop 10 skills in current dataset:")
    print(top.to_string())


if __name__ == "__main__":
    run_skill_extraction()