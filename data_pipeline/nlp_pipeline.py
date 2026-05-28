"""
data_pipeline/nlp_pipeline.py
================================
NLP enrichment pipeline — runs AFTER skill_extractor.py.

What it does:
  1. Loads job descriptions from PostgreSQL
  2. Cleans and tokenises text (lowercasing, stopword removal, lemmatisation)
  3. Runs TF-IDF vectorisation to find the most important terms per job
  4. Categorises each job into a role (Data Scientist, ML Engineer, etc.)
  5. Writes role_category and top_keywords columns back to PostgreSQL

Run:
    python -m data_pipeline.nlp_pipeline
"""

import logging
import re
import pandas as pd
import numpy as np

from sqlalchemy import text
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.database import engine

# ── logging ──────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ================================================================
#  ROLE CATEGORISATION RULES
#  Each role maps to a list of trigger keywords.
#  Rules are checked in ORDER — first match wins.
#  Keep the most specific roles at the top.
# ================================================================

ROLE_RULES: list[tuple[str, list[str]]] = [
    ("ML Engineer",      ["machine learning engineer", "ml engineer", "mlops",
                          "model deployment", "model serving", "feature engineering"]),
    ("Data Scientist",   ["data scientist", "data science", "predictive model",
                          "statistical model", "hypothesis testing", "experimentation"]),
    ("AI Engineer",      ["ai engineer", "artificial intelligence", "llm",
                          "generative ai", "prompt engineering", "langchain",
                          "transformer", "deep learning"]),
    ("Data Engineer",    ["data engineer", "data pipeline", "etl", "elt",
                          "apache spark", "kafka", "airflow", "dbt",
                          "data warehouse", "data lake"]),
    ("Data Analyst",     ["data analyst", "business analyst", "reporting",
                          "dashboard", "tableau", "power bi", "looker",
                          "sql analyst", "bi developer"]),
    ("Backend Developer",["backend", "back-end", "api developer", "rest api",
                          "django", "fastapi", "flask", "microservices",
                          "node.js", "spring boot"]),
    ("DevOps Engineer",  ["devops", "site reliability", "sre", "infrastructure",
                          "kubernetes", "docker", "ci/cd", "terraform",
                          "ansible", "helm", "platform engineer"]),
    ("Full Stack Developer", ["full stack", "full-stack", "frontend", "react",
                              "angular", "vue", "next.js"]),
    ("Software Engineer",["software engineer", "software developer",
                          "swe", "software development"]),
]

# Pre-compile regex patterns with word boundaries for performance and accuracy
ROLE_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    (role, [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in keywords])
    for role, keywords in ROLE_RULES
]

OTHER_CATEGORY = "Other"


# ================================================================
#  STOPWORDS
#  Domain-specific additions on top of sklearn's English list.
# ================================================================

EXTRA_STOPWORDS: list[str] = [
    "experience", "required", "responsibilities", "skills", "ability",
    "work", "team", "years", "role", "position", "company", "job",
    "looking", "strong", "good", "knowledge", "understanding", "help",
    "must", "will", "using", "use", "also", "including", "well", "new",
    "need", "working", "candidate", "excellent", "opportunity", "join",
    "develop", "provide", "ensure", "support", "manage", "responsible",
    "preferred", "plus", "bonus", "minimum", "maximum",
]


# ================================================================
#  TEXT CLEANING
# ================================================================

def clean_text(text: str) -> str:
    """
    Lightweight text normalisation for TF-IDF input.

    Steps:
      - Lowercase
      - Remove HTML tags (sometimes present in raw API data)
      - Remove special characters, keep only letters and spaces
      - Collapse multiple spaces
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML
    text = re.sub(r"[^a-z\s]", " ", text)          # keep letters only
    text = re.sub(r"\s+", " ", text).strip()        # collapse spaces

    return text


# ================================================================
#  ROLE CATEGORISATION
# ================================================================

def categorise_role(title: str, description: str) -> str:
    """
    Assign a role category by checking job title first, then description.

    Parameters
    ----------
    title       : str  job_title value
    description : str  job_description value (already lowercased is fine)

    Returns
    -------
    str  one of the ROLE_RULES keys or 'Other'
    """
    # 1. Check title first (high precision)
    for role, patterns in ROLE_PATTERNS:
        for pattern in patterns:
            if pattern.search(title):
                return role

    # 2. Check description as fallback
    for role, patterns in ROLE_PATTERNS:
        for pattern in patterns:
            if pattern.search(description):
                return role

    return OTHER_CATEGORY


# ================================================================
#  TF-IDF: TOP KEYWORDS PER JOB
# ================================================================

def extract_top_keywords(
    descriptions: pd.Series,
    top_n: int = 10,
    max_features: int = 5000,
) -> list[str]:
    """
    Run TF-IDF across all job descriptions and return the top_n
    most important terms for each document.

    Parameters
    ----------
    descriptions : pd.Series  of cleaned job description strings
    top_n        : int         terms to extract per job
    max_features : int         vocabulary cap for TfidfVectorizer

    Returns
    -------
    list[str]  one comma-separated keyword string per row,
               in the same order as the input Series.
    """
    # Fill empty descriptions so TF-IDF doesn't fail on empty docs
    filled = descriptions.fillna("").apply(clean_text)

    # Need at least 2 non-empty documents for TF-IDF to be meaningful
    non_empty_count = (filled.str.strip() != "").sum()
    if non_empty_count < 2:
        logger.warning(
            "Only %d non-empty descriptions — skipping TF-IDF, "
            "returning empty keywords.", non_empty_count
        )
        return [""] * len(descriptions)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=list(
            TfidfVectorizer(stop_words="english")
            .get_stop_words()
            | set(EXTRA_STOPWORDS)
        ),
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=2,                 # ignore terms in fewer than 2 docs
        sublinear_tf=True,        # log(1+tf) for smoother scores
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(filled)
    except ValueError as e:
        logger.error("TF-IDF vectorisation failed: %s", e)
        return [""] * len(descriptions)

    feature_names = np.array(vectorizer.get_feature_names_out())
    results: list[str] = []

    for row_idx in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[row_idx].toarray().flatten()

        if row.sum() == 0:
            results.append("")
            continue

        # Get indices of top_n highest TF-IDF scores
        top_indices = row.argsort()[-top_n:][::-1]
        top_terms   = feature_names[top_indices].tolist()

        results.append(", ".join(top_terms))

    return results


# ================================================================
#  DATABASE HELPERS
# ================================================================

def _ensure_columns_exist() -> None:
    """
    Add role_category and top_keywords columns if they don't exist yet.
    Safe to run multiple times (IF NOT EXISTS).
    """
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE jobs
              ADD COLUMN IF NOT EXISTS role_category TEXT,
              ADD COLUMN IF NOT EXISTS top_keywords   TEXT;
        """))
    logger.info("Ensured role_category and top_keywords columns exist.")


def _write_results(df: pd.DataFrame) -> int:
    """
    Bulk-write role_category and top_keywords back to PostgreSQL.

    Returns the number of rows updated.
    """
    updated = 0

    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("""
                    UPDATE jobs
                    SET    role_category = :role,
                           top_keywords  = :keywords
                    WHERE  job_id = :job_id
                """),
                {
                    "role":     row["role_category"],
                    "keywords": row["top_keywords"],
                    "job_id":   row["job_id"],
                },
            )
            updated += 1

    return updated


# ================================================================
#  ANALYTICS HELPERS  (importable by eda.py / dashboard)
# ================================================================

def get_role_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Count jobs per role_category.

    Parameters
    ----------
    df : pd.DataFrame  must contain role_category column

    Returns
    -------
    pd.Series  role → count, sorted descending
    """
    if "role_category" not in df.columns:
        raise ValueError("DataFrame must contain 'role_category' column.")

    return (
        df["role_category"]
        .fillna(OTHER_CATEGORY)
        .value_counts()
    )


def get_skills_by_role(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each role category, return the top 5 most common skills.

    Returns
    -------
    pd.DataFrame  columns: role_category, skill, count
    """
    records = []

    for role, group in df.groupby("role_category"):
        all_skills: list[str] = []
        for cell in group["extracted_skills"].dropna():
            all_skills.extend(s.strip() for s in cell.split(",") if s.strip())

        skill_counts = (
            pd.Series(all_skills)
            .value_counts()
            .head(5)
        )

        for skill, count in skill_counts.items():
            records.append({"role_category": role, "skill": skill, "count": count})

    return pd.DataFrame(records)


# ================================================================
#  MAIN PIPELINE
# ================================================================

def run_nlp_pipeline() -> None:
    """
    Full NLP enrichment run:
      1. Ensure DB columns exist
      2. Load jobs
      3. Extract top TF-IDF keywords per job
      4. Categorise each job into a role
      5. Write results back to PostgreSQL
      6. Print summary
    """
    logger.info("Starting NLP pipeline...")

    # Step 1 — ensure columns
    _ensure_columns_exist()

    # Step 2 — load data
    df = pd.read_sql(
        text("SELECT job_id, job_title, job_description FROM jobs"),
        engine,
    )
    logger.info("Loaded %d jobs from PostgreSQL.", len(df))

    if df.empty:
        logger.warning("No jobs found — run the ETL pipeline first.")
        return

    # Step 3 — TF-IDF keywords
    logger.info("Running TF-IDF vectorisation...")
    df["top_keywords"] = extract_top_keywords(df["job_description"])

    # Step 4 — role categorisation
    logger.info("Categorising job roles...")
    df["role_category"] = df.apply(
        lambda row: categorise_role(
            str(row["job_title"]),
            str(row["job_description"]),
        ),
        axis=1,
    )

    # Step 5 — write back
    logger.info("Writing results to PostgreSQL...")
    updated = _write_results(df)
    logger.info("Updated %d rows.", updated)

    # Step 6 — summary
    print("\n── Role Distribution ──────────────────────────")
    print(get_role_distribution(df).to_string())

    print("\n── Sample Top Keywords ────────────────────────")
    sample = df[df["top_keywords"].str.strip() != ""][["job_title", "top_keywords"]].head(5)
    for _, row in sample.iterrows():
        print(f"\n  {row['job_title']}")
        print(f"  {row['top_keywords']}")

    logger.info("NLP pipeline complete.")


if __name__ == "__main__":
    run_nlp_pipeline()