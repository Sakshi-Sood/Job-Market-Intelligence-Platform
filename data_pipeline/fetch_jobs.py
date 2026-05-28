"""
fetch_jobs.py — API ingestion + incremental sync orchestrator.

Fetches job listings from the JSearch API, cleans them,
stamps them with a `last_updated` timestamp, and upserts
into PostgreSQL via the load_data module.
"""

import logging
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from data_pipeline.clean_data import clean_jobs
from data_pipeline.load_data import upsert_jobs

from datetime import datetime, timezone

# ------------------------------------
# Logging
# ------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ------------------------------------
# Environment
# ------------------------------------

load_dotenv()

API_KEY = os.getenv("RAPIDAPI_KEY")
API_HOST = os.getenv("RAPIDAPI_HOST")

# ------------------------------------
# API Configuration
# ------------------------------------

URL = "https://jsearch.p.rapidapi.com/search"

# Default query — used when no user input is provided
DEFAULT_QUERY = "Data Scientist in India"
DEFAULT_NUM_PAGES = 1

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST,
}

# ------------------------------------
# Required columns (post-clean schema)
# ------------------------------------

REQUIRED_COLUMNS: list[str] = [
    "job_id",
    "job_title",
    "employer_name",
    "job_city",
    "job_country",
    "job_employment_type",
    "job_is_remote",
    "job_posted_at",
    "job_apply_link",
    "job_description",
    "job_min_salary",
    "job_max_salary",
    "job_salary_currency",
    "salary_available",
    "last_updated",
]


def run_pipeline(
    query: str | None = None,
    num_pages: int | None = None,
) -> None:
    """
    End-to-end ETL pipeline:

    1. Fetch jobs from the JSearch API.
    2. Clean and normalise the data.
    3. Stamp each record with `last_updated`.
    4. Upsert into PostgreSQL (insert new / update existing).
    5. Log sync metrics.

    Parameters
    ----------
    query      : str | None   Search query (e.g. "ML Engineer in USA").
                               Defaults to DEFAULT_QUERY if not provided.
    num_pages  : int | None   Number of API pages to fetch (1-5).
                               Defaults to DEFAULT_NUM_PAGES.
    """

    try:

        # =========================
        # FETCH API RESPONSE
        # =========================

        search_query = query or DEFAULT_QUERY
        pages = num_pages or DEFAULT_NUM_PAGES

        logger.info(
            "Fetching jobs from JSearch API — query=%r, pages=%d",
            search_query, pages,
        )

        query_params = {
            "query": search_query,
            "page": "1",
            "num_pages": str(pages),
        }

        response = requests.get(
            URL,
            headers=HEADERS,
            params=query_params,
        )
        response.raise_for_status()

        data = response.json()

        # =========================
        # EXTRACT JOB DATA
        # =========================

        jobs = data.get("data", [])

        if not jobs:
            logger.warning("No jobs returned by the API — aborting pipeline.")
            return

        # =========================
        # CONVERT TO DATAFRAME
        # =========================

        df = pd.DataFrame(jobs)

        logger.info("Total jobs fetched from API: %d", len(df))

        # =========================
        # SCHEMA DRIFT GUARD
        # =========================
        # Ensure every required column exists
        # (handles fields the API may omit).

        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = None
                logger.info(
                    "Added missing column '%s' with NULL default.", col
                )

        # =========================
        # CLEAN DATA
        # =========================

        df = clean_jobs(df)

        # =========================
        # ADD SYNC TIMESTAMP
        # =========================

        df["last_updated"] = datetime.now(timezone.utc)

        # Keep only required columns in the correct order
        df = df[REQUIRED_COLUMNS]

        # =========================
        # PREVIEW CLEANED DATA
        # =========================

        logger.info("Sample cleaned data:\n%s", df.head(3).to_string())

        # =========================
        # UPSERT INTO POSTGRESQL
        # =========================

        logger.info("Starting upsert synchronization …")

        metrics = upsert_jobs(df)

        logger.info(
            "Pipeline complete — "
            "total_records=%d | rows_affected=%d | status=%s",
            metrics["total_records"],
            metrics["rows_affected"],
            metrics["status"],
        )

    except requests.exceptions.RequestException:
        logger.exception("API request failed.")
        raise

    except Exception:
        logger.exception("Pipeline error occurred.")
        raise


# ------------------------------------
# Entry-point
# ------------------------------------

if __name__ == "__main__":
    run_pipeline()