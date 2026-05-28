"""
load_data.py — PostgreSQL UPSERT synchronization module.

Performs incremental upsert (INSERT ... ON CONFLICT DO UPDATE)
against the `jobs` table, ensuring rerun safety and update
propagation for all mutable fields.
"""

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert

from backend.database import engine

# ------------------------------------
# Logger
# ------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ------------------------------------
# Mutable columns updated on conflict
# ------------------------------------

UPSERT_COLUMNS: list[str] = [
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


def upsert_jobs(df: pd.DataFrame) -> dict[str, Any]:
    """
    Upsert a DataFrame of job records into PostgreSQL.

    - New rows are inserted.
    - Existing rows (matching `job_id`) are updated with the
      latest values for every mutable column.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned job data; must include a `job_id` column.

    Returns
    -------
    dict[str, Any]
        Sync metrics with keys:
            total_records, rows_affected, sync_timestamp, status
    """

    sync_timestamp = datetime.now(timezone.utc)

    # -- Prepare metrics skeleton --
    metrics: dict[str, Any] = {
        "total_records": 0,
        "rows_affected": 0,
        "sync_timestamp": sync_timestamp.isoformat(),
        "status": "pending",
    }

    try:
        # Convert NaN → None for PostgreSQL compatibility
        df = df.where(df.notnull(), None)

        # Also convert any remaining pandas NaT → Python None
        # (NaT in datetime columns isn't caught by notnull in all cases)
        import pandas as _pd
        for col in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
            df[col] = df[col].astype(object)
            df.loc[df[col].isna(), col] = None

        # Reflect the existing table schema
        metadata = MetaData()
        jobs_table = Table(
            "jobs",
            metadata,
            autoload_with=engine,
        )

        records = df.to_dict(orient="records")

        # Final safety: convert any remaining NaT to None in record dicts
        for rec in records:
            for key, val in rec.items():
                if isinstance(val, _pd.NaT.__class__) or (hasattr(val, 'isnull') and val != val):
                    rec[key] = None

        metrics["total_records"] = len(records)

        if not records:
            logger.warning("No records to sync — skipping upsert.")
            metrics["status"] = "skipped"
            return metrics

        logger.info(
            "Prepared %d record(s) for synchronization.", len(records)
        )

        # -- Build the UPSERT statement --
        with engine.begin() as connection:

            stmt = insert(jobs_table).values(records)

            update_set = {
                col: getattr(stmt.excluded, col)
                for col in UPSERT_COLUMNS
            }

            stmt = stmt.on_conflict_do_update(
                index_elements=["job_id"],
                set_=update_set,
            )

            logger.info("Executing upsert against PostgreSQL …")

            result = connection.execute(stmt)

            rows_affected = result.rowcount
            metrics["rows_affected"] = rows_affected
            metrics["status"] = "success"

            logger.info(
                "Sync complete — %d row(s) inserted/updated.", rows_affected
            )

    except Exception:
        metrics["status"] = "error"
        logger.exception("Upsert failed during synchronization.")
        raise

    return metrics