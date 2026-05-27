"""
FastAPI REST API for the Job Market Intelligence Platform.

Exposes endpoints for:
  - Job listings (with filters)
  - EDA metrics  (cities, companies, employment, remote, salary)
  - NLP insights (top skills, role distribution, skills by role)
  - Pipeline trigger (run ETL + NLP on demand)
  - Health check


Docs auto-generated at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# ── logging ──────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

# ================================================================
#  LIFESPAN — runs on startup / shutdown
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Job Market API starting up...")
    # verify DB is reachable
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection verified.")
    except Exception as e:
        logger.error("Database connection failed on startup: %s", e)
    yield
    logger.info("Job Market API shutting down.")


# ================================================================
#  APP INSTANCE
# ================================================================

app = FastAPI(
    title="Job Market Intelligence API",
    description=(
        "REST API for the AI-Powered Job Market Analytics Platform. "
        "Provides job listings, EDA metrics, NLP insights, and pipeline triggers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit (same machine or Docker network) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
#  HELPER — load jobs with optional filters
# ================================================================

def _load_jobs(
    city: Optional[str]            = None,
    employment_type: Optional[str] = None,
    remote_only: bool              = False,
    salary_only: bool              = False,
    role: Optional[str]            = None,
    limit: int                     = 500,
) -> pd.DataFrame:
    """
    Load jobs from PostgreSQL and apply optional filters.
    Returns a pandas DataFrame.
    """
    query = "SELECT * FROM jobs WHERE 1=1"
    params: dict = {}

    if city:
        query += " AND job_city ILIKE :city"
        params["city"] = f"%{city}%"

    if employment_type:
        query += " AND job_employment_type = :emp_type"
        params["emp_type"] = employment_type

    if remote_only:
        query += " AND job_is_remote = TRUE"

    if salary_only:
        query += " AND salary_available = TRUE"

    if role:
        query += " AND role_category = :role"
        params["role"] = role

    query += " LIMIT :limit"
    params["limit"] = limit

    try:
        df = pd.read_sql(text(query), engine, params=params)
        return df
    except Exception as e:
        logger.error("DB query failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# ================================================================
#  RESPONSE MODELS
# ================================================================

class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    total_jobs: int


class MetricItem(BaseModel):
    label: str
    value: int


class SalaryMetrics(BaseModel):
    count: int
    average_salary: Optional[float]


class PipelineStatus(BaseModel):
    status: str
    message: str


# ================================================================
#  ROUTES
# ================================================================

# ── Health check ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Returns API health status and total job count.
    Used by Docker HEALTHCHECK and monitoring tools.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM jobs"))
            total  = result.scalar()
        return {"status": "ok", "db_connected": True, "total_jobs": total}
    except Exception as e:
        return {"status": "degraded", "db_connected": False, "total_jobs": 0}


# ── Jobs ─────────────────────────────────────────────────────────

@app.get("/jobs", tags=["Jobs"])
def get_jobs(
    city:            Optional[str] = Query(None, description="Filter by city (partial match)"),
    employment_type: Optional[str] = Query(None, description="FULLTIME / PARTTIME / CONTRACTOR"),
    remote_only:     bool          = Query(False, description="Show remote jobs only"),
    salary_only:     bool          = Query(False, description="Show only jobs with salary data"),
    role:            Optional[str] = Query(None, description="Filter by role category"),
    limit:           int           = Query(100, ge=1, le=500, description="Max results (1-500)"),
):
    """
    Retrieve job listings with optional filters.

    All filters are combinable. Results are capped at 500 per request.
    """
    df = _load_jobs(city, employment_type, remote_only, salary_only, role, limit)

    # Convert timestamps to strings for JSON serialisation
    for col in df.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns:
        df[col] = df[col].astype(str)

    # Replace NaN with None so FastAPI serialises to null
    df = df.where(df.notnull(), None)

    return {
        "total":    len(df),
        "filters":  {
            "city": city, "employment_type": employment_type,
            "remote_only": remote_only, "salary_only": salary_only,
            "role": role,
        },
        "jobs": df.to_dict(orient="records"),
    }


@app.get("/jobs/{job_id}", tags=["Jobs"])
def get_job(job_id: str):
    """Retrieve a single job by its job_id."""
    try:
        df = pd.read_sql(
            text("SELECT * FROM jobs WHERE job_id = :job_id"),
            engine,
            params={"job_id": job_id},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    for col in df.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns:
        df[col] = df[col].astype(str)

    df = df.where(df.notnull(), None)
    return df.to_dict(orient="records")[0]


# ── EDA Metrics ──────────────────────────────────────────────────

@app.get("/metrics/cities", tags=["EDA Metrics"])
def top_cities(top_n: int = Query(10, ge=1, le=50)):
    """Top hiring cities by job count."""
    df  = _load_jobs(limit=5000)
    res = get_top_cities(df, top_n=top_n)
    return [{"city": city, "count": int(count)} for city, count in res.items()]


@app.get("/metrics/companies", tags=["EDA Metrics"])
def top_companies(top_n: int = Query(10, ge=1, le=50)):
    """Top hiring companies by job count."""
    df  = _load_jobs(limit=5000)
    res = get_top_companies(df, top_n=top_n)
    return [{"company": company, "count": int(count)} for company, count in res.items()]


@app.get("/metrics/employment-types", tags=["EDA Metrics"])
def employment_types():
    """Distribution of employment types (FULLTIME, PARTTIME, etc.)."""
    df  = _load_jobs(limit=5000)
    res = get_employment_distribution(df)
    return [{"type": t, "count": int(c)} for t, c in res.items()]


@app.get("/metrics/remote", tags=["EDA Metrics"])
def remote_distribution():
    """Remote vs on-site job breakdown."""
    df  = _load_jobs(limit=5000)
    res = get_remote_distribution(df)
    label_map = {True: "Remote", False: "On-site"}
    return [
        {"label": label_map.get(k, str(k)), "count": int(v)}
        for k, v in res.items()
    ]


@app.get("/metrics/salary", response_model=SalaryMetrics, tags=["EDA Metrics"])
def salary_metrics():
    """Average salary and count of jobs with salary data."""
    df  = _load_jobs(limit=5000)
    res = get_salary_metrics(df)
    return {
        "count":          res.get("count", 0),
        "average_salary": res.get("average_salary"),
    }


@app.get("/metrics/data-quality", tags=["EDA Metrics"])
def data_quality():
    """Percentage of missing values per column."""
    df  = _load_jobs(limit=5000)
    res = get_data_quality_metrics(df)
    return [
        {"column": col, "missing_pct": round(float(pct), 2)}
        for col, pct in res.items()
    ]


# ── NLP / ML Insights ────────────────────────────────────────────

@app.get("/insights/skills", tags=["NLP Insights"])
def top_skills(top_n: int = Query(20, ge=1, le=50)):
    """
    Most in-demand skills extracted from job descriptions.
    Returns empty list if NLP pipeline has not been run yet.
    """
    df = _load_jobs(limit=5000)

    if "extracted_skills" not in df.columns or df["extracted_skills"].isna().all():
        return {"message": "Run skill_extractor first.", "skills": []}

    res = get_top_skills(df, top_n=top_n)
    return {
        "total_jobs_analysed": int((df["skill_count"].fillna(0) > 0).sum()),
        "skills": [
            {"skill": skill, "count": int(count)}
            for skill, count in res.items()
        ],
    }


@app.get("/insights/roles", tags=["NLP Insights"])
def role_distribution():
    """
    Breakdown of jobs by categorised role (Data Scientist, ML Engineer, etc.).
    Returns empty list if NLP pipeline has not been run yet.
    """
    df = _load_jobs(limit=5000)

    if "role_category" not in df.columns or df["role_category"].isna().all():
        return {"message": "Run nlp_pipeline first.", "roles": []}

    res = get_role_distribution(df)
    return [{"role": role, "count": int(count)} for role, count in res.items()]


@app.get("/insights/skills-by-role", tags=["NLP Insights"])
def skills_by_role():
    """Top 5 skills for each role category."""
    df = _load_jobs(limit=5000)

    if "role_category" not in df.columns or df["role_category"].isna().all():
        return {"message": "Run nlp_pipeline first.", "data": []}

    res = get_skills_by_role(df)
    return res.to_dict(orient="records")


# ── Pipeline trigger ──────────────────────────────────────────────

@app.post("/pipeline/run", response_model=PipelineStatus, tags=["Pipeline"])
def trigger_pipeline():
    """
    Trigger the full ETL + NLP pipeline synchronously.

    This is a blocking call — use it for demos and manual refreshes.
    For production, schedule via Airflow or Cron instead.
    """
    try:
        from data_pipeline.fetch_jobs      import run_pipeline
        from data_pipeline.skill_extractor import run_skill_extraction
        from data_pipeline.nlp_pipeline    import run_nlp_pipeline

        logger.info("Pipeline triggered via API...")
        run_pipeline()
        run_skill_extraction()
        run_nlp_pipeline()
        logger.info("Pipeline completed via API.")

        return {
            "status":  "success",
            "message": "ETL + NLP pipeline completed successfully.",
        }
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")