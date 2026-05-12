import logging
import pandas as pd
from sqlalchemy import text

from backend.database import engine

# ---------------------------------------------------------------------------
#  Logging — configured here at the entrypoint, not inside library modules
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

from analytics.metrics import (
    get_employment_distribution,
    get_top_cities,
    get_remote_distribution,
    get_salary_metrics,
    get_data_quality_metrics,
    get_top_companies
)

# =========================
# LOAD DATA FROM POSTGRESQL
# =========================

query = text("SELECT * FROM jobs")

df = pd.read_sql(query, engine)


# =========================
# BASIC DATASET INFO
# =========================

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())


# =========================
# MISSING VALUE ANALYSIS
# =========================

print("\nMissing Values:\n")

print(df.isnull().sum())


# =========================
# CREATE EDA DATASET
# =========================

eda_df = df.copy()


# Remove placeholder city values
eda_df = eda_df[
    eda_df["job_city"] != "Not Specified"
]


# Remove rows with missing employment type
eda_df = eda_df[
    eda_df["job_employment_type"] != "Not Specified"
]


# =========================
# EMPLOYMENT TYPE ANALYSIS
# =========================

print("\nEmployment Type Distribution:\n")

employment_distribution = (
    get_employment_distribution(eda_df)
)

print(employment_distribution)

# =========================
# TOP HIRING CITIES
# =========================

print("\nTop Hiring Cities:\n")

top_cities = (
    get_top_cities(eda_df)
)

print(top_cities)

# =========================
# TOP HIRING COMPANIES
# =========================

print("\nTop Hiring Companies:\n")

top_companies = get_top_companies(eda_df, top_n=10)
print(top_companies)


# =========================
# REMOTE JOB ANALYSIS
# =========================

print("\nRemote Job Distribution:\n")

remote_distribution = (
    get_remote_distribution(eda_df)
)

print(remote_distribution)


# =========================
# SALARY ANALYSIS
# =========================

salary_metrics = get_salary_metrics(eda_df)

print("\nSalary Metrics:\n")

print(salary_metrics)


# =========================
# DATA QUALITY METRICS
# =========================

print("\nData Quality Metrics:\n")

data_quality_metrics = (
    get_data_quality_metrics(df)
)

print(data_quality_metrics)


# =========================
# DATA QUALITY INSIGHTS
# =========================

print("\nData Quality Insights:\n")

print(
    f"Rows with valid city data: "
    f"{eda_df.shape[0]}"
)

print(
    f"Rows with salary data: "
    f"{salary_metrics['count']}"
)

remote_percentage = round(
    (
        eda_df["job_is_remote"]
        .fillna(False)
        .mean()
    ) * 100,
    2
)

print(
    f"Percentage Remote Jobs: "
    f"{remote_percentage}%"
)

from analytics.visualizations import generate_all_visualizations

# =========================
# VISUALIZATIONS
# =========================

print("\n[*] Generating visualizations...\n")

chart_paths = generate_all_visualizations(
    df=eda_df,
    top_cities=top_cities,
    top_companies=top_companies,
    employment_distribution=employment_distribution,
    remote_distribution=remote_distribution,
    missing_pct=data_quality_metrics,
)

import matplotlib.pyplot as plt

print("\n[+] All charts saved to: outputs/visualizations/")
print("[OK] EDA complete - rendering matplotlib figures...\n")

# Display all matplotlib figures at once (non-blocking)
plt.show()