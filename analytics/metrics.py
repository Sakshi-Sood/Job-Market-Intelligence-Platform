import pandas as pd


# =========================
# EMPLOYMENT TYPE METRICS
# =========================

def get_employment_distribution(df):

    return (
        df["job_employment_type"]
        .value_counts()
    )


# =========================
# TOP HIRING CITIES
# =========================

def get_top_cities(df, top_n=10):

    valid_cities = df[
        df["job_city"] != "Not Specified"
    ]

    return (
        valid_cities["job_city"]
        .value_counts()
        .head(top_n)
    )


# =========================
# REMOTE JOB METRICS
# =========================

def get_remote_distribution(df):

    return (
        df["job_is_remote"]
        .fillna(False)
        .value_counts()
    )


# =========================
# SALARY METRICS
# =========================

def get_salary_metrics(df):

    salary_jobs = df[
        df["salary_available"]
        .fillna(False) == True
    ].copy()

    if salary_jobs.empty:

        return {
            "count": 0,
            "average_salary": None
        }

    salary_jobs["avg_salary"] = (
        salary_jobs["job_min_salary"] +
        salary_jobs["job_max_salary"]
    ) / 2

    return {
        "count": salary_jobs.shape[0],
        "average_salary": salary_jobs["avg_salary"].mean()
    }

# =========================
# DATA QUALITY METRICS
# =========================

def get_data_quality_metrics(df):

    total_rows = len(df)

    missing_percentages = (
        df.isnull().sum() / total_rows
    ) * 100

    return missing_percentages.sort_values(
        ascending=False
    )