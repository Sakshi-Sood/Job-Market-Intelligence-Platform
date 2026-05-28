import pandas as pd

def clean_jobs(df):

    # =========================
    # REQUIRED COLUMNS
    # =========================

    required_columns = [
        "job_id",
        "job_title",
        "employer_name",
        "job_city",
        "job_country",
        "job_employment_type",
        "job_is_remote",
        "job_posted_at_datetime_utc",
        "job_apply_link",
        "job_description",
        "job_min_salary",
        "job_max_salary",
        "job_salary_currency"
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    df = df[required_columns].copy()

    # Rename columns
    df.rename(
        columns={
            "job_posted_at_datetime_utc": "job_posted_at"
        },
        inplace=True
    )

    # Salary feature
    df["salary_available"] = (
        (df["job_min_salary"].fillna(0) > 0)
        |
        (df["job_max_salary"].fillna(0) > 0)
    )

    # Remove duplicate jobs
    df = df.drop_duplicates(subset=["job_id"])

    # Missing value analysis
    print("\nMissing Values:\n")
    print(df.isnull().sum())

    # Fill text-based missing values
    text_columns = [
        "job_city",
        "job_country",
        "job_employment_type",
        "job_description",
        "job_salary_currency"
    ]

    for col in text_columns:
        df[col] = df[col].fillna("Not Specified")

    # Fill boolean missing values
    df["job_is_remote"] = df["job_is_remote"].fillna(False)

    # Fill salary missing values
    df["job_min_salary"] = df["job_min_salary"].fillna(0)
    df["job_max_salary"] = df["job_max_salary"].fillna(0)

    # Clean descriptions
    df["job_description"] = (
        df["job_description"]
        .str.replace(r"\r\n", " ", regex=True)
        .str.lower()
    )

    # Convert timestamp
    df["job_posted_at"] = pd.to_datetime(
        df["job_posted_at"],
        errors="coerce"
    )

    # Convert NaT to None
    df["job_posted_at"] = df["job_posted_at"].where(
        df["job_posted_at"].notna(), None
    )

    return df