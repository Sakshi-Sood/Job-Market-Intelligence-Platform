import pandas as pd

def clean_jobs(df):

    # Remove duplicate jobs
    df.drop_duplicates(subset="job_id", inplace=True)

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

    return df