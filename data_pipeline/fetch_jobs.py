import requests
import os
import pandas as pd

from dotenv import load_dotenv

from data_pipeline.clean_data import clean_jobs
from backend.database import engine


# Load environment variables
load_dotenv()

API_KEY = os.getenv("RAPIDAPI_KEY")
API_HOST = os.getenv("RAPIDAPI_HOST")

url = "https://jsearch.p.rapidapi.com/search"

querystring = {
    "query": "Python Developer in India",
    "page": "1",
    "num_pages": "1"
}

headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}


try:
    # Fetch API response
    response = requests.get(
        url,
        headers=headers,
        params=querystring
    )

    # Raise error for bad responses
    response.raise_for_status()

    data = response.json()

    # Extract job data
    jobs = data.get("data", [])

    if not jobs:
        print("No jobs found from API.")
        exit()

    # Convert to DataFrame
    df = pd.DataFrame(jobs)

    # Debugging: view API columns
    print("\nColumns received from API:\n")
    print(df.columns.tolist())

    # Required columns
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

    # Add missing columns dynamically
    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    # Select columns
    df = df[required_columns]

    # Rename timestamp column
    df.rename(
        columns={
            "job_posted_at_datetime_utc": "job_posted_at"
        },
        inplace=True
    )

    # Salary availability feature
    df["salary_available"] = (
        (df["job_min_salary"].fillna(0) > 0) |
        (df["job_max_salary"].fillna(0) > 0)
    )

    # Clean data
    df = clean_jobs(df)

    # Preview cleaned data
    print("\nCleaned Data:\n")
    print(df.head())

    # Insert into PostgreSQL
    df.to_sql(
        name="jobs",
        con=engine,
        if_exists="append",
        index=False
    )

    print("\nData inserted successfully!")

except Exception as e:
    print("\nError occurred:")
    print(e)