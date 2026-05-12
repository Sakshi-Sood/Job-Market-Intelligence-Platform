import requests
import os
import pandas as pd

from dotenv import load_dotenv

from data_pipeline.clean_data import clean_jobs
from backend.database import engine

from data_pipeline.load_data import upsert_jobs

# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

API_KEY = os.getenv("RAPIDAPI_KEY")
API_HOST = os.getenv("RAPIDAPI_HOST")


# =========================
# API CONFIG
# =========================

url = "https://jsearch.p.rapidapi.com/search"

querystring = {
    "query": "Data Scientist in India",
    "page": "1",
    "num_pages": "1"
}

headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}


try:

    # =========================
    # FETCH API RESPONSE
    # =========================

    response = requests.get(
        url,
        headers=headers,
        params=querystring
    )

    response.raise_for_status()

    data = response.json()


    # =========================
    # EXTRACT JOB DATA
    # =========================

    jobs = data.get("data", [])

    if not jobs:
        print("No jobs found from API.")
        exit()


    # =========================
    # CONVERT TO DATAFRAME
    # =========================

    df = pd.DataFrame(jobs)

    print(f"\nTotal Jobs Fetched: {len(df)}")


    # =========================
    # CLEAN DATA
    # =========================

    df = clean_jobs(df)


    # =========================
    # PREVIEW CLEANED DATA
    # =========================

    print("\nSample Cleaned Data:\n")

    print(df.head(3))

    # =========================
    # REMOVE EXISTING DATABASE JOBS
    # =========================

    existing_jobs_query = """
    SELECT job_id FROM jobs
    """

    existing_ids = pd.read_sql(
        existing_jobs_query,
        engine
    )

    existing_ids = existing_ids["job_id"].tolist()


    df = df[
        ~df["job_id"].isin(existing_ids)
    ]

    # =========================
    # INSERT INTO POSTGRESQL
    # =========================

    upsert_jobs(df)

    print("\nData inserted successfully!")


except Exception as e:

    print("\nError occurred:")

    print(e)