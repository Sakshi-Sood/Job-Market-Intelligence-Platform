import requests
import os
import pandas as pd
from dotenv import load_dotenv
from clean_data import clean_jobs

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

response = requests.get(url, headers=headers, params=querystring)

data = response.json()

# Extract only job data
jobs = data["data"]

# Convert to DataFrame
df = pd.DataFrame(jobs)

# Select important columns
df = df[
    [
        "job_id",
        "job_title",
        "employer_name",
        "job_city",
        "job_country",
        "job_employment_type",
        "job_apply_link",
        "job_description"
    ]
]

df = clean_jobs(df)
print(df.head())

