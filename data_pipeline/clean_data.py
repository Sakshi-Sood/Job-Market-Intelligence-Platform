import pandas as pd

def clean_jobs(df):

    # Remove duplicates
    df.drop_duplicates(subset="job_id", inplace=True)

    # Check missing values
    print("\nMissing Values:\n")
    print(df.isnull().sum())

    # Fill missing values
    df.fillna("Not Specified", inplace=True)

    # Clean text columns
    df["job_description"] = (
        df["job_description"]
        .str.replace(r"\r\n", " ", regex=True)
        .str.lower()
    )

    return df