from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import MetaData, Table

from backend.database import engine


def upsert_jobs(df):

    # Convert NaN to None
    df = df.where(
        df.notnull(),
        None
    )

    metadata = MetaData()

    jobs_table = Table(
        "jobs",
        metadata,
        autoload_with=engine
    )

    records = df.to_dict(
        orient="records"
    )

    if not records:
        print("\nNo records to insert.")
        return

    with engine.begin() as connection:

        stmt = insert(
            jobs_table
        )

        stmt = stmt.values(records)

        stmt = stmt.on_conflict_do_nothing(
            index_elements=["job_id"]
        )

        result = connection.execute(stmt)

        print(
            f"\nInserted Rows: "
            f"{result.rowcount}"
        )