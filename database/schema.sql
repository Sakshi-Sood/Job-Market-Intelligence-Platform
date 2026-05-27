-- ================================================================
-- database/schema.sql
-- Auto-executed by PostgreSQL container on first start.
-- Creates the jobs table with all columns including NLP fields.
-- ================================================================

CREATE TABLE IF NOT EXISTS jobs (
    job_id               VARCHAR(255) PRIMARY KEY,
    job_title            TEXT,
    employer_name        TEXT,
    job_city             TEXT         DEFAULT 'Not Specified',
    job_country          TEXT         DEFAULT 'Not Specified',
    job_employment_type  TEXT         DEFAULT 'Not Specified',
    job_is_remote        BOOLEAN      DEFAULT FALSE,
    job_posted_at        TIMESTAMPTZ,
    job_apply_link       TEXT,
    job_description      TEXT,
    job_min_salary       FLOAT        DEFAULT 0,
    job_max_salary       FLOAT        DEFAULT 0,
    job_salary_currency  TEXT         DEFAULT 'Not Specified',
    salary_available     BOOLEAN      DEFAULT FALSE,
    extracted_skills     TEXT,
    skill_count          INTEGER      DEFAULT 0,
    role_category        TEXT,
    top_keywords         TEXT,
    last_updated         TIMESTAMPTZ  DEFAULT NOW()
);

-- Index on commonly filtered columns
CREATE INDEX IF NOT EXISTS idx_jobs_city
    ON jobs (job_city);

CREATE INDEX IF NOT EXISTS idx_jobs_employment_type
    ON jobs (job_employment_type);

CREATE INDEX IF NOT EXISTS idx_jobs_remote
    ON jobs (job_is_remote);

CREATE INDEX IF NOT EXISTS idx_jobs_role
    ON jobs (role_category);

CREATE INDEX IF NOT EXISTS idx_jobs_posted_at
    ON jobs (job_posted_at DESC);