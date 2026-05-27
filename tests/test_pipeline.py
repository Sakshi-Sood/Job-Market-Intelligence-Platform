# tests/test_pipeline.py
# ================================================================
# Unit tests for the core pipeline modules.
# These run in CI against a real PostgreSQL test instance.
# ================================================================

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ================================================================
#  CLEAN DATA TESTS
# ================================================================

from data_pipeline.clean_data import clean_jobs

class TestCleanJobs:

    def _sample_df(self):
        return pd.DataFrame([{
            "job_id":                    "test_001",
            "job_title":                 "Data Scientist",
            "employer_name":             "Acme Corp",
            "job_city":                  "Bangalore",
            "job_country":               "IN",
            "job_employment_type":       "FULLTIME",
            "job_is_remote":             True,
            "job_posted_at_datetime_utc":"2026-01-01T00:00:00Z",
            "job_apply_link":            "https://example.com",
            "job_description":           "Python, SQL, Machine Learning",
            "job_min_salary":            80000.0,
            "job_max_salary":            120000.0,
            "job_salary_currency":       "USD",
        }])

    def test_returns_dataframe(self):
        df = clean_jobs(self._sample_df())
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self):
        df = clean_jobs(self._sample_df())
        assert "job_id" in df.columns
        assert "salary_available" in df.columns

    def test_salary_available_true_when_salary_given(self):
        df = clean_jobs(self._sample_df())
        assert df["salary_available"].iloc[0] == True

    def test_salary_available_false_when_no_salary(self):
        raw = self._sample_df()
        raw["job_min_salary"] = None
        raw["job_max_salary"] = None
        df = clean_jobs(raw)
        assert df["salary_available"].iloc[0] == False

    def test_removes_duplicates(self):
        raw = pd.concat([self._sample_df(), self._sample_df()], ignore_index=True)
        df  = clean_jobs(raw)
        assert len(df) == 1

    def test_fills_missing_city(self):
        raw = self._sample_df()
        raw["job_city"] = None
        df = clean_jobs(raw)
        assert df["job_city"].iloc[0] == "Not Specified"

    def test_fills_missing_remote(self):
        raw = self._sample_df()
        raw["job_is_remote"] = None
        df = clean_jobs(raw)
        assert df["job_is_remote"].iloc[0] == False

    def test_description_lowercased(self):
        raw = self._sample_df()
        raw["job_description"] = "PYTHON AND SQL"
        df = clean_jobs(raw)
        assert df["job_description"].iloc[0] == "python and sql"


# ================================================================
#  SKILL EXTRACTOR TESTS
# ================================================================

from data_pipeline.skill_extractor import extract_skills_from_text, extract_skills

class TestSkillExtractor:

    def test_detects_python(self):
        skills = extract_skills_from_text("we need python and sql experience")
        assert "python" in skills

    def test_detects_multiple_skills(self):
        skills = extract_skills_from_text(
            "experience with python, docker, and aws required"
        )
        assert "python" in skills
        assert "docker" in skills
        assert "aws"    in skills

    def test_returns_sorted_list(self):
        skills = extract_skills_from_text("docker and python and aws")
        assert skills == sorted(skills)

    def test_empty_string_returns_empty(self):
        assert extract_skills_from_text("") == []

    def test_none_returns_empty(self):
        assert extract_skills_from_text(None) == []

    def test_no_false_positives(self):
        # 'r' should not match inside 'spark'
        skills = extract_skills_from_text("apache spark experience")
        assert "r" not in skills

    def test_case_insensitive(self):
        skills = extract_skills_from_text("PYTHON and DOCKER")
        assert "python" in skills
        assert "docker" in skills

    def test_extract_skills_adds_columns(self):
        df = pd.DataFrame([{
            "job_id":          "001",
            "job_description": "python and sql developer needed",
        }])
        result = extract_skills(df)
        assert "extracted_skills" in result.columns
        assert "skill_count"      in result.columns

    def test_skill_count_correct(self):
        df = pd.DataFrame([{
            "job_id":          "001",
            "job_description": "python, sql, docker",
        }])
        result = extract_skills(df)
        assert result["skill_count"].iloc[0] >= 3


# ================================================================
#  NLP PIPELINE TESTS
# ================================================================

from data_pipeline.nlp_pipeline import categorise_role, clean_text

class TestNlpPipeline:

    def test_categorise_data_scientist(self):
        role = categorise_role("Data Scientist", "predictive model experimentation")
        assert role == "Data Scientist"

    def test_categorise_ml_engineer(self):
        role = categorise_role("ML Engineer", "model deployment and mlops")
        assert role == "ML Engineer"

    def test_categorise_devops(self):
        role = categorise_role("DevOps Engineer", "kubernetes docker ci/cd terraform")
        assert role == "DevOps Engineer"

    def test_categorise_unknown_returns_other(self):
        role = categorise_role("Receptionist", "answering phone calls")
        assert role == "Other"

    def test_clean_text_lowercases(self):
        assert clean_text("PYTHON") == "python"

    def test_clean_text_strips_html(self):
        result = clean_text("<p>python developer</p>")
        assert "<p>" not in result
        assert "python" in result

    def test_clean_text_removes_special_chars(self):
        result = clean_text("python@developer!")
        assert "@" not in result
        assert "!" not in result

    def test_clean_text_empty_returns_empty(self):
        assert clean_text("") == ""

    def test_clean_text_none_returns_empty(self):
        assert clean_text(None) == ""


# ================================================================
#  METRICS TESTS
# ================================================================

from analytics.metrics import (
    get_employment_distribution,
    get_top_cities,
    get_remote_distribution,
    get_salary_metrics,
    get_top_companies,
)

class TestMetrics:

    def _sample_df(self):
        return pd.DataFrame([
            {
                "job_city": "Bangalore", "job_employment_type": "FULLTIME",
                "job_is_remote": True,  "employer_name": "TechCorp",
                "salary_available": True, "job_min_salary": 80000,
                "job_max_salary": 120000,
            },
            {
                "job_city": "Mumbai",    "job_employment_type": "PARTTIME",
                "job_is_remote": False, "employer_name": "DataCo",
                "salary_available": False, "job_min_salary": 0,
                "job_max_salary": 0,
            },
            {
                "job_city": "Bangalore", "job_employment_type": "FULLTIME",
                "job_is_remote": True,  "employer_name": "TechCorp",
                "salary_available": True, "job_min_salary": 90000,
                "job_max_salary": 130000,
            },
        ])

    def test_top_cities_returns_series(self):
        result = get_top_cities(self._sample_df())
        assert isinstance(result, pd.Series)

    def test_top_cities_correct_count(self):
        result = get_top_cities(self._sample_df())
        assert result["Bangalore"] == 2

    def test_employment_distribution(self):
        result = get_employment_distribution(self._sample_df())
        assert result["FULLTIME"] == 2

    def test_remote_distribution(self):
        result = get_remote_distribution(self._sample_df())
        assert result[True] == 2

    def test_salary_metrics_count(self):
        result = get_salary_metrics(self._sample_df())
        assert result["count"] == 2

    def test_salary_metrics_average(self):
        result = get_salary_metrics(self._sample_df())
        # avg of (80k+120k)/2=100k and (90k+130k)/2=110k → 105k
        assert result["average_salary"] == pytest.approx(105000.0)

    def test_top_companies(self):
        result = get_top_companies(self._sample_df())
        assert result["TechCorp"] == 2