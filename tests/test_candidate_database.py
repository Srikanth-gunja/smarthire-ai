"""Tests for CandidateDatabase — tests Pandas read/query layer."""

from __future__ import annotations

import pytest

from tools.candidate_database import CandidateDatabase


@pytest.fixture
def db(tmp_path):
    """Create a CandidateDatabase with a test CSV."""
    csv_content = """candidate_id,name,email,phone,skills,experience_years,education,status,application_date
C001,Alice Smith,alice@test.com,(555) 111-1111,"Python,React,Docker",7.0,BS CS MIT,shortlisted,2025-01-15
C002,Bob Jones,bob@test.com,(555) 222-2222,"JavaScript,Node.js",4.0,BS IT UT Austin,applied,2025-01-18
C003,Carol White,carol@test.com,(555) 333-3333,"Python,SQL,TensorFlow",5.0,MS Data Science Columbia,interviewed,2025-01-20
"""
    csv_path = tmp_path / "test_candidates.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    return CandidateDatabase(csv_path=str(csv_path))


class TestCandidateDatabaseInit:
    """Tests for CandidateDatabase initialization."""

    def test_init_stores_path(self, tmp_path):
        """Database stores the CSV path."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("col1,col2\nval1,val2", encoding="utf-8")
        db = CandidateDatabase(csv_path=str(csv_path))
        assert db.csv_path == csv_path


class TestGetCandidate:
    """Tests for get_candidate method."""

    def test_get_existing_candidate(self, db):
        """Returns candidate dict for existing ID."""
        result = db.get_candidate("C001")
        assert result is not None
        assert result["name"] == "Alice Smith"
        assert result["status"] == "shortlisted"

    def test_get_nonexistent_candidate(self, db):
        """Returns None for non-existent ID."""
        result = db.get_candidate("C999")
        assert result is None

    def test_get_candidate_returns_dict(self, db):
        """Returns a dict with all columns."""
        result = db.get_candidate("C002")
        assert isinstance(result, dict)
        assert "email" in result
        assert "skills" in result


class TestSearchCandidates:
    """Tests for search_candidates method."""

    def test_search_by_name(self, db):
        """Search by name substring returns matching candidates."""
        results = db.search_candidates({"name": "Alice"})
        assert len(results) == 1
        assert results[0]["name"] == "Alice Smith"

    def test_search_by_name_case_insensitive(self, db):
        """Name search is case-insensitive."""
        results = db.search_candidates({"name": "alice"})
        assert len(results) == 1

    def test_search_by_skills(self, db):
        """Search by skills returns candidates with matching skill."""
        results = db.search_candidates({"skills": "Python"})
        assert len(results) == 2

    def test_search_by_status(self, db):
        """Search by status returns matching candidates."""
        results = db.search_candidates({"status": "shortlisted"})
        assert len(results) == 1
        assert results[0]["name"] == "Alice Smith"

    def test_search_by_min_experience(self, db):
        """Search by minimum experience filters correctly."""
        results = db.search_candidates({"min_experience": "6"})
        assert len(results) == 1
        assert results[0]["name"] == "Alice Smith"

    def test_search_combined_criteria(self, db):
        """Search with multiple criteria combines with AND logic."""
        results = db.search_candidates({"skills": "Python", "status": "shortlisted"})
        assert len(results) == 1

    def test_search_no_results(self, db):
        """Search returning no results gives empty list."""
        results = db.search_candidates({"name": "Nonexistent"})
        assert results == []


class TestGetAllCandidates:
    """Tests for get_all_candidates method."""

    def test_get_all(self, db):
        """Returns all candidates."""
        results = db.get_all_candidates()
        assert len(results) == 3

    def test_get_all_returns_dicts(self, db):
        """Each result is a dict."""
        results = db.get_all_candidates()
        for r in results:
            assert isinstance(r, dict)


class TestGetCandidatesByStatus:
    """Tests for get_candidates_by_status method."""

    def test_by_status(self, db):
        """Filters candidates by status."""
        results = db.get_candidates_by_status("applied")
        assert len(results) == 1
        assert results[0]["name"] == "Bob Jones"


class TestCountCandidates:
    """Tests for count_candidates method."""

    def test_count(self, db):
        """Returns correct count."""
        assert db.count_candidates() == 3


class TestRefresh:
    """Tests for refresh method."""

    def test_refresh_reloads(self, db):
        """Refresh forces reload of CSV data."""
        db.refresh()
        assert db.count_candidates() == 3


class TestMissingCSV:
    """Tests for missing CSV file handling."""

    def test_missing_csv(self, tmp_path):
        """Handles missing CSV file gracefully."""
        db = CandidateDatabase(csv_path=str(tmp_path / "nonexistent.csv"))
        assert db.get_candidate("C001") is None
        assert db.get_all_candidates() == []
        assert db.count_candidates() == 0
