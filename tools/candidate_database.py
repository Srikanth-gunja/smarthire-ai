"""Candidate Database — SQLite-based read/query layer for candidates.

Single responsibility: Read, search, and query candidate records from
the SQLite database. Provides a simple interface for the HR Assistant
and other agents to look up candidate information.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from db.database import Database
from utils.observability import instrument_tool_methods

logger = logging.getLogger(__name__)


@instrument_tool_methods
class CandidateDatabase:
    """SQLite-based read/query layer for candidate records."""

    def __init__(self, csv_path: str | None = None) -> None:
        """Initialize the Candidate Database."""
        self.db = Database()
        # CSV remains a useful import/testing adapter; production workflows
        # use the persisted SQLite candidate records by default.
        self.csv_path = Path(csv_path) if csv_path else None

    def _load(self) -> list[dict[str, Any]]:
        """Load all candidates from the database.

        Returns:
            List of candidate dicts.
        """
        if self.csv_path is not None:
            if not self.csv_path.exists():
                return []
            with self.csv_path.open(encoding="utf-8", newline="") as file:
                return [dict(row) for row in csv.DictReader(file)]
        rows = self.db.get_candidates()
        return [dict(row) for row in rows]

    def get_candidate(self, candidate_id: str) -> dict | None:
        """Retrieve a candidate by ID.

        Args:
            candidate_id: The unique candidate identifier.

        Returns:
            Candidate dict if found, None otherwise.
        """
        candidates = self._load()
        for c in candidates:
            if c.get("id") == candidate_id or c.get("candidate_id") == candidate_id:
                return c
        return None

    def search_candidates(self, criteria: dict) -> list[dict]:
        """Search candidates by criteria.

        Supported criteria keys:
            - name: substring match (case-insensitive)
            - skills: at least one skill must be present
            - status: exact match (not used in DB, kept for compatibility)
            - min_experience: minimum experience years (numeric)

        Args:
            criteria: Dict of search criteria to match against.

        Returns:
            List of matching candidate dicts.
        """
        candidates = self._load()
        if not candidates:
            return []

        results = []
        for c in candidates:
            match = True

            if "name" in criteria and criteria["name"].lower() not in c.get("name", "").lower():
                match = False

            if "skills" in criteria:
                required_skills = [s.strip().lower() for s in criteria["skills"].split(",")]
                candidate_skills = c.get("extracted_skills") or c.get("skills", "")
                if isinstance(candidate_skills, str):
                    try:
                        import json
                        candidate_skills = json.loads(candidate_skills)
                    except json.JSONDecodeError:
                        candidate_skills = candidate_skills.split(",")
                if isinstance(candidate_skills, list):
                    candidate_skills_lower = [s.lower() for s in candidate_skills]
                else:
                    candidate_skills_lower = []
                if not any(s in candidate_skills_lower for s in required_skills):
                    match = False

            if "min_experience" in criteria:
                min_exp = float(criteria["min_experience"])
                experience = c.get("extracted_experience_years") or c.get("experience_years") or 0
                if float(experience) < min_exp:
                    match = False

            if "status" in criteria and c.get("status") != criteria["status"]:
                match = False

            if match:
                results.append(c)

        logger.info("Search with criteria %s returned %d results", criteria, len(results))
        return results

    def get_all_candidates(self) -> list[dict]:
        """Retrieve all candidates.

        Returns:
            List of all candidate dicts.
        """
        return self._load()

    def get_candidates_by_status(self, status: str) -> list[dict]:
        """Retrieve candidates filtered by status when it is available.

        Args:
            status: The status to filter by.

        Returns:
            Matching candidate dicts. SQLite candidates do not currently have
            a lifecycle status, so that source returns an empty list.
        """
        return self.search_candidates({"status": status})

    def count_candidates(self) -> int:
        """Count total candidates in the database.

        Returns:
            Total number of candidate records.
        """
        return len(self._load())

    def refresh(self) -> None:
        """Force reload the data (no-op since we query DB each time)."""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = CandidateDatabase()

    print(f"Total candidates: {db.count_candidates()}")
    print("\n--- All Candidates ---")
    for c in db.get_all_candidates():
        print(f"  {c['candidate_id']}: {c['name']} ({c['status']})")

    print("\n--- Search: Python skill ---")
    for c in db.search_candidates({"skills": "Python"}):
        print(f"  {c['candidate_id']}: {c['name']}")

    print("\n--- Search: shortlisted ---")
    for c in db.get_candidates_by_status("shortlisted"):
        print(f"  {c['candidate_id']}: {c['name']}")
