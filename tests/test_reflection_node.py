"""Tests for the Reflection Node — PRD Section 12 validation checklist.

Covers the structured per-check output, retry decisions, corrections, and
persistence of reflection_validated / reflection_notes to the DB.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from agents.reflection_node import (
    CHECK_CANDIDATES,
    check_all_questions_answered,
    check_interview_schedule_conflicts,
    run_reflection,
    verify_candidate_recommendations_match_jd,
)
from db.database import Database
from tools.calendar_tool import CalendarTool


def _clean_state(**overrides) -> dict:
    """A state that should pass every reflection check."""
    state = {
        "conversation_history": [
            HumanMessage(content="Screen and rank these candidates")
        ],
        "job_description": {
            "job_title": "Senior Developer",
            "required_skills": ["Python", "React"],
        },
        "resumes": [
            {
                "candidate_name": "Sarah Chen",
                "skills": ["Python", "React", "Docker"],
            }
        ],
        "candidate_rankings": [
            {
                "candidate_name": "Sarah Chen",
                "match_score": 85.0,
                "skills_match": ["Python", "React"],
                "skills_gap": ["Kubernetes"],
                "experience_match": True,
                "justification": "Strong match.",
                "rank": 1,
            }
        ],
    }
    state.update(overrides)
    return state


@pytest.fixture
def no_db_writes():
    """Prevent run_reflection from touching the real SQLite DB in tests.

    run_reflection imports ``Database`` lazily via ``db.database``, so the
    class is patched at its definition site.
    """
    with patch("db.database.Database") as mock_db_cls:
        mock_db = MagicMock()
        mock_db.persist_job_description.return_value = "jd-test"
        mock_db_cls.return_value = mock_db
        yield mock_db


# ── Check (a): candidate recommendations vs JD ─────────────────────────


class TestCandidateSkillValidation:
    def test_passes_when_skills_overlap(self):
        result = verify_candidate_recommendations_match_jd(_clean_state())
        assert result["passed"] is True
        assert result["issues"] == []
        assert result["retry_hint"] is None

    def test_flags_candidate_with_zero_overlap(self):
        state = _clean_state(
            candidate_rankings=[
                {
                    "candidate_name": "Unqualified hire",
                    "match_score": 20.0,
                    "skills_match": ["Basket Weaving"],
                    "skills_gap": ["Python", "React"],
                    "experience_match": False,
                    "justification": "No overlap.",
                    "rank": 1,
                }
            ]
        )
        result = verify_candidate_recommendations_match_jd(state)
        assert result["passed"] is False
        assert any("Unqualified hire" in i for i in result["issues"])
        assert any("zero overlap" in i for i in result["issues"])
        assert result["retry_hint"] == "candidate_matching"

    def test_flags_fabricated_skill(self):
        # Candidate claims "Kubernetes" but the resume only lists Python.
        state = _clean_state(
            candidate_rankings=[
                {
                    "candidate_name": "Sarah Chen",
                    "match_score": 85.0,
                    "skills_match": ["Python", "Kubernetes"],
                    "skills_gap": [],
                    "experience_match": True,
                    "justification": "Match.",
                    "rank": 1,
                }
            ]
        )
        result = verify_candidate_recommendations_match_jd(state)
        assert result["passed"] is False
        assert any("fabrication" in i.lower() for i in result["issues"])
        assert result["retry_hint"] == "candidate_matching"

    def test_required_skill_claim_not_flagged_as_fabrication(self):
        # A JD-required skill claimed by the candidate is trusted even if the
        # resume skill list is abbreviated.
        state = _clean_state(
            candidate_rankings=[
                {
                    "candidate_name": "Sarah Chen",
                    "match_score": 85.0,
                    "skills_match": ["React"],
                    "skills_gap": [],
                    "experience_match": True,
                    "justification": "Match.",
                    "rank": 1,
                }
            ]
        )
        result = verify_candidate_recommendations_match_jd(state)
        assert result["passed"] is True


# ── Check (b): interview schedule conflicts ────────────────────────────


class TestScheduleConflictValidation:
    def test_passes_with_no_slots(self):
        result, cleaned = check_interview_schedule_conflicts({})
        assert result["passed"] is True
        assert cleaned == []

    def test_detects_and_removes_overlapping_slots(self):
        state = {
            "interview_slots": [
                {
                    "candidate_name": "Alice",
                    "date": "2026-08-10",
                    "time_start": "10:00",
                    "time_end": "11:00",
                    "interviewer": "Bob Tech Lead",
                    "interview_type": "technical",
                    "status": "proposed",
                },
                {
                    "candidate_name": "Carol",
                    "date": "2026-08-10",
                    "time_start": "10:30",
                    "time_end": "11:30",
                    "interviewer": "Bob Tech Lead",
                    "interview_type": "technical",
                    "status": "proposed",
                },
            ]
        }
        calendar = CalendarTool()
        result, cleaned = check_interview_schedule_conflicts(state, calendar)
        assert result["passed"] is False
        assert len(cleaned) == 1
        assert cleaned[0]["candidate_name"] == "Alice"
        assert result["retry_hint"] == "interview_scheduling"
        assert any("Conflict" in i for i in result["issues"])
        assert result["corrections"], "correction should record the removed slot"


# ── Check (c): query completeness ──────────────────────────────────────


class TestQueryCompleteness:
    def test_flags_missing_schedule_output(self):
        state = {
            "conversation_history": [
                HumanMessage(content="Schedule an interview for Sarah")
            ]
        }
        result = check_all_questions_answered(state)
        assert result["passed"] is False
        assert result["retry_hint"] == "interview_scheduling"

    def test_passes_when_intent_covered(self):
        state = _clean_state()
        result = check_all_questions_answered(state)
        assert result["passed"] is True

    def test_flags_missing_ranking_output(self):
        state = {
            "conversation_history": [
                HumanMessage(content="Rank the best candidates")
            ]
        }
        result = check_all_questions_answered(state)
        assert result["passed"] is False
        assert result["retry_hint"] == "candidate_matching"


# ── run_reflection orchestration ───────────────────────────────────────


class TestRunReflection:
    def test_structured_notes_output(self, no_db_writes):
        result = run_reflection(_clean_state())
        notes = result["reflection_notes"]

        assert result["reflection_validated"] is True
        assert notes["validation_passed"] is True
        assert notes["issues_found"] == []
        assert len(notes["checks_run"]) == 4
        assert len(notes["checks"]) == 4

        for check in notes["checks"]:
            assert set(check) >= {"check", "name", "passed", "issues", "corrections"}
            assert isinstance(check["passed"], bool)

        assert "reflection_attempts" in notes
        assert "correction_attempted" in notes
        assert "corrections_made" in notes

    def test_retry_agent_set_on_failure(self, no_db_writes):
        state = _clean_state(
            candidate_rankings=[
                {
                    "candidate_name": "Weak Hire",
                    "match_score": 15.0,
                    "skills_match": ["Juggling"],
                    "skills_gap": ["Python"],
                    "experience_match": False,
                    "justification": "Weak.",
                    "rank": 1,
                }
            ]
        )
        result = run_reflection(state)
        assert result["reflection_validated"] is False
        assert result["retry_agent"] == "candidate_matching"
        assert result["reflection_feedback"], "feedback must be provided to the agent"
        assert "candidate recommendations match jd" in result[
            "reflection_feedback"
        ].lower()
        assert result["reflection_attempts"] == 1

    def test_no_retry_on_second_pass(self, no_db_writes):
        # A prior pass already attempted a correction; the second pass must
        # not loop again even if validation still fails.
        state = _clean_state(
            reflection_attempts=1,
            candidate_rankings=[
                {
                    "candidate_name": "Weak Hire",
                    "match_score": 15.0,
                    "skills_match": ["Juggling"],
                    "skills_gap": ["Python"],
                    "experience_match": False,
                    "justification": "Weak.",
                    "rank": 1,
                }
            ],
        )
        result = run_reflection(state)
        assert result["reflection_validated"] is False
        assert result["retry_agent"] is None
        assert result["reflection_attempts"] == 2

    def test_correction_flag_carried_forward(self, no_db_writes):
        # Notes from a prior pass said a retry happened; the final pass notes
        # must still report it.
        prior_notes = {
            "correction_attempted": True,
            "retry_agent": "candidate_matching",
        }
        state = _clean_state(
            reflection_attempts=1,
            reflection_notes=prior_notes,
            candidate_rankings=[
                {
                    "candidate_name": "Weak Hire",
                    "match_score": 15.0,
                    "skills_match": ["Juggling"],
                    "skills_gap": ["Python"],
                    "experience_match": False,
                    "justification": "Weak.",
                    "rank": 1,
                }
            ],
        )
        result = run_reflection(state)
        notes = result["reflection_notes"]
        assert notes["correction_attempted"] is True
        assert notes["retry_agent"] == "candidate_matching"

    def test_conflict_cleans_slots_in_state(self, no_db_writes):
        state = _clean_state(
            interview_slots=[
                {
                    "candidate_name": "Alice",
                    "date": "2026-08-10",
                    "time_start": "10:00",
                    "time_end": "11:00",
                    "interviewer": "Bob Tech Lead",
                    "interview_type": "technical",
                    "status": "proposed",
                },
                {
                    "candidate_name": "Carol",
                    "date": "2026-08-10",
                    "time_start": "10:30",
                    "time_end": "11:30",
                    "interviewer": "Bob Tech Lead",
                    "interview_type": "technical",
                    "status": "proposed",
                },
            ]
        )
        result = run_reflection(state)
        assert result["reflection_validated"] is False
        assert result["retry_agent"] == "interview_scheduling"
        assert result["interview_slots"] == [state["interview_slots"][0]]
        assert result["reflection_notes"]["revised_slots"] == [
            state["interview_slots"][0]
        ]

    def test_persists_to_db(self, no_db_writes):
        run_reflection(_clean_state())
        mock_db = no_db_writes
        mock_db.update_ranking_reflection.assert_called_once()
        _jd_id, validated, notes = mock_db.update_ranking_reflection.call_args[0]
        assert validated is True
        assert notes["validation_passed"] is True
        # The persisted notes must be JSON-serialisable (structured checks).
        json.dumps(notes)


# ── DB persistence layer ───────────────────────────────────────────────


class TestReflectionPersistence:
    def test_update_ranking_reflection_stores_json(self, tmp_path):
        db = Database(db_path=tmp_path / "test.db")
        db.init_db()

        jd_id = db.persist_job_description(
            {"job_title": "Dev", "required_skills": ["Python"]}
        )
        cand_id = db.persist_candidate("Alice", "resume text", skills=["Python"])
        db.insert_candidate_ranking(cand_id, jd_id, 80.0, rank_position=1)

        notes = {
            "validation_passed": False,
            "checks": [{"check": CHECK_CANDIDATES, "passed": False}],
        }
        db.update_ranking_reflection(jd_id, False, notes)

        row = db.fetch_one(
            "SELECT reflection_validated, reflection_notes "
            "FROM candidate_rankings WHERE jd_id = ?",
            (jd_id,),
        )
        assert row["reflection_validated"] == 0
        stored = json.loads(row["reflection_notes"])
        assert stored["validation_passed"] is False
        assert stored["checks"][0]["check"] == CHECK_CANDIDATES

    def test_update_ranking_reflection_noop_without_jd(self, tmp_path):
        db = Database(db_path=tmp_path / "test.db")
        db.init_db()
        db.update_ranking_reflection(None, True, {"validation_passed": True})
