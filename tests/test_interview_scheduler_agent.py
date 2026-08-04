"""Tests for InterviewSchedulerAgent — tests plumbing with mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.interview_scheduler_agent import InterviewSchedulerAgent
from tools.calendar_tool import CalendarTool
from utils.models import InterviewSchedulingInput, InterviewSchedulingOutput


@pytest.fixture
def mock_llm():
    """Create a mock ChatOllama LLM."""
    return MagicMock()


@pytest.fixture
def calendar(tmp_path):
    """Create a CalendarTool with a temporary store."""
    return CalendarTool(store_path=str(tmp_path / "test_slots.json"))


@pytest.fixture
def agent(mock_llm, calendar):
    """Create an InterviewSchedulerAgent with mocked LLM and real calendar."""
    return InterviewSchedulerAgent(mock_llm, calendar)


class TestInterviewSchedulerAgentInit:
    """Tests for InterviewSchedulerAgent initialization."""

    def test_init_stores_llm(self, mock_llm, calendar):
        """Agent stores the LLM instance."""
        agent = InterviewSchedulerAgent(mock_llm, calendar)
        assert agent.llm is mock_llm

    def test_init_stores_calendar(self, mock_llm, calendar):
        """Agent stores the CalendarTool instance."""
        agent = InterviewSchedulerAgent(mock_llm, calendar)
        assert agent.calendar is calendar


class TestProposeSchedule:
    """Tests for propose_schedule method."""

    def test_propose_schedule_returns_output(self, agent):
        """propose_schedule returns an InterviewSchedulingOutput."""
        input_data = InterviewSchedulingInput(
            candidates=["Sarah Chen"],
            availability=[
                {
                    "candidate_name": "Sarah Chen",
                    "date": "2025-02-10",
                    "preferred_times": [{"time_start": "10:00", "time_end": "12:00"}],
                },
            ],
            interviewer_preferences={
                "interviewers": ["Alice Manager"],
                "interview_types": ["technical"],
            },
        )
        result = agent.propose_schedule(input_data)
        assert isinstance(result, InterviewSchedulingOutput)
        assert len(result.proposed_slots) == 1
        assert result.proposed_slots[0].candidate_name == "Sarah Chen"

    def test_propose_schedule_multiple_candidates(self, agent):
        """propose_schedule handles multiple candidates."""
        input_data = InterviewSchedulingInput(
            candidates=["Sarah Chen", "James Okafor"],
            availability=[
                {
                    "candidate_name": "Sarah Chen",
                    "date": "2025-02-10",
                    "preferred_times": [{"time_start": "10:00", "time_end": "12:00"}],
                },
                {
                    "candidate_name": "James Okafor",
                    "date": "2025-02-10",
                    "preferred_times": [{"time_start": "14:00", "time_end": "16:00"}],
                },
            ],
            interviewer_preferences={
                "interviewers": ["Alice Manager"],
                "interview_types": ["technical", "behavioral"],
            },
        )
        result = agent.propose_schedule(input_data)
        assert len(result.proposed_slots) == 2

    def test_propose_schedule_no_availability(self, agent):
        """propose_schedule handles missing availability data."""
        input_data = InterviewSchedulingInput(
            candidates=["Sarah Chen"],
            availability=[],
            interviewer_preferences={},
        )
        result = agent.propose_schedule(input_data)
        assert len(result.proposed_slots) == 0


class TestDetectConflicts:
    """Tests for _detect_conflicts method."""

    def test_no_conflicts(self, agent):
        """No conflicts when slots don't overlap."""
        from utils.models import InterviewSlot
        slots = [
            InterviewSlot(
                candidate_name="A", date="2025-02-10",
                time_start="10:00", time_end="11:00",
                interviewer="Alice", interview_type="technical", status="confirmed",
            ),
            InterviewSlot(
                candidate_name="B", date="2025-02-10",
                time_start="14:00", time_end="15:00",
                interviewer="Alice", interview_type="behavioral", status="confirmed",
            ),
        ]
        conflicts = agent._detect_conflicts(slots)
        assert len(conflicts) == 0

    def test_interviewer_conflict(self, agent):
        """Conflict detected when same interviewer has overlapping slots."""
        from utils.models import InterviewSlot
        slots = [
            InterviewSlot(
                candidate_name="A", date="2025-02-10",
                time_start="10:00", time_end="11:00",
                interviewer="Alice", interview_type="technical", status="confirmed",
            ),
            InterviewSlot(
                candidate_name="B", date="2025-02-10",
                time_start="10:30", time_end="11:30",
                interviewer="Alice", interview_type="behavioral", status="confirmed",
            ),
        ]
        conflicts = agent._detect_conflicts(slots)
        assert len(conflicts) > 0
        assert "Overlap" in conflicts[0]

    def test_candidate_conflict(self, agent):
        """Conflict detected when same candidate has overlapping slots."""
        from utils.models import InterviewSlot
        slots = [
            InterviewSlot(
                candidate_name="A", date="2025-02-10",
                time_start="10:00", time_end="11:00",
                interviewer="Alice", interview_type="technical", status="confirmed",
            ),
            InterviewSlot(
                candidate_name="A", date="2025-02-10",
                time_start="10:30", time_end="11:30",
                interviewer="Bob", interview_type="behavioral", status="confirmed",
            ),
        ]
        conflicts = agent._detect_conflicts(slots)
        assert any("Candidate conflict" in c for c in conflicts)


class TestGetAvailabilityForCandidate:
    """Tests for _get_availability_for_candidate method."""

    def test_get_availability(self, agent):
        """Returns availability for the specified candidate."""
        availability = [
            {"candidate_name": "Alice", "date": "2025-02-10"},
            {"candidate_name": "Bob", "date": "2025-02-10"},
        ]
        result = agent._get_availability_for_candidate("Alice", availability)
        assert len(result) == 1
        assert result[0]["candidate_name"] == "Alice"

    def test_get_availability_not_found(self, agent):
        """Returns empty list when candidate not found."""
        availability = [{"candidate_name": "Alice", "date": "2025-02-10"}]
        result = agent._get_availability_for_candidate("Bob", availability)
        assert result == []
