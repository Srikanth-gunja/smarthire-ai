"""Integration tests for the SmartHire AI graph with mocked agents.

These tests verify the graph wiring: that the Supervisor routes to the
correct agents in the right order, that state flows between nodes, and
that the full pipeline produces the expected final_response.

All LLM and agent calls are mocked — we are testing graph shape and
state propagation, not agent internals.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from graph import build_graph
from memory.state import SmartHireState
from supervisor import Supervisor
from utils.models import (
    CandidateMatchingOutput,
    ExecutionPlan,
    HRAssistantOutput,
    InterviewSchedulingOutput,
    RankedCandidate,
    ResumeScreeningOutput,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    """A generic mock LLM (used only to satisfy Supervisor.__init__)."""
    return MagicMock()


@pytest.fixture
def mock_supervisor(mock_llm):
    """A Supervisor whose classify_intent is patched per-test."""
    sup = Supervisor(mock_llm)
    with patch.object(sup, "classify_intent"):
        yield sup


@pytest.fixture
def mock_resume_agent():
    """ResumeScreeningAgent mock that returns a fixed ScreeningResult."""
    agent = MagicMock()
    agent.screen_resume.return_value = ResumeScreeningOutput(
        candidate_name="Sarah Chen",
        skills=["Python", "React", "Docker", "PostgreSQL"],
        experience_years=6.0,
        education=[
            {"degree": "BS Computer Science", "institution": "MIT", "year": 2018}
        ],
        summary="Strong Python developer with full-stack experience.",
        match_score=85.0,
        extracted_fields={"certifications": [], "past_roles": []},
    )
    return agent


@pytest.fixture
def mock_matching_agent():
    """CandidateMatchingAgent mock that returns a single ranked candidate."""
    agent = MagicMock()
    agent.rank_candidates.return_value = CandidateMatchingOutput(
        ranked_candidates=[
            RankedCandidate(
                candidate_name="Sarah Chen",
                match_score=85.0,
                skills_match=["Python", "React"],
                skills_gap=["Kubernetes"],
                experience_match=True,
                justification="Strong match with core requirements.",
                rank=1,
            ),
        ],
        total_candidates_evaluated=1,
        summary="Top candidate identified: Sarah Chen.",
    )
    return agent


@pytest.fixture
def mock_scheduler_agent():
    """InterviewSchedulerAgent mock with no proposed slots."""
    agent = MagicMock()
    agent.propose_schedule.return_value = InterviewSchedulingOutput(
        proposed_slots=[],
        conflicts=[],
        summary="No interview slots could be proposed.",
    )
    return agent


@pytest.fixture
def mock_hr_agent():
    """HRAssistantAgent mock that returns a canned answer."""
    agent = MagicMock()
    agent.answer_query.return_value = HRAssistantOutput(
        answer="The hiring process has 5 stages: screening, matching, interview, offer, onboarding.",
        sources=["hr_knowledge_base.md"],
        confidence=0.92,
        needs_escalation=False,
    )
    return agent


def _build(
    mock_supervisor,
    mock_resume_agent,
    mock_matching_agent,
    mock_scheduler_agent,
    mock_hr_agent,
):
    """Helper to compile the graph with mocks."""
    return build_graph(
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    )


# ── Scenario A: resume_screening → candidate_matching ─────────────────


class TestResumeThenMatching:
    """Scenario a: resume+JD upload → screening then matching fires."""

    def test_agents_invoked_in_order(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="multi_intent",
            agents_to_invoke=["resume_screening", "candidate_matching"],
            reasoning="User wants to screen resumes and rank candidates.",
        )

        compiled = _build(
            mock_supervisor,
            mock_resume_agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [
                HumanMessage(content="Screen these resumes and rank the candidates")
            ],
            "job_description": {
                "job_title": "Senior Developer",
                "required_skills": ["Python", "React"],
            },
        }

        result = compiled.invoke(state)

        # Supervisor classified correctly
        assert result["current_intent"] == "multi_intent"

        # Both agents were invoked
        mock_resume_agent.screen_resume.assert_called_once()
        mock_matching_agent.rank_candidates.assert_called_once()

        # State was populated by both agents
        assert len(result.get("resumes", [])) == 1
        assert result["resumes"][0]["candidate_name"] == "Sarah Chen"
        assert len(result.get("candidate_rankings", [])) == 1
        assert result["candidate_rankings"][0]["candidate_name"] == "Sarah Chen"

        # active_agents should be drained
        assert result.get("active_agents", []) == []

        # Pipeline completed
        assert result.get("final_response")

        # Reflection notes populated
        notes = result.get("reflection_notes", {})
        assert "validation_passed" in notes
        assert "issues_found" in notes
        assert "checks_run" in notes
        assert len(notes["checks_run"]) == 4

    def test_state_flows_between_nodes(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="multi_intent",
            agents_to_invoke=["resume_screening", "candidate_matching"],
            reasoning="multi-step",
        )

        compiled = _build(
            mock_supervisor,
            mock_resume_agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [HumanMessage(content="Analyze resumes")],
            "job_description": {"job_title": "Dev"},
        }

        compiled.invoke(state)

        # Verify the matching agent received data from screening
        call_args = mock_matching_agent.rank_candidates.call_args
        input_to_matching = call_args[0][0]  # positional arg
        assert len(input_to_matching.resumes) == 1
        assert input_to_matching.resumes[0]["candidate_name"] == "Sarah Chen"

    def test_conversation_history_accumulates(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="resume_screening",
            agents_to_invoke=["resume_screening"],
            reasoning="screen only",
        )

        compiled = _build(
            mock_supervisor,
            mock_resume_agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [HumanMessage(content="Screen resumes")],
            "job_description": {},
        }

        result = compiled.invoke(state)

        # Should have: user msg + supervisor msg + screening msg = 3
        history = result["conversation_history"]
        assert len(history) >= 3
        assert isinstance(history[0], HumanMessage)


# ── Scenario B: interview_scheduling ─────────────────────────────────


class TestInterviewScheduling:
    """Scenario b: schedule an interview → interview_scheduling fires."""

    def test_correct_agent_invoked(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="interview_scheduling",
            agents_to_invoke=["interview_scheduling"],
            reasoning="User wants to schedule an interview.",
        )

        compiled = _build(
            mock_supervisor,
            mock_resume_agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [
                HumanMessage(content="Schedule an interview for candidate Sarah Chen")
            ],
            "candidate_rankings": [
                {"candidate_name": "Sarah Chen", "match_score": 85.0}
            ],
        }

        result = compiled.invoke(state)

        assert result["current_intent"] == "interview_scheduling"
        mock_resume_agent.screen_resume.assert_not_called()
        mock_matching_agent.rank_candidates.assert_not_called()
        mock_hr_agent.answer_query.assert_not_called()

        # The scheduler mock produces no slots, so reflection flags the
        # incomplete result and loops back once for a correction attempt.
        assert mock_scheduler_agent.propose_schedule.call_count == 2
        feedback_call = mock_scheduler_agent.propose_schedule.call_args[0][0]
        assert "interview scheduling but no slots" in (
            feedback_call.reflection_feedback or ""
        ).lower()

        # State has interview_slots
        assert "interview_slots" in result
        assert result.get("final_response")

        # Reflection notes populated + retry recorded
        notes = result.get("reflection_notes", {})
        assert "reflection_validated" in result
        assert notes["correction_attempted"] is True
        assert notes["retry_agent"] == "interview_scheduling"
        assert "validation_passed" in notes
        assert "issues_found" in notes

    def test_scheduler_receives_rankings_as_candidates(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="interview_scheduling",
            agents_to_invoke=["interview_scheduling"],
            reasoning="schedule",
        )

        compiled = _build(
            mock_supervisor,
            mock_resume_agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [
                HumanMessage(content="Schedule interviews for top candidates")
            ],
            "candidate_rankings": [
                {"candidate_name": "Alice", "match_score": 90.0},
                {"candidate_name": "Bob", "match_score": 80.0},
            ],
        }

        compiled.invoke(state)

        call_args = mock_scheduler_agent.propose_schedule.call_args
        input_to_scheduler = call_args[0][0]
        # Should pick top 3 candidates from rankings
        assert "Alice" in input_to_scheduler.candidates
        assert "Bob" in input_to_scheduler.candidates


# ── Scenario C: hr_assistant ─────────────────────────────────────────


class TestHRAssistant:
    """Scenario c: HR question → hr_assistant fires."""

    def test_correct_agent_invoked(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="hr_question",
            agents_to_invoke=["hr_assistant"],
            reasoning="User is asking about application status.",
        )

        compiled = _build(
            mock_supervisor,
            mock_resume_agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [
                HumanMessage(content="What's the status of my application?")
            ],
        }

        result = compiled.invoke(state)

        assert result["current_intent"] == "hr_question"
        mock_hr_agent.answer_query.assert_called_once()
        mock_resume_agent.screen_resume.assert_not_called()
        mock_matching_agent.rank_candidates.assert_not_called()
        mock_scheduler_agent.propose_schedule.assert_not_called()

        # hr_answers and final_response populated
        assert len(result.get("hr_answers", [])) == 1
        assert result["hr_answers"][0]["confidence"] == 0.92
        assert result.get("final_response")
        assert "hiring process" in result["final_response"].lower()

        # Reflection notes populated
        notes = result.get("reflection_notes", {})
        assert "validation_passed" in notes
        assert "issues_found" in notes

    def test_hr_receives_query_from_last_message(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="hr_question",
            agents_to_invoke=["hr_assistant"],
            reasoning="hr",
        )

        compiled = _build(
            mock_supervisor,
            mock_resume_agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [
                HumanMessage(content="What are the interview stages?")
            ],
        }

        compiled.invoke(state)

        call_args = mock_hr_agent.answer_query.call_args
        input_to_hr = call_args[0][0]
        assert "interview stages" in input_to_hr.query.lower()


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases: empty state, single agent, no agents."""

    def test_empty_agents_routes_to_memory_update(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="greeting",
            agents_to_invoke=[],
            reasoning="No agents needed.",
        )

        compiled = _build(
            mock_supervisor,
            mock_resume_agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [HumanMessage(content="Hello!")],
        }

        result = compiled.invoke(state)

        # No agents should have been called
        mock_resume_agent.screen_resume.assert_not_called()
        mock_matching_agent.rank_candidates.assert_not_called()
        mock_scheduler_agent.propose_schedule.assert_not_called()
        mock_hr_agent.answer_query.assert_not_called()

        # Pipeline still completes
        assert result.get("final_response")

    def test_single_agent_direct_to_memory_update(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="resume_screening",
            agents_to_invoke=["resume_screening"],
            reasoning="Just screening.",
        )

        compiled = _build(
            mock_supervisor,
            mock_resume_agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [HumanMessage(content="Screen this resume")],
            "job_description": {"job_title": "Dev"},
        }

        result = compiled.invoke(state)

        mock_resume_agent.screen_resume.assert_called_once()
        mock_matching_agent.rank_candidates.assert_not_called()
        assert result.get("resumes")
        assert result.get("final_response")

    def test_existing_state_is_extended_not_replaced(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="hr_question",
            agents_to_invoke=["hr_assistant"],
            reasoning="hr",
        )

        compiled = _build(
            mock_supervisor,
            mock_resume_agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [HumanMessage(content="Hi")],
            "job_description": {"job_title": "Preserved"},
            "resumes": [{"candidate_name": "Existing"}],
        }

        result = compiled.invoke(state)

        # Pre-existing data should survive
        assert result.get("job_description", {}).get("job_title") == "Preserved"
        assert len(result.get("resumes", [])) == 1
        assert result["resumes"][0]["candidate_name"] == "Existing"


# ── Reflection validation: broken candidate scenario ──────────────────


class TestReflectionCatchesIssues:
    """Verify reflection catches deliberately broken agent outputs."""

    def test_skill_mismatch_flagged(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        """A candidate with zero JD skill overlap is flagged by reflection."""
        # Mock the matching agent to return a candidate with NO matching skills
        mock_matching_agent.rank_candidates.return_value = CandidateMatchingOutput(
            ranked_candidates=[
                RankedCandidate(
                    candidate_name="Unqualified hire",
                    match_score=20.0,
                    skills_match=["Basket Weaving", "Juggling"],
                    skills_gap=["Python", "React", "Docker"],
                    experience_match=False,
                    justification="No matching skills at all.",
                    rank=1,
                ),
            ],
            total_candidates_evaluated=1,
            summary="Weak candidate pool.",
        )

        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="multi_intent",
            agents_to_invoke=["resume_screening", "candidate_matching"],
            reasoning="Screen and rank.",
        )

        compiled = _build(
            mock_supervisor,
            mock_resume_agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [
                HumanMessage(content="Screen and rank candidates")
            ],
            "job_description": {
                "job_title": "Senior Developer",
                "required_skills": ["Python", "React", "Docker"],
            },
        }

        result = compiled.invoke(state)

        # Reflection should have flagged the skill mismatch
        notes = result.get("reflection_notes", {})
        assert notes["validation_passed"] is False
        assert len(notes["issues_found"]) > 0
        assert any("Unqualified hire" in issue for issue in notes["issues_found"])
        assert any("zero overlap" in issue for issue in notes["issues_found"])

    def test_no_issues_when_skills_match(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        """Reflection passes when candidate skills match JD requirements."""
        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="multi_intent",
            agents_to_invoke=["resume_screening", "candidate_matching"],
            reasoning="Screen and rank.",
        )

        compiled = _build(
            mock_supervisor,
            mock_resume_agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [
                HumanMessage(content="Screen and rank candidates")
            ],
            "job_description": {
                "job_title": "Senior Developer",
                "required_skills": ["Python", "React"],
            },
        }

        result = compiled.invoke(state)

        notes = result.get("reflection_notes", {})
        assert notes["validation_passed"] is True
        assert notes["issues_found"] == []

    def test_retries_candidate_matching_with_feedback(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        """A failing skill check loops back to candidate_matching once,
        feeding the reflection feedback into the correction attempt."""
        # First call returns a candidate with zero JD skill overlap (fails
        # reflection); the correction attempt returns a valid one.
        broken_output = CandidateMatchingOutput(
            ranked_candidates=[
                RankedCandidate(
                    candidate_name="Unqualified hire",
                    match_score=15.0,
                    skills_match=["Juggling"],
                    skills_gap=["Python", "React"],
                    experience_match=False,
                    justification="No overlap.",
                    rank=1,
                ),
            ],
            total_candidates_evaluated=1,
            summary="Weak candidate pool.",
        )
        fixed_output = CandidateMatchingOutput(
            ranked_candidates=[
                RankedCandidate(
                    candidate_name="Sarah Chen",
                    match_score=85.0,
                    skills_match=["Python", "React"],
                    skills_gap=["Kubernetes"],
                    experience_match=True,
                    justification="Strong match.",
                    rank=1,
                ),
            ],
            total_candidates_evaluated=1,
            summary="Top candidate identified: Sarah Chen.",
        )
        mock_matching_agent.rank_candidates.side_effect = [
            broken_output,
            fixed_output,
        ]

        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="multi_intent",
            agents_to_invoke=["resume_screening", "candidate_matching"],
            reasoning="Screen and rank.",
        )

        compiled = _build(
            mock_supervisor,
            mock_resume_agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [
                HumanMessage(content="Screen and rank candidates")
            ],
            "job_description": {
                "job_title": "Senior Developer",
                "required_skills": ["Python", "React"],
            },
        }

        result = compiled.invoke(state)

        # Candidate Matching ran twice: initial + one correction attempt.
        assert mock_matching_agent.rank_candidates.call_count == 2

        # The correction attempt received the reflection feedback.
        retry_call = mock_matching_agent.rank_candidates.call_args[0][0]
        assert retry_call.reflection_feedback
        assert "zero overlap" in retry_call.reflection_feedback.lower()

        # Reflection ran twice and validated the corrected output.
        assert result.get("reflection_validated") is True
        notes = result.get("reflection_notes", {})
        assert notes["validation_passed"] is True
        assert notes["correction_attempted"] is True
        assert notes["retry_agent"] == "candidate_matching"
        assert notes["reflection_attempts"] == 2

        # The final rankings come from the corrected pass.
        assert result["candidate_rankings"][0]["candidate_name"] == "Sarah Chen"
