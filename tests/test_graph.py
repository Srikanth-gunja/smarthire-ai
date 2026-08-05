"""Integration tests for the SmartHire AI graph with mocked agents.

These tests verify the graph wiring: that the Supervisor routes to the
correct agents in the right order, that state flows between nodes, and
that the full pipeline produces the expected final_response.

All LLM and agent calls are mocked — we are testing graph shape and
state propagation, not agent internals.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from graph import (
    _parse_recruiter_results_intent,
    answer_candidate_query,
    answer_recruiter_chat,
    build_graph,
)
from memory.state import SmartHireState
from supervisor import Supervisor
from utils.models import (
    CandidateMatchingOutput,
    ExecutionPlan,
    HRAssistantOutput,
    InterviewSchedulingOutput,
    RankedCandidate,
    ResumeScreeningOutput,
    SupervisorInput,
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
    """ResumeScreeningAgent mock that returns a fixed ScreeningResult.

    The graph node now drives screening through the parallel
    ``screen_batch_async`` interface, so the mock exposes an awaitable batch
    that returns one output per non-empty resume input (and an exception for
    empty ones, mirroring the real agent's isolation behaviour).
    """
    agent = MagicMock()
    output = ResumeScreeningOutput(
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
    agent.screen_resume = MagicMock(return_value=output)

    async def _fake_batch(resumes, jd_data, on_progress=None, jd_analysis_task=None):
        results = []
        for index, resume in enumerate(resumes):
            text = str(resume.get("text") or "").strip()
            if not text:
                results.append(ValueError("Resume contains no extractable text."))
                continue
            results.append(output)
            if on_progress:
                on_progress(
                    index + 1,
                    len(resumes),
                    resume.get("name") or "Resume",
                    output,
                    None,
                )
        return results

    agent.screen_batch_async = MagicMock(side_effect=_fake_batch)
    return agent


@pytest.fixture
def mock_matching_agent():
    """CandidateMatchingAgent mock that returns a single ranked candidate."""
    agent = MagicMock()
    agent.rank_candidates_async = AsyncMock(
        return_value=CandidateMatchingOutput(
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
        mock_resume_agent.screen_batch_async.assert_called_once()
        mock_matching_agent.rank_candidates_async.assert_called_once()

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
        call_args = mock_matching_agent.rank_candidates_async.call_args
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
        mock_resume_agent.screen_batch_async.assert_not_called()
        mock_matching_agent.rank_candidates_async.assert_not_called()
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
        mock_resume_agent.screen_batch_async.assert_not_called()
        mock_matching_agent.rank_candidates_async.assert_not_called()
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

    def test_hr_receives_role_and_candidate_context_is_scoped(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        """A candidate's HR context never contains recruiter-side workflow data."""
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
                HumanMessage(content="When is my interview?")
            ],
            "user_role": "candidate",
            "candidate_rankings": [
                {"candidate_name": "Sarah Chen", "match_score": 85.0}
            ],
            "interview_slots": [
                {"candidate_name": "Sarah Chen", "date": "2026-08-10"}
            ],
        }

        compiled.invoke(state)

        call_args = mock_hr_agent.answer_query.call_args
        input_to_hr = call_args[0][0]
        assert input_to_hr.user_role == "candidate"
        assert "candidate_rankings" not in input_to_hr.context
        assert "interview_slots" not in input_to_hr.context

    def test_hr_receives_recruiter_workflow_context(
        self,
        mock_supervisor,
        mock_resume_agent,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        """A recruiter's HR context does include workflow data."""
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
                HumanMessage(content="Show candidate summaries")
            ],
            "user_role": "recruiter",
            "candidate_rankings": [
                {"candidate_name": "Sarah Chen", "match_score": 85.0}
            ],
        }

        compiled.invoke(state)

        call_args = mock_hr_agent.answer_query.call_args
        input_to_hr = call_args[0][0]
        assert input_to_hr.user_role == "recruiter"
        assert "candidate_rankings" in input_to_hr.context


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
        mock_resume_agent.screen_batch_async.assert_not_called()
        mock_matching_agent.rank_candidates_async.assert_not_called()
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

        mock_resume_agent.screen_batch_async.assert_called_once()
        mock_matching_agent.rank_candidates_async.assert_not_called()
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


# ── Supervisor role policy ────────────────────────────────────────────


class TestParallelScreening:
    """Parallel screening: per-resume failures are isolated and recorded."""

    def test_failed_resume_recorded_and_rest_still_ranked(
        self,
        mock_supervisor,
        mock_matching_agent,
        mock_scheduler_agent,
        mock_hr_agent,
    ):
        """One bad resume must not cancel the batch or stall downstream agents."""
        mock_supervisor.classify_intent.return_value = ExecutionPlan(
            intent="multi_intent",
            agents_to_invoke=["resume_screening", "candidate_matching"],
            reasoning="Screen and rank.",
        )

        good = ResumeScreeningOutput(
            candidate_name="Good Candidate",
            skills=["Python", "React"],
            experience_years=4.0,
            education=[],
            summary="Good match.",
            match_score=80.0,
            extracted_fields={"certifications": [], "past_roles": []},
        )
        agent = MagicMock()

        async def _fake_batch(resumes, jd_data, on_progress=None, jd_analysis_task=None):
            results = []
            for resume in resumes:
                if "good" in str(resume.get("name", "")).lower():
                    results.append(good)
                else:
                    results.append(ValueError("Corrupt file: could not parse resume."))
            return results

        agent.screen_batch_async = MagicMock(side_effect=_fake_batch)

        compiled = build_graph(
            mock_supervisor,
            agent,
            mock_matching_agent,
            mock_scheduler_agent,
            mock_hr_agent,
        )

        state: SmartHireState = {
            "conversation_history": [
                HumanMessage(content="Screen and rank candidates")
            ],
            "resume_inputs": [
                {"name": "good.txt", "text": "Good resume"},
                {"name": "bad.txt", "text": "Bad resume"},
            ],
            "job_description": {
                "job_title": "Senior Developer",
                "required_skills": ["Python", "React"],
            },
        }

        result = compiled.invoke(state)

        # The good resume screened; the bad one is recorded as a failure.
        assert len(result.get("resumes", [])) == 1
        assert result["resumes"][0]["candidate_name"] == "Good Candidate"
        assert len(result.get("screening_failures", [])) == 1
        failure = result["screening_failures"][0]
        assert failure["screening_status"] == "failed"
        assert failure["filename"] == "bad.txt"
        assert "could not parse" in failure["error"]

        # The rest of the pipeline still ran with the successful batch.
        mock_matching_agent.rank_candidates_async.assert_called_once()
        assert result.get("candidate_rankings")


class TestSupervisorRolePolicy:
    """Role-aware routing: candidates only reach the HR Assistant."""

    def _supervisor_with_plan(self, mock_llm, plan: ExecutionPlan) -> Supervisor:
        structured = MagicMock()
        structured.invoke.return_value = plan
        mock_llm.with_structured_output.return_value = structured
        return Supervisor(mock_llm)

    def test_candidate_scheduling_query_routes_to_hr_only(self, mock_llm):
        """Even an 'interview/schedule' query from a candidate stays in HR."""
        sup = self._supervisor_with_plan(
            mock_llm,
            ExecutionPlan(
                intent="interview_scheduling",
                agents_to_invoke=["interview_scheduling"],
                reasoning="User wants to schedule an interview.",
            ),
        )
        result = sup.classify_intent(
            SupervisorInput(user_query="When is my interview?", user_role="candidate")
        )
        assert result.intent == "hr_question"
        assert result.agents_to_invoke == ["hr_assistant"]

    def test_candidate_hr_query_routes_to_hr(self, mock_llm):
        sup = self._supervisor_with_plan(
            mock_llm,
            ExecutionPlan(
                intent="hr_question",
                agents_to_invoke=["hr_assistant"],
                reasoning="Status question.",
            ),
        )
        result = sup.classify_intent(
            SupervisorInput(user_query="What is my application status?", user_role="candidate")
        )
        assert result.agents_to_invoke == ["hr_assistant"]

    def test_recruiter_agents_preserved(self, mock_llm):
        """Recruiter routing is unchanged by the role policy."""
        sup = self._supervisor_with_plan(
            mock_llm,
            ExecutionPlan(
                intent="interview_scheduling",
                agents_to_invoke=["interview_scheduling"],
                reasoning="Schedule for shortlisted candidate.",
            ),
        )
        result = sup.classify_intent(
            SupervisorInput(user_query="Schedule an interview", user_role="recruiter")
        )
        assert result.agents_to_invoke == ["interview_scheduling"]

    def test_fallback_candidate_routes_to_hr(self, mock_llm):
        """The keyword fallback is role-aware too."""
        sup = Supervisor(mock_llm)
        result = sup._fallback_classify("screen resumes please", role="candidate")
        assert result.agents_to_invoke == ["hr_assistant"]

    def test_fallback_recruiter_routes_to_screening(self, mock_llm):
        """Recruiter fallback keeps recruiter routing."""
        sup = Supervisor(mock_llm)
        result = sup._fallback_classify("screen resumes please", role="recruiter")
        assert result.agents_to_invoke == ["resume_screening"]


class TestAnswerCandidateQuery:
    """Candidate chat bypasses the pipeline and answers via the HR Assistant."""

    @patch("utils.llm_factory.get_llm")
    @patch("agents.hr_assistant_agent.HRAssistantAgent")
    def test_answers_directly_with_candidate_role(self, mock_hr_cls, mock_get_llm):
        mock_hr = MagicMock()
        mock_hr.answer_query.return_value = HRAssistantOutput(
            answer="Most roles have 3 interview rounds.",
            sources=["interview_rounds"],
            confidence=0.9,
            needs_escalation=False,
        )
        mock_hr_cls.return_value = mock_hr

        answer = answer_candidate_query(
            "How many interview rounds are there?",
            "sess-123",
            {"prior_answers": ["previous answer"]},
        )

        assert answer == "Most roles have 3 interview rounds."
        input_to_hr = mock_hr.answer_query.call_args[0][0]
        assert input_to_hr.user_role == "candidate"
        assert "prior_answers" in input_to_hr.context


class TestAnswerRecruiterChat:
    """Recruiter chat answers from stored results, never re-runs the pipeline."""

    _SAMPLE_STATE = {
        "resumes": [
            {"candidate_name": "Sarah Chen", "match_score": 85.0,
             "skills": ["Python", "React"]},
        ],
        "candidate_rankings": [
            {"candidate_name": "Sarah Chen", "match_score": 85.0, "rank": 1,
             "justification": "Strong match."},
        ],
        "interview_slots": [
            {"candidate_name": "Sarah Chen", "date": "2026-08-10",
             "time_start": "09:00", "time_end": "10:00",
             "interviewer": "Bob Tech Lead", "interview_type": "technical",
             "status": "confirmed"},
        ],
        "job_description": {"job_title": "Senior Developer"},
    }

    def test_intent_classification(self):
        assert _parse_recruiter_results_intent("screen resumes") == "screening"
        assert _parse_recruiter_results_intent("rank candidates") == "rankings"
        assert _parse_recruiter_results_intent("compare candidates") == "rankings"
        assert _parse_recruiter_results_intent("schedule interviews") == "scheduling"
        assert _parse_recruiter_results_intent("hiring insights") == "insights"
        assert _parse_recruiter_results_intent("What is the interview process?") is None
        assert _parse_recruiter_results_intent("best practices for scheduling") is None

    def test_screening_answers_from_stored_results(self):
        answer = answer_recruiter_chat("screen resumes", "sess", dict(self._SAMPLE_STATE))
        assert "Sarah Chen" in answer
        assert "85%" in answer
        assert "1 screened resume" in answer

    def test_rankings_answers_from_stored_results(self):
        answer = answer_recruiter_chat("rank candidates", "sess", dict(self._SAMPLE_STATE))
        assert "#1 Sarah Chen" in answer
        assert "85% match" in answer

    def test_scheduling_answers_from_stored_results(self):
        answer = answer_recruiter_chat("schedule interviews", "sess", dict(self._SAMPLE_STATE))
        assert "Sarah Chen" in answer
        assert "confirmed" in answer

    def test_insights_answers_from_stored_results(self):
        answer = answer_recruiter_chat("hiring insights", "sess", dict(self._SAMPLE_STATE))
        assert "Senior Developer" in answer
        assert "Sarah Chen" in answer

    def test_no_resumes_screened_message(self):
        answer = answer_recruiter_chat("screen resumes", "sess", {})
        assert "No resumes have been screened yet" in answer

    def test_no_rankings_message(self):
        answer = answer_recruiter_chat("rank candidates", "sess", {})
        assert "No candidates have been ranked yet" in answer

    def test_no_slots_message(self):
        answer = answer_recruiter_chat("schedule interviews", "sess", {})
        assert "No interviews have been scheduled yet" in answer

    def test_no_data_insights_message(self):
        answer = answer_recruiter_chat("hiring insights", "sess", {})
        assert "No hiring data yet" in answer

    @patch("utils.llm_factory.get_llm")
    @patch("agents.hr_assistant_agent.HRAssistantAgent")
    def test_process_question_uses_hr_knowledge_base(self, mock_hr_cls, mock_get_llm):
        mock_hr = MagicMock()
        mock_hr.answer_query.return_value = HRAssistantOutput(
            answer="Here are the recruiter guidelines.",
            sources=["recruiter_guidelines"],
            confidence=0.9,
            needs_escalation=False,
        )
        mock_hr_cls.return_value = mock_hr

        answer = answer_recruiter_chat(
            "What are the best practices for scheduling?",
            "sess",
            dict(self._SAMPLE_STATE),
        )

        assert answer == "Here are the recruiter guidelines."
        input_to_hr = mock_hr.answer_query.call_args[0][0]
        assert input_to_hr.user_role == "recruiter"
        # The workflow context is included for the recruiter HR answer.
        assert "workflow" in input_to_hr.context


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
        mock_matching_agent.rank_candidates_async.return_value = CandidateMatchingOutput(
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
        mock_matching_agent.rank_candidates_async.side_effect = [
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
        assert mock_matching_agent.rank_candidates_async.call_count == 2

        # The correction attempt received the reflection feedback.
        retry_call = mock_matching_agent.rank_candidates_async.call_args[0][0]
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
