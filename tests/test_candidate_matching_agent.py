"""Tests for CandidateMatchingAgent — tests plumbing with mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.candidate_matching_agent import (
    CandidateMatchingAgent,
    LLMMatchResult,
    LLMRankedCandidate,
)
from utils.models import (
    CandidateMatchingInput,
    CandidateMatchingOutput,
    RankedCandidate,
)


@pytest.fixture
def mock_llm():
    """Create a mock ChatOllama LLM."""
    return MagicMock()


@pytest.fixture
def agent(mock_llm):
    """Create a CandidateMatchingAgent with a mocked LLM."""
    return CandidateMatchingAgent(mock_llm)


@pytest.fixture
def sample_resumes():
    """Return sample screened resume data."""
    return [
        {"candidate_name": "Alice", "skills": ["Python", "React", "Docker"], "experience_years": 7, "match_score": 85},
        {"candidate_name": "Bob", "skills": ["JavaScript", "Node.js"], "experience_years": 4, "match_score": 60},
    ]


@pytest.fixture
def sample_jd():
    """Return a sample parsed JD dict."""
    return {
        "job_title": "Senior Developer",
        "required_skills": ["Python", "React", "Docker"],
        "preferred_skills": ["Kubernetes"],
        "min_experience_years": 5,
    }


class TestCandidateMatchingAgentInit:
    """Tests for CandidateMatchingAgent initialization."""

    def test_init_stores_llm(self, mock_llm):
        """Agent stores the LLM instance."""
        agent = CandidateMatchingAgent(mock_llm)
        assert agent.llm is mock_llm


class TestRankCandidates:
    """Tests for rank_candidates method."""

    def test_rank_candidates_returns_output(self, agent, mock_llm, sample_resumes, sample_jd):
        """rank_candidates returns a CandidateMatchingOutput."""
        mock_result = LLMMatchResult(
            ranked_candidates=[
                LLMRankedCandidate(
                    candidate_name="Alice", match_score=85.0,
                    skills_match=["Python", "React", "Docker"], skills_gap=[],
                    experience_match=True, justification="Strong match.",
                ),
                LLMRankedCandidate(
                    candidate_name="Bob", match_score=60.0,
                    skills_match=[], skills_gap=["Python", "React", "Docker"],
                    experience_match=False, justification="Missing skills.",
                ),
            ],
            summary="Alice is the stronger candidate.",
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_structured.return_value = mock_result
        mock_llm.with_structured_output.return_value = mock_structured

        with patch("agents.candidate_matching_agent.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_cls.from_messages.return_value = mock_prompt

            input_data = CandidateMatchingInput(resumes=sample_resumes, job_description=sample_jd)
            result = agent.rank_candidates(input_data)

        assert isinstance(result, CandidateMatchingOutput)
        assert len(result.ranked_candidates) == 2
        assert result.ranked_candidates[0].rank == 1
        assert result.ranked_candidates[1].rank == 2
        assert result.total_candidates_evaluated == 2

    def test_rank_empty_resumes(self, agent):
        """rank_candidates handles empty resume list."""
        input_data = CandidateMatchingInput(resumes=[], job_description={"required_skills": ["Python"]})
        result = agent.rank_candidates(input_data)
        assert result.total_candidates_evaluated == 0
        assert result.ranked_candidates == []


class TestComputeSkillsMatch:
    """Tests for _compute_skills_match method."""

    def test_perfect_match(self, agent):
        """All skills match when resume has all JD skills."""
        matched, missing = agent._compute_skills_match(
            ["Python", "React", "Docker"], ["Python", "React", "Docker"],
        )
        assert matched == ["Python", "React", "Docker"]
        assert missing == []

    def test_partial_match(self, agent):
        """Partial skills match correctly identified."""
        matched, missing = agent._compute_skills_match(
            ["Python", "React"], ["Python", "React", "Docker"],
        )
        assert "Python" in matched
        assert "Docker" in missing

    def test_no_match(self, agent):
        """No skills match when resume has different skills."""
        matched, missing = agent._compute_skills_match(
            ["Java", "Spring"], ["Python", "React"],
        )
        assert matched == []
        assert missing == ["Python", "React"]

    def test_case_insensitive(self, agent):
        """Skill matching is case-insensitive."""
        matched, missing = agent._compute_skills_match(
            ["python", "REACT"], ["Python", "React", "Docker"],
        )
        assert len(matched) == 2
        assert len(missing) == 1


class TestComputeCompositeScore:
    """Tests for _compute_composite_score method."""

    def test_perfect_match_score(self, agent, sample_jd):
        """Score is high when all skills match and experience is met."""
        score = agent._compute_composite_score(
            skills_match=["Python", "React", "Docker"], skills_gap=[],
            experience_match=True, jd=sample_jd,
        )
        assert score >= 80

    def test_no_match_score(self, agent, sample_jd):
        """Score is low when no skills match and experience not met."""
        score = agent._compute_composite_score(
            skills_match=[], skills_gap=["Python", "React", "Docker"],
            experience_match=False, jd=sample_jd,
        )
        assert score < 40

    def test_skills_gap_penalty(self, agent, sample_jd):
        """Score increases slightly when no skills gap exists."""
        score_with_gap = agent._compute_composite_score(["Python"], ["React"], True, sample_jd)
        score_without_gap = agent._compute_composite_score(["Python", "React"], [], True, sample_jd)
        assert score_without_gap > score_with_gap


class TestGenerateJustification:
    """Tests for _generate_justification method."""

    def test_justification_with_matches(self, agent):
        """Justification includes matched skills."""
        justification = agent._generate_justification("Alice", ["Python", "React"], [], True, 85.0)
        assert "Alice" in justification
        assert "Python" in justification
        assert "85" in justification

    def test_justification_with_gaps(self, agent):
        """Justification mentions missing skills."""
        justification = agent._generate_justification("Bob", [], ["Python", "Docker"], False, 40.0)
        assert "Missing" in justification
        assert "Python" in justification


class TestRankedCandidateModel:
    """Tests for the RankedCandidate Pydantic model."""

    def test_model_creation(self):
        """RankedCandidate can be created with valid data."""
        model = RankedCandidate(
            candidate_name="Test", match_score=75.0,
            skills_match=["Python"], skills_gap=["React"],
            experience_match=True, justification="Good match.", rank=1,
        )
        assert model.rank == 1
        assert model.match_score == 75.0

    def test_rank_must_be_positive(self):
        """Rank must be >= 1."""
        with pytest.raises(ValueError):
            RankedCandidate(
                candidate_name="Test", match_score=50,
                skills_match=[], skills_gap=[], experience_match=False,
                justification="", rank=0,
            )
