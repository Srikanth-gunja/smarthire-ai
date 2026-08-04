"""Tests for ResumeScreeningAgent — tests plumbing with mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.resume_screening_agent import ResumeScreeningAgent, ScreeningResult
from utils.models import ResumeScreeningInput


@pytest.fixture
def mock_llm():
    """Create a mock ChatOllama LLM."""
    return MagicMock()


@pytest.fixture
def agent(mock_llm):
    """Create a ResumeScreeningAgent with a mocked LLM."""
    return ResumeScreeningAgent(mock_llm)


@pytest.fixture
def sample_jd():
    """Return a sample parsed JD dict."""
    return {
        "job_title": "Senior Developer",
        "required_skills": ["Python", "React", "Docker"],
        "preferred_skills": ["Kubernetes", "AWS"],
        "min_experience_years": 5,
        "education_requirements": {"degree_level": "Bachelor"},
    }


@pytest.fixture
def sample_resume_text():
    """Return a sample resume text."""
    return "Jane Doe\nPython developer with 6 years experience.\nSkills: Python, React, Docker"


class TestResumeScreeningAgentInit:
    """Tests for ResumeScreeningAgent initialization."""

    def test_init_stores_llm(self, mock_llm):
        """Agent stores the LLM instance."""
        agent = ResumeScreeningAgent(mock_llm)
        assert agent.llm is mock_llm

    def test_init_creates_parser(self, mock_llm):
        """Agent creates a ResumeParser."""
        agent = ResumeScreeningAgent(mock_llm)
        assert agent.parser is not None


class TestScreenResume:
    """Tests for screen_resume method."""

    def test_screen_resume_returns_output(self, agent, mock_llm, sample_jd):
        """screen_resume returns a ResumeScreeningOutput."""
        from tools.resume_parser import ExtractedResume

        mock_parser_result = ExtractedResume(
            candidate_name="Jane Doe",
            skills=["Python", "React"],
            experience_years=6.0,
            education=[],
            certifications=[],
            past_roles=[],
        )
        mock_screening_result = ScreeningResult(
            candidate_name="Jane Doe",
            skills=["Python", "React", "Docker"],
            experience_years=6.0,
            education=[],
            summary="Strong Python developer.",
            match_score=75.0,
        )

        call_count = 0

        def mock_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_parser_result
            return mock_screening_result

        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = mock_side_effect
        mock_structured.side_effect = mock_side_effect
        mock_llm.with_structured_output.return_value = mock_structured

        with patch("agents.resume_screening_agent.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = mock_side_effect
            mock_chain.side_effect = mock_side_effect
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_cls.from_messages.return_value = mock_prompt

            with patch("tools.resume_parser.ChatPromptTemplate"):
                input_data = ResumeScreeningInput(
                    resume_text="Jane Doe\nPython developer",
                    job_description=sample_jd,
                )
                result = agent.screen_resume(input_data)

        assert result.candidate_name == "Jane Doe"
        assert result.match_score == 75.0
        assert isinstance(result.skills, list)


class TestCalculateMatchScore:
    """Tests for _calculate_match_score method."""

    def test_perfect_match(self, agent, sample_jd):
        """Score is 100 when all required skills match."""
        score = agent._calculate_match_score(
            ["Python", "React", "Docker"], sample_jd
        )
        assert score == 100.0

    def test_partial_match(self, agent, sample_jd):
        """Score reflects partial skill match."""
        score = agent._calculate_match_score(["Python", "React"], sample_jd)
        assert 0 < score < 100

    def test_no_match(self, agent, sample_jd):
        """Score is 0 when no required skills match."""
        score = agent._calculate_match_score(["Java", "Spring"], sample_jd)
        assert score == 0.0

    def test_empty_required_skills(self, agent):
        """Score is 50 when JD has no required skills."""
        score = agent._calculate_match_score(["Python"], {"required_skills": []})
        assert score == 50.0

    def test_case_insensitive(self, agent, sample_jd):
        """Skill matching is case-insensitive."""
        score = agent._calculate_match_score(
            ["python", "REACT", "Docker"], sample_jd
        )
        assert score == 100.0


class TestScreeningResultModel:
    """Tests for the ScreeningResult Pydantic model."""

    def test_model_creation(self):
        """ScreeningResult can be created with valid data."""
        model = ScreeningResult(
            candidate_name="Test",
            skills=["Python"],
            experience_years=3.0,
            education=[],
            summary="Test summary",
            match_score=80.0,
        )
        assert model.candidate_name == "Test"
        assert model.match_score == 80.0

    def test_match_score_bounds(self):
        """Match score must be between 0 and 100."""
        ScreeningResult(
            candidate_name="Test", skills=[], experience_years=0,
            education=[], summary="", match_score=0,
        )
        ScreeningResult(
            candidate_name="Test", skills=[], experience_years=0,
            education=[], summary="", match_score=100,
        )
        with pytest.raises(ValueError):
            ScreeningResult(
                candidate_name="Test", skills=[], experience_years=0,
                education=[], summary="", match_score=101,
            )
