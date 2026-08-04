"""Tests for JDAnalyzer — tests plumbing and data shaping with mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.jd_analyzer import ExtractedJD, JDAnalyzer


@pytest.fixture
def mock_llm():
    """Create a mock ChatOllama LLM."""
    return MagicMock()


@pytest.fixture
def analyzer(mock_llm):
    """Create a JDAnalyzer with a mocked LLM."""
    return JDAnalyzer(mock_llm)


@pytest.fixture
def sample_jd_text():
    """Return a sample JD text."""
    return "Senior Developer position. Required: Python, React, Docker."


class TestJDAnalyzerInit:
    """Tests for JDAnalyzer initialization."""

    def test_init_stores_llm(self, mock_llm):
        """Analyzer stores the LLM instance."""
        analyzer = JDAnalyzer(mock_llm)
        assert analyzer.llm is mock_llm


class TestAnalyze:
    """Tests for analyze method."""

    def test_analyze_returns_dict(self, analyzer, mock_llm, sample_jd_text):
        """analyze returns a dict with expected keys."""
        mock_result = ExtractedJD(
            job_title="Senior Developer",
            required_skills=["Python", "React", "Docker"],
            preferred_skills=["Kubernetes"],
            min_experience_years=5,
            max_experience_years=None,
            education_requirements={"degree_level": "Bachelor"},
            summary="Senior developer role.",
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_structured.return_value = mock_result
        mock_llm.with_structured_output.return_value = mock_structured

        with patch("tools.jd_analyzer.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_cls.from_messages.return_value = mock_prompt

            result = analyzer.analyze(sample_jd_text)

        assert isinstance(result, dict)
        assert "job_title" in result
        assert "required_skills" in result
        assert "preferred_skills" in result
        assert "min_experience_years" in result

    def test_analyze_extracts_skills(self, analyzer, mock_llm, sample_jd_text):
        """analyze extracts required and preferred skills."""
        mock_result = ExtractedJD(
            job_title="Senior Developer",
            required_skills=["Python", "React"],
            preferred_skills=["Kubernetes"],
            min_experience_years=5,
            max_experience_years=None,
            education_requirements={"degree_level": "Bachelor"},
            summary="Role summary.",
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_structured.return_value = mock_result
        mock_llm.with_structured_output.return_value = mock_structured

        with patch("tools.jd_analyzer.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_cls.from_messages.return_value = mock_prompt

            result = analyzer.analyze(sample_jd_text)

        assert "Python" in result["required_skills"]
        assert "Kubernetes" in result["preferred_skills"]


class TestAnalyzeFile:
    """Tests for analyze_file method."""

    def test_analyze_file_reads_txt(self, analyzer, mock_llm, tmp_path):
        """analyze_file reads and analyzes a .txt file."""
        jd_file = tmp_path / "test_jd.txt"
        jd_file.write_text("Senior Developer Required", encoding="utf-8")

        mock_result = ExtractedJD(
            job_title="Senior Developer",
            required_skills=["Python"],
            preferred_skills=[],
            min_experience_years=5,
            max_experience_years=None,
            education_requirements={"degree_level": "Bachelor"},
            summary="Test.",
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_structured.return_value = mock_result
        mock_llm.with_structured_output.return_value = mock_structured

        with patch("tools.jd_analyzer.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_cls.from_messages.return_value = mock_prompt

            result = analyzer.analyze_file(str(jd_file))

        assert result["job_title"] == "Senior Developer"
        assert result["raw_text"] == "Senior Developer Required"


class TestExtractedJDModel:
    """Tests for the ExtractedJD Pydantic model."""

    def test_model_creation(self):
        """ExtractedJD can be created with valid data."""
        model = ExtractedJD(
            job_title="Developer",
            required_skills=["Python"],
            preferred_skills=[],
            min_experience_years=3,
            max_experience_years=5,
            education_requirements={"degree_level": "Bachelor"},
            summary="Test role.",
        )
        assert model.job_title == "Developer"
        assert model.min_experience_years == 3

    def test_model_optional_max_experience(self):
        """ExtractedJD allows None for max_experience_years."""
        model = ExtractedJD(
            job_title="Developer",
            required_skills=[],
            preferred_skills=[],
            min_experience_years=0,
            max_experience_years=None,
            education_requirements={},
            summary=".",
        )
        assert model.max_experience_years is None
