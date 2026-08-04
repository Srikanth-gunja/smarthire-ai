"""Tests for ResumeParser — tests plumbing and data shaping with mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.resume_parser import ExtractedResume, ResumeParser


@pytest.fixture
def mock_llm():
    """Create a mock ChatOllama LLM."""
    return MagicMock()


@pytest.fixture
def parser(mock_llm):
    """Create a ResumeParser with a mocked LLM."""
    return ResumeParser(mock_llm)


@pytest.fixture
def sample_resume_text():
    """Return a sample resume text."""
    return """
    JOHN DOE
    Email: john.doe@email.com | Phone: (555) 123-4567
    SUMMARY
    Software engineer with 5 years of experience in Python and React.
    SKILLS
    Python, React, JavaScript, Docker, AWS
    """


class TestResumeParserInit:
    """Tests for ResumeParser initialization."""

    def test_init_stores_llm(self, mock_llm):
        """Parser stores the LLM instance."""
        parser = ResumeParser(mock_llm)
        assert parser.llm is mock_llm


class TestParseText:
    """Tests for parse_text method."""

    def test_parse_text_returns_dict(self, parser, mock_llm, sample_resume_text):
        """parse_text returns a dict with expected keys."""
        mock_result = ExtractedResume(
            candidate_name="John Doe",
            skills=["Python", "React", "JavaScript"],
            experience_years=5.0,
            education=[{"degree": "BS Computer Science", "institution": "MIT", "year": "2018"}],
            certifications=[],
            past_roles=[],
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_structured.return_value = mock_result
        mock_llm.with_structured_output.return_value = mock_structured

        with patch("tools.resume_parser.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_cls.from_messages.return_value = mock_prompt

            result = parser.parse_text(sample_resume_text)

        assert isinstance(result, dict)
        assert "candidate_name" in result
        assert "skills" in result
        assert "experience_years" in result
        assert "education" in result

    def test_parse_text_extracts_name(self, parser, mock_llm, sample_resume_text):
        """parse_text extracts the candidate name."""
        mock_result = ExtractedResume(
            candidate_name="John Doe",
            skills=["Python"],
            experience_years=5.0,
            education=[],
            certifications=[],
            past_roles=[],
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_structured.return_value = mock_result
        mock_llm.with_structured_output.return_value = mock_structured

        with patch("tools.resume_parser.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_cls.from_messages.return_value = mock_prompt

            result = parser.parse_text(sample_resume_text)

        assert result["candidate_name"] == "John Doe"


class TestParseFile:
    """Tests for parse_file method."""

    def test_parse_file_txt(self, parser, mock_llm, tmp_path):
        """parse_file reads and parses a .txt file."""
        resume_file = tmp_path / "test_resume.txt"
        resume_file.write_text("Test Resume Content", encoding="utf-8")

        mock_result = ExtractedResume(
            candidate_name="Test Person",
            skills=["Testing"],
            experience_years=2.0,
            education=[],
            certifications=[],
            past_roles=[],
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_structured.return_value = mock_result
        mock_llm.with_structured_output.return_value = mock_structured

        with patch("tools.resume_parser.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_cls.from_messages.return_value = mock_prompt

            result = parser.parse_file(str(resume_file))

        assert result["candidate_name"] == "Test Person"
        assert result["raw_text"] == "Test Resume Content"

    def test_parse_file_unsupported_raises(self, parser, tmp_path):
        """parse_file raises ValueError for unsupported formats."""
        unsupported = tmp_path / "test.xyz"
        unsupported.write_text("content", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse_file(str(unsupported))


class TestExtractedResumeModel:
    """Tests for the ExtractedResume Pydantic model."""

    def test_model_creation(self):
        """ExtractedResume can be created with valid data."""
        model = ExtractedResume(
            candidate_name="Jane Smith",
            skills=["Python", "SQL"],
            experience_years=3.5,
            education=[{"degree": "BS CS", "institution": "Stanford", "year": "2020"}],
        )
        assert model.candidate_name == "Jane Smith"
        assert len(model.skills) == 2
        assert model.experience_years == 3.5

    def test_model_defaults(self):
        """ExtractedResume has correct defaults for optional fields."""
        model = ExtractedResume(
            candidate_name="Test",
            skills=[],
            experience_years=0,
            education=[],
        )
        assert model.certifications == []
        assert model.past_roles == []

    def test_model_dump(self):
        """ExtractedResume.model_dump returns a dict."""
        model = ExtractedResume(
            candidate_name="Test",
            skills=["Python"],
            experience_years=1.0,
            education=[],
        )
        dumped = model.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["candidate_name"] == "Test"
