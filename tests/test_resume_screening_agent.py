"""Tests for ResumeScreeningAgent — tests plumbing with mocked LLM."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from agents.resume_screening_agent import ResumeScreeningAgent, ScreeningResult
from utils.models import ResumeScreeningInput, ResumeScreeningOutput


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
        # The persisted score uses grounded parser facts, not the screening
        # model's potentially incomplete or fabricated score.
        assert result.match_score == pytest.approx(66.7)
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


class TestScreenBatchAsync:
    """Tests for the parallel batch fan-out (asyncio + semaphore)."""

    @pytest.fixture
    def sample_output(self):
        """A fixed successful screening output for batch tests."""
        return ResumeScreeningOutput(
            candidate_name="Jane Doe",
            skills=["Python", "React"],
            experience_years=6.0,
            education=[],
            summary="Strong Python developer.",
            match_score=80.0,
            extracted_fields={"certifications": [], "past_roles": []},
        )

    @pytest.fixture
    def parsed(self):
        """A fixed parser result for batch tests."""
        return {
            "candidate_name": "Jane Doe",
            "skills": ["Python", "React"],
            "experience_years": 6.0,
            "education": [],
            "certifications": [],
            "past_roles": [],
        }

    def _stub_parse_and_screen(self, agent, parsed, screen_impl):
        """Replace parser + scoring with pure-async stand-ins."""

        async def fake_parse(resume_text):
            return parsed

        agent.parser.parse_text_async = fake_parse
        agent.screen_parsed_async = screen_impl

    def test_fans_out_and_returns_results_in_order(self, agent, sample_jd, sample_output, parsed):
        """Empty-text resumes fail in isolation; others return outputs."""
        async def fake_screen(parsed, resume_text, jd_data, resume_filename=None):
            return sample_output

        self._stub_parse_and_screen(agent, parsed, fake_screen)

        resumes = [
            {"name": "a.txt", "text": "resume a"},
            {"name": "b.txt", "text": "resume b"},
            {"name": "c.txt", "text": ""},  # empty -> isolated failure
        ]
        results = asyncio.run(agent.screen_batch_async(resumes, sample_jd))

        assert isinstance(results[0], ResumeScreeningOutput)
        assert isinstance(results[1], ResumeScreeningOutput)
        assert isinstance(results[2], Exception)
        assert "no extractable text" in str(results[2]).lower()

    def test_failure_does_not_cancel_in_flight_tasks(
        self, agent, sample_jd, sample_output, parsed
    ):
        """A failing resume must not cancel the other in-flight screenings."""
        async def fake_screen(parsed, resume_text, jd_data, resume_filename=None):
            if resume_filename == "boom.txt":
                raise RuntimeError("Transient 503 from provider")
            return sample_output

        self._stub_parse_and_screen(agent, parsed, fake_screen)

        resumes = [
            {"name": "boom.txt", "text": "will fail"},
            {"name": "ok.txt", "text": "fine"},
            {"name": "ok2.txt", "text": "fine too"},
        ]
        results = asyncio.run(agent.screen_batch_async(resumes, sample_jd))

        assert isinstance(results[0], Exception)
        assert isinstance(results[1], ResumeScreeningOutput)
        assert isinstance(results[2], ResumeScreeningOutput)

    def test_reports_progress_per_resume(self, agent, sample_jd, sample_output, parsed):
        """on_progress fires as each resume finishes with its result/error."""
        async def fake_screen(parsed, resume_text, jd_data, resume_filename=None):
            if resume_filename == "bad.txt":
                raise ValueError("bad parse")
            return sample_output

        self._stub_parse_and_screen(agent, parsed, fake_screen)

        progress = []
        resumes = [
            {"name": "a.txt", "text": "resume a"},
            {"name": "bad.txt", "text": "bad"},
            {"name": "c.txt", "text": "resume c"},
        ]
        asyncio.run(
            agent.screen_batch_async(
                resumes,
                sample_jd,
                on_progress=lambda done, total, name, result, error: progress.append(
                    (done, total, name, result, error)
                ),
            )
        )

        assert len(progress) == 3
        assert progress[-1][:2] == (3, 3)
        by_name = {
            name: (result, error)
            for done, total, name, result, error in progress
        }
        assert isinstance(by_name["a.txt"][0], ResumeScreeningOutput)
        assert by_name["a.txt"][1] is None
        assert by_name["bad.txt"][0] is None
        assert isinstance(by_name["bad.txt"][1], ValueError)

    def test_caps_concurrency_with_semaphore(self, agent, sample_jd, sample_output, parsed):
        """Concurrent screening is capped by SCREENING_CONCURRENCY."""
        active = 0
        max_active = 0

        async def fake_screen(parsed, resume_text, jd_data, resume_filename=None):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return sample_output

        self._stub_parse_and_screen(agent, parsed, fake_screen)

        resumes = [
            {"name": f"r{i}.txt", "text": f"resume {i}"}
            for i in range(10)
        ]
        os.environ["SCREENING_CONCURRENCY"] = "3"
        try:
            asyncio.run(agent.screen_batch_async(resumes, sample_jd))
        finally:
            os.environ.pop("SCREENING_CONCURRENCY", None)

        assert 1 <= max_active <= 3

    def test_awaits_shared_jd_analysis_task(
        self, agent, sample_jd, sample_output, parsed
    ):
        """Resumes are scored against the JD from the shared analysis task."""
        seen_jd = []

        async def fake_screen(parsed, resume_text, jd_data, resume_filename=None):
            seen_jd.append(jd_data)
            return sample_output

        self._stub_parse_and_screen(agent, parsed, fake_screen)

        async def analyze_jd():
            await asyncio.sleep(0.01)
            return {**sample_jd, "required_skills": ["Python", "Docker"]}

        async def run():
            jd_task = asyncio.create_task(analyze_jd())
            return await agent.screen_batch_async(
                [{"name": "a.txt", "text": "resume a"}],
                sample_jd,
                jd_analysis_task=jd_task,
            )

        results = asyncio.run(run())

        assert isinstance(results[0], ResumeScreeningOutput)
        assert seen_jd and seen_jd[0]["required_skills"] == ["Python", "Docker"]

    def test_jd_task_failure_does_not_fail_batch(
        self, agent, sample_jd, sample_output, parsed
    ):
        """A failing JD analysis task falls back to the raw JD, batch survives."""
        seen_jd = []

        async def fake_screen(parsed, resume_text, jd_data, resume_filename=None):
            seen_jd.append(jd_data)
            return sample_output

        self._stub_parse_and_screen(agent, parsed, fake_screen)

        async def analyze_jd():
            await asyncio.sleep(0.01)
            raise RuntimeError("JD LLM unavailable")

        async def run():
            jd_task = asyncio.create_task(analyze_jd())
            return await agent.screen_batch_async(
                [{"name": "a.txt", "text": "resume a"}],
                sample_jd,
                jd_analysis_task=jd_task,
            )

        results = asyncio.run(run())

        assert isinstance(results[0], ResumeScreeningOutput)
        assert seen_jd and seen_jd[0]["required_skills"] == sample_jd["required_skills"]
