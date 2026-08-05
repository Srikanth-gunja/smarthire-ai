"""Tests for HRAssistantAgent — role-aware behaviour with a JSON knowledge base."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.hr_assistant_agent import (
    GREETING_PATTERNS,
    UNAVAILABLE_ANSWER,
    HRAssistantAgent,
    HRAnswer,
)
from tools.candidate_database import CandidateDatabase
from utils.models import HRAssistantInput, HRAssistantOutput


@pytest.fixture
def mock_llm():
    """Create a mock ChatOllama LLM."""
    return MagicMock()


@pytest.fixture
def kb_path(tmp_path):
    """Create a temporary JSON HR knowledge base file."""
    kb = {
        "topics": {
            "recruitment_process": {
                "keywords": ["hiring process", "hiring stages", "stages of hiring", "recruitment process"],
                "content": (
                    "The recruitment process has 8 stages: Application Submission, "
                    "Resume Screening, Phone Screen, Technical Interview, Behavioral "
                    "Interview, Final Decision, Offer Stage, Onboarding. The total "
                    "timeline is 4-6 weeks."
                ),
            },
            "interview_rounds": {
                "keywords": ["how many rounds", "interview rounds", "number of interviews"],
                "content": "Most roles have 3 rounds: phone screen, technical, behavioral.",
            },
            "recruiter_guidelines": {
                "keywords": ["screen resumes", "rank candidates", "schedule interviews", "scheduling", "best practices"],
                "content": "Schedule phone screens within 1 week of shortlisting.",
            },
        }
    }
    kb_file = tmp_path / "hr_kb.json"
    kb_file.write_text(json.dumps(kb), encoding="utf-8")
    return str(kb_file)


@pytest.fixture
def db(tmp_path):
    """Create a test CandidateDatabase."""
    csv_content = """candidate_id,name,email,phone,skills,experience_years,education,status,application_date
C001,Alice Smith,alice@test.com,(555) 111-1111,"Python,React",7.0,BS CS,shortlisted,2025-01-15
"""
    csv_path = tmp_path / "test_candidates.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    return CandidateDatabase(csv_path=str(csv_path))


@pytest.fixture
def agent(mock_llm, kb_path, db):
    """Create an HRAssistantAgent with mocked LLM and test KB."""
    return HRAssistantAgent(mock_llm, candidate_db=db, knowledge_base_path=kb_path)


class TestHRAssistantAgentInit:
    """Tests for HRAssistantAgent initialization."""

    def test_init_stores_llm(self, mock_llm, kb_path):
        """Agent stores the LLM instance."""
        agent = HRAssistantAgent(mock_llm, knowledge_base_path=kb_path)
        assert agent.llm is mock_llm

    def test_init_loads_kb(self, agent):
        """Agent loads the knowledge base."""
        kb = agent._load_knowledge_base()
        assert "Recruitment Process" in kb


class TestAnswerQuery:
    """Tests for answer_query method."""

    def _mock_structured(self, mock_llm, answer="The process has 8 stages."):
        from langchain_core.runnables import RunnableLambda

        mock_result = HRAnswer(
            answer=answer,
            confidence=0.9,
            needs_escalation=False,
            relevant_sections=["recruitment_process"],
        )
        # A RunnableLambda stands in for the structured-output LLM so the real
        # ChatPromptTemplate chain executes end to end in the test.
        mock_llm.with_structured_output.return_value = RunnableLambda(
            lambda _: mock_result
        )
        return mock_result

    def test_answer_query_returns_output(self, agent, mock_llm):
        """answer_query returns an HRAssistantOutput."""
        self._mock_structured(mock_llm)

        result = agent.answer_query(
            HRAssistantInput(query="What are the hiring stages?", user_role="candidate")
        )

        assert isinstance(result, HRAssistantOutput)
        assert result.answer == "The process has 8 stages."
        assert result.needs_escalation is False

    def test_answer_query_recruiter_role(self, agent, mock_llm):
        """Recruiter role questions are answered too."""
        self._mock_structured(mock_llm, answer="Here are the recruiter guidelines.")

        result = agent.answer_query(
            HRAssistantInput(
                query="What are the best practices for scheduling?", user_role="recruiter"
            )
        )

        assert result.answer == "Here are the recruiter guidelines."

    def test_unavailable_when_no_kb_match(self, agent):
        """Questions with no KB match return a polite unavailable answer."""
        result = agent.answer_query(
            HRAssistantInput(query="xyz123 unknown topic", user_role="candidate")
        )
        assert result.answer == UNAVAILABLE_ANSWER
        assert result.confidence == 0.0

    def test_candidate_never_gets_recruiter_topic(self, agent):
        """A candidate asking recruiter-style questions gets an unavailable answer."""
        result = agent.answer_query(
            HRAssistantInput(query="Screen resumes", user_role="candidate")
        )
        assert result.answer == UNAVAILABLE_ANSWER

    def test_greeting_returns_friendly_message(self, agent):
        """Greetings get a friendly canned response, not an LLM call."""
        result = agent.answer_query(
            HRAssistantInput(query="hello", user_role="candidate")
        )
        assert "HR Assistant" in result.answer

    def test_escalation_for_legal_query(self, agent):
        """Legal questions trigger escalation."""
        result = agent.answer_query(HRAssistantInput(query="I want to sue the company"))
        assert result.needs_escalation is True
        assert result.confidence == 0.0

    def test_escalation_for_discrimination(self, agent):
        """Discrimination questions trigger escalation."""
        result = agent.answer_query(HRAssistantInput(query="I experienced discrimination"))
        assert result.needs_escalation is True

    def test_escalation_for_accommodation(self, agent):
        """Accommodation requests trigger escalation."""
        result = agent.answer_query(HRAssistantInput(query="I need a disability accommodation"))
        assert result.needs_escalation is True


class TestLoadKnowledgeBase:
    """Tests for _load_knowledge_base method."""

    def test_loads_content(self, agent):
        """Loads the knowledge base content."""
        kb = agent._load_knowledge_base()
        assert len(kb) > 0
        assert "Recruitment Process" in kb

    def test_caches_content(self, agent):
        """Knowledge base is cached after first load."""
        kb1 = agent._load_knowledge_base()
        kb2 = agent._load_knowledge_base()
        assert kb1 is kb2

    def test_missing_kb(self, mock_llm, tmp_path):
        """Handles missing knowledge base file."""
        agent = HRAssistantAgent(mock_llm, knowledge_base_path=str(tmp_path / "nonexistent.json"))
        kb = agent._load_knowledge_base()
        assert "No knowledge base available" in kb


class TestRetrieveRelevantSections:
    """Tests for _retrieve_relevant_sections method."""

    def test_relevant_sections(self, agent):
        """Returns sections relevant to the query."""
        sections = agent._retrieve_relevant_sections(
            "What is the recruitment process?", "candidate"
        )
        assert len(sections) > 0
        assert sections[0]["topic"] == "recruitment_process"

    def test_irrelevant_query(self, agent):
        """Returns no sections for irrelevant queries."""
        sections = agent._retrieve_relevant_sections("xyz123", "candidate")
        assert len(sections) == 0

    def test_recruiter_topic_excluded_for_candidates(self, agent):
        """Recruiter-only topics never match for candidates."""
        sections = agent._retrieve_relevant_sections("screen resumes", "candidate")
        assert all(s["topic"] != "recruiter_guidelines" for s in sections)


class TestShouldEscalate:
    """Tests for _should_escalate method."""

    def test_legal_escalation(self, agent):
        """Legal questions need escalation."""
        assert agent._should_escalate("I need legal advice") is True

    def test_normal_question_no_escalation(self, agent):
        """Normal questions don't need escalation."""
        assert agent._should_escalate("What are the hiring stages?") is False

    def test_accommodation_escalation(self, agent):
        """Accommodation requests need escalation."""
        assert agent._should_escalate("I need a religious accommodation") is True


class TestGreetingPattern:
    """Tests for greeting detection."""

    def test_matches_greetings(self):
        assert GREETING_PATTERNS.match("hi")
        assert GREETING_PATTERNS.match("Hello!")
        assert GREETING_PATTERNS.match("good morning")

    def test_not_a_greeting(self):
        assert not GREETING_PATTERNS.match("How many interview rounds are there?")


class TestLookupCandidate:
    """Tests for lookup_candidate method."""

    def test_lookup_existing(self, agent):
        """Returns candidate data for existing ID."""
        result = agent.lookup_candidate("C001")
        assert result is not None
        assert result["name"] == "Alice Smith"

    def test_lookup_nonexistent(self, agent):
        """Returns None for non-existent ID."""
        result = agent.lookup_candidate("C999")
        assert result is None


class TestSearchCandidates:
    """Tests for search_candidates method."""

    def test_search_by_name(self, agent):
        """Search by name returns matching candidates."""
        results = agent.search_candidates("Alice")
        assert len(results) == 1


class TestHRAnswerModel:
    """Tests for the HRAnswer Pydantic model."""

    def test_model_creation(self):
        """HRAnswer can be created with valid data."""
        model = HRAnswer(answer="Test answer", confidence=0.8, needs_escalation=False)
        assert model.answer == "Test answer"
        assert model.confidence == 0.8

    def test_model_defaults(self):
        """HRAnswer has correct defaults."""
        model = HRAnswer(answer="Test", confidence=0.5, needs_escalation=False)
        assert model.relevant_sections == []


@patch("agents.hr_assistant_agent.HRAssistantAgent._load_kb_data", return_value={})
class TestNoKnowledgeBase:
    """Behaviour when the knowledge base is missing."""

    def test_unavailable_answer(self, _mock, mock_llm):
        """Empty KB always yields the unavailable answer."""
        agent = HRAssistantAgent(mock_llm)
        result = agent.answer_query(
            HRAssistantInput(query="What is the hiring process?", user_role="candidate")
        )
        assert result.answer == UNAVAILABLE_ANSWER
