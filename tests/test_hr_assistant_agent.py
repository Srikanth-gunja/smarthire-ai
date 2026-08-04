"""Tests for HRAssistantAgent — tests plumbing with mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.hr_assistant_agent import HRAnswer, HRAssistantAgent
from tools.candidate_database import CandidateDatabase
from utils.models import HRAssistantInput, HRAssistantOutput


@pytest.fixture
def mock_llm():
    """Create a mock ChatOllama LLM."""
    return MagicMock()


@pytest.fixture
def kb_path(tmp_path):
    """Create a temporary HR knowledge base file."""
    kb_content = """# HR Knowledge Base

## Recruitment Process

1. Application Submission
2. Resume Screening
3. Phone Screen
4. Technical Interview
5. Behavioral Interview
6. Final Decision
7. Offer Stage
8. Onboarding

## Timeline

Total process takes 4-6 weeks.
"""
    kb_file = tmp_path / "hr_kb.md"
    kb_file.write_text(kb_content, encoding="utf-8")
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

    def test_answer_query_returns_output(self, agent, mock_llm):
        """answer_query returns an HRAssistantOutput."""
        mock_result = HRAnswer(
            answer="The process has 8 stages.",
            confidence=0.9,
            needs_escalation=False,
            relevant_sections=["Recruitment Process"],
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result
        mock_structured.return_value = mock_result
        mock_llm.with_structured_output.return_value = mock_structured

        with patch("agents.hr_assistant_agent.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_cls.from_messages.return_value = mock_prompt

            result = agent.answer_query(HRAssistantInput(query="What are the hiring stages?"))

        assert isinstance(result, HRAssistantOutput)
        assert result.answer == "The process has 8 stages."
        assert result.needs_escalation is False

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
        agent = HRAssistantAgent(mock_llm, knowledge_base_path=str(tmp_path / "nonexistent.md"))
        kb = agent._load_knowledge_base()
        assert "No knowledge base available" in kb


class TestRetrievePolicyContext:
    """Tests for _retrieve_policy_context method."""

    def test_relevant_sections(self, agent):
        """Returns sections relevant to the query."""
        sections = agent._retrieve_policy_context("What is the recruitment process?")
        assert len(sections) > 0

    def test_irrelevant_query(self, agent):
        """Returns few/no sections for irrelevant queries."""
        sections = agent._retrieve_policy_context("xyz123")
        assert len(sections) == 0


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
