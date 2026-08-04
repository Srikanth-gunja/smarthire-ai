"""Tests for ConversationMemory — session persistence and context summary.

Tests cover:
  - Session creation, retrieval, and clearing
  - Conversation history append and retrieval with limits
  - Job description storage and retrieval
  - Shortlisted candidate deduplication
  - Interview preferences merge
  - Context summary generation (bounded prompt injection)
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from memory.conversation_memory import ConversationMemory, SessionData

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def memory(tmp_path):
    """Fresh ConversationMemory instance backed by an isolated temp DB."""
    return ConversationMemory(db_path=tmp_path / "test_memory.db")


@pytest.fixture
def session(memory):
    """A memory instance with one pre-created session."""
    sid = memory.create_session()
    return memory, sid


# ── Session lifecycle ─────────────────────────────────────────────────


class TestSessionLifecycle:
    """Tests for create_session, get_session, clear_session."""

    def test_create_session_returns_id(self, memory):
        """create_session returns a non-empty string id."""
        sid = memory.create_session()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_create_multiple_sessions_unique_ids(self, memory):
        """Each call to create_session returns a unique id."""
        s1 = memory.create_session()
        s2 = memory.create_session()
        assert s1 != s2

    def test_get_session_returns_existing(self, memory):
        """get_session returns data for an existing session."""
        sid = memory.create_session()
        data = memory.get_session(sid)
        assert isinstance(data, SessionData)
        assert data.conversation_history == []

    def test_get_session_creates_if_missing(self, memory):
        """get_session creates a new session if id doesn't exist."""
        data = memory.get_session("nonexistent")
        assert isinstance(data, SessionData)

    def test_clear_session_removes_data(self, memory):
        """clear_session removes the session from the store."""
        sid = memory.create_session()
        memory.append_turn(sid, HumanMessage(content="Hello"))
        memory.clear_session(sid)
        data = memory.get_session(sid)
        assert data.conversation_history == []

    def test_clear_nonexistent_session_no_error(self, memory):
        """clear_session on unknown id does not raise."""
        memory.clear_session("nope")  # should not raise

    def test_reset_session_keeps_identity_but_removes_history(self, memory):
        """reset_session retains the session row and clears its workspace."""
        sid = memory.create_session("recruiter")
        memory.append_turn(sid, HumanMessage(content="Hello"))
        memory.reset_session(sid, "recruiter")
        assert memory.db.session_exists(sid)
        assert memory.get_history(sid) == []


# ── Conversation history ──────────────────────────────────────────────


class TestConversationHistory:
    """Tests for append_turn and get_history."""

    def test_append_then_retrieve(self, session):
        """Messages appended are returned by get_history."""
        mem, sid = session
        msg = HumanMessage(content="Hi there")
        mem.append_turn(sid, msg)
        history = mem.get_history(sid)
        assert len(history) == 1
        assert history[0].content == "Hi there"

    def test_append_multiple_messages(self, session):
        """Multiple messages accumulate in order."""
        mem, sid = session
        mem.append_turn(sid, HumanMessage(content="Q1"))
        mem.append_turn(sid, AIMessage(content="A1"))
        mem.append_turn(sid, HumanMessage(content="Q2"))
        history = mem.get_history(sid)
        assert len(history) == 3
        assert history[0].content == "Q1"
        assert history[2].content == "Q2"

    def test_get_history_limit(self, session):
        """limit parameter returns only the last N messages."""
        mem, sid = session
        for i in range(10):
            mem.append_turn(sid, HumanMessage(content=f"msg-{i}"))
        history = mem.get_history(sid, limit=3)
        assert len(history) == 3
        assert history[0].content == "msg-7"
        assert history[2].content == "msg-9"

    def test_get_history_limit_larger_than_history(self, session):
        """limit larger than history returns all messages."""
        mem, sid = session
        mem.append_turn(sid, HumanMessage(content="only"))
        history = mem.get_history(sid, limit=100)
        assert len(history) == 1

    def test_get_history_empty_session(self, session):
        """Empty session returns empty list."""
        mem, sid = session
        assert mem.get_history(sid) == []

    def test_history_is_independent_copy(self, session):
        """Returned list is a copy, not a reference to internal store."""
        mem, sid = session
        mem.append_turn(sid, HumanMessage(content="original"))
        history = mem.get_history(sid)
        history.clear()
        # Internal store should be unaffected
        assert len(mem.get_history(sid)) == 1


# ── Job descriptions ──────────────────────────────────────────────────


class TestJobDescriptions:
    """Tests for store_job_description and get_job_descriptions."""

    def test_store_and_retrieve(self, session):
        """JD stored is returned by get_job_descriptions."""
        mem, sid = session
        jd = {"job_title": "Dev", "required_skills": ["Python"]}
        mem.store_job_description(sid, jd)
        jds = mem.get_job_descriptions(sid)
        assert len(jds) == 1
        assert jds[0]["job_title"] == "Dev"

    def test_multiple_jds(self, session):
        """Multiple JDs are stored in order."""
        mem, sid = session
        mem.store_job_description(sid, {"job_title": "Dev"})
        mem.store_job_description(sid, {"job_title": "PM"})
        jds = mem.get_job_descriptions(sid)
        assert len(jds) == 2
        assert jds[1]["job_title"] == "PM"

    def test_empty_session_returns_empty_list(self, session):
        """No JDs stored returns empty list."""
        mem, sid = session
        assert mem.get_job_descriptions(sid) == []


# ── Shortlisted candidates ────────────────────────────────────────────


class TestShortlistedCandidates:
    """Tests for add_shortlisted_candidate and get_shortlisted_candidates."""

    def test_add_and_retrieve(self, session):
        """Candidate added is returned."""
        mem, sid = session
        candidate = {"candidate_name": "Alice", "score": 90}
        mem.add_shortlisted_candidate(sid, candidate)
        candidates = mem.get_shortlisted_candidates(sid)
        assert len(candidates) == 1
        assert candidates[0]["candidate_name"] == "Alice"

    def test_deduplication_by_name(self, session):
        """Adding the same candidate name replaces the old entry."""
        mem, sid = session
        mem.add_shortlisted_candidate(
            sid, {"candidate_name": "Alice", "score": 80}
        )
        mem.add_shortlisted_candidate(
            sid, {"candidate_name": "Alice", "score": 95}
        )
        candidates = mem.get_shortlisted_candidates(sid)
        assert len(candidates) == 1
        assert candidates[0]["score"] == 95

    def test_different_names_not_deduped(self, session):
        """Different candidate names are kept separate."""
        mem, sid = session
        mem.add_shortlisted_candidate(sid, {"candidate_name": "Alice"})
        mem.add_shortlisted_candidate(sid, {"candidate_name": "Bob"})
        assert len(mem.get_shortlisted_candidates(sid)) == 2


# ── Interview preferences ─────────────────────────────────────────────


class TestInterviewPreferences:
    """Tests for update_interview_preferences and get_interview_preferences."""

    def test_set_and_get(self, session):
        """Preferences set are returned."""
        mem, sid = session
        mem.update_interview_preferences(sid, {"date": "2025-03-01"})
        prefs = mem.get_interview_preferences(sid)
        assert prefs["date"] == "2025-03-01"

    def test_merge_preferences(self, session):
        """New preferences merge with existing ones."""
        mem, sid = session
        mem.update_interview_preferences(sid, {"date": "2025-03-01"})
        mem.update_interview_preferences(sid, {"format": "video"})
        prefs = mem.get_interview_preferences(sid)
        assert prefs["date"] == "2025-03-01"
        assert prefs["format"] == "video"

    def test_overwrite_same_key(self, session):
        """Same key is overwritten with new value."""
        mem, sid = session
        mem.update_interview_preferences(sid, {"date": "2025-03-01"})
        mem.update_interview_preferences(sid, {"date": "2025-04-01"})
        assert mem.get_interview_preferences(sid)["date"] == "2025-04-01"

    def test_empty_session_returns_empty_dict(self, session):
        """No preferences returns empty dict."""
        mem, sid = session
        assert mem.get_interview_preferences(sid) == {}


# ── Context summary ───────────────────────────────────────────────────


class TestContextSummary:
    """Tests for get_context_summary."""

    def test_empty_session(self, session):
        """Empty session returns a placeholder message."""
        mem, sid = session
        summary = mem.get_context_summary(sid)
        assert "New session" in summary

    def test_includes_jd_title(self, session):
        """Summary includes the latest JD title."""
        mem, sid = session
        mem.store_job_description(
            sid, {"job_title": "Backend Engineer", "required_skills": ["Go"]}
        )
        summary = mem.get_context_summary(sid)
        assert "Backend Engineer" in summary
        assert "Go" in summary

    def test_includes_candidate_names(self, session):
        """Summary includes shortlisted candidate names."""
        mem, sid = session
        mem.add_shortlisted_candidate(sid, {"candidate_name": "Alice"})
        mem.add_shortlisted_candidate(sid, {"candidate_name": "Bob"})
        summary = mem.get_context_summary(sid)
        assert "Alice" in summary
        assert "Bob" in summary

    def test_includes_interview_preferences(self, session):
        """Summary includes interview preferences."""
        mem, sid = session
        mem.update_interview_preferences(sid, {"format": "video"})
        summary = mem.get_context_summary(sid)
        assert "video" in summary

    def test_includes_recent_messages(self, session):
        """Summary includes the most recent conversation messages."""
        mem, sid = session
        mem.append_turn(sid, HumanMessage(content="Hello"))
        mem.append_turn(sid, AIMessage(content="Hi!"))
        summary = mem.get_context_summary(sid)
        assert "User:" in summary
        assert "Assistant:" in summary
        assert "Hello" in summary

    def test_summary_bounded_by_context_window(self, session):
        """Summary only includes the last _CONTEXT_WINDOW_LIMIT messages."""
        mem, sid = session
        # Add 25 messages (limit is 20)
        for i in range(25):
            mem.append_turn(sid, HumanMessage(content=f"message-{i}"))
        summary = mem.get_context_summary(sid)
        # The earliest messages should not appear
        assert "message-0" not in summary
        assert "message-4" not in summary
        # The latest should
        assert "message-24" in summary

    def test_long_messages_truncated(self, session):
        """Very long messages are truncated in the summary."""
        mem, sid = session
        long_msg = "x" * 500
        mem.append_turn(sid, HumanMessage(content=long_msg))
        summary = mem.get_context_summary(sid)
        assert len(summary) < 500
        assert "…" in summary
