"""Tests for ChatAudit — chat_messages persistence and session mirroring.

Tests cover:
  - User / assistant / agent turn persistence and ordering
  - display_only filtering (agent turns stay as an audit trail)
  - log_turn (user + intermediate agent notes + final answer)
  - clear / empty-session behaviour
  - session-id mirror file (save / load / restore / clear)
  - Database.get_sessions ordering (Past Sessions list)
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from db.database import Database
from memory.chat_audit import ChatAudit


@pytest.fixture
def audit(tmp_path):
    """Fresh ChatAudit backed by an isolated temp DB + session mirror file."""
    return ChatAudit(
        db_path=tmp_path / "chat_audit.db",
        session_file=tmp_path / "active_session.local",
    )


@pytest.fixture
def session(audit):
    """A ChatAudit with one valid session row already created."""
    sid = "session-a"
    audit.db.upsert_session(sid)
    return audit, sid


# ── Logging & loading ─────────────────────────────────────────────────


class TestLogAndLoad:
    """Tests for log_user / log_assistant / log_agent and load_messages."""

    def test_user_assistant_roundtrip(self, session):
        """User and assistant turns are persisted and reloaded in order."""
        audit, sid = session
        audit.log_user(sid, "Hello")
        audit.log_assistant(sid, "Hi!")
        messages = audit.load_messages(sid)
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi!"

    def test_agent_turn_logged_with_agent_name(self, session):
        """Agent turns get role 'agent:<name>' and the agent_name column."""
        audit, sid = session
        audit.log_agent(sid, "resume_screening", "[ResumeScreening] Screened Alice")
        messages = audit.load_messages(sid, display_only=False)
        assert len(messages) == 1
        assert messages[0]["role"] == "agent:resume_screening"
        assert messages[0]["agent_name"] == "resume_screening"

    def test_display_only_hides_agent_turns(self, session):
        """Default load hides agent turns but keeps user/assistant."""
        audit, sid = session
        audit.log_user(sid, "question")
        audit.log_agent(sid, "supervisor", "[Supervisor] Intent: hr_question")
        audit.log_assistant(sid, "answer")
        shown = audit.load_messages(sid)
        assert [m["role"] for m in shown] == ["user", "assistant"]

    def test_ordering_oldest_first(self, session):
        """Messages are returned oldest → newest."""
        audit, sid = session
        for i in range(5):
            audit.log_user(sid, f"msg-{i}")
        messages = audit.load_messages(sid)
        assert [m["content"] for m in messages] == [f"msg-{i}" for i in range(5)]

    def test_load_messages_empty_session(self, audit):
        """Unknown session returns an empty list."""
        assert audit.load_messages("does-not-exist") == []

    def test_clear_removes_messages(self, session):
        """clear removes every chat message for a session."""
        audit, sid = session
        audit.log_user(sid, "hello")
        audit.log_assistant(sid, "world")
        audit.clear(sid)
        assert audit.load_messages(sid) == []


# ── log_turn ──────────────────────────────────────────────────────────


class TestLogTurn:
    """Tests for the one-call turn logger used by the Chat UI."""

    def test_logs_user_agent_and_assistant(self, session):
        """A full turn persists user prompt, agent notes, and final answer."""
        audit, sid = session
        result = {
            "conversation_history": [
                AIMessage(
                    content="[Supervisor] Intent: hr_question. "
                    "Routing to: hr_assistant. Reasoning: question."
                ),
                AIMessage(content="[HRAssistant] The process has 5 stages."),
            ],
            "final_response": "The hiring process has 5 stages.",
        }
        audit.log_turn(
            sid, user_content="What are the stages?", result=result,
            prior_history_count=0,
        )
        all_messages = audit.load_messages(sid, display_only=False)
        assert [m["role"] for m in all_messages] == [
            "user",
            "agent:supervisor",
            "agent:hr_assistant",
            "assistant",
        ]
        assert all_messages[-1]["content"] == "The hiring process has 5 stages."

    def test_prior_history_count_skips_old_messages(self, session):
        """Only messages added after prior_history_count are logged."""
        audit, sid = session
        result = {
            "conversation_history": [
                AIMessage(content="[Supervisor] old turn"),
                AIMessage(content="[Supervisor] Intent: new turn"),
            ],
            "final_response": "new answer",
        }
        audit.log_turn(
            sid, user_content="q2", result=result, prior_history_count=1,
        )
        all_messages = audit.load_messages(sid, display_only=False)
        assert [m["content"] for m in all_messages] == [
            "q2",
            "[Supervisor] Intent: new turn",
            "new answer",
        ]

    def test_user_content_skipped_when_none(self, session):
        """Passing no user_content (e.g. Screen & Rank) skips the user turn."""
        audit, sid = session
        result = {
            "conversation_history": [
                AIMessage(content="[CandidateMatching] Ranked 2 candidates."),
            ],
            "final_response": "Top candidate: Alice (85%).",
        }
        audit.log_turn(sid, result=result, prior_history_count=0)
        all_messages = audit.load_messages(sid, display_only=False)
        assert [m["role"] for m in all_messages] == [
            "agent:candidate_matching",
            "assistant",
        ]

    def test_answer_fallback_when_no_result(self, session):
        """When no graph result is given, the assistant answer is logged."""
        audit, sid = session
        audit.log_turn(
            sid,
            user_content="Hello",
            answer="Ollama is not reachable.",
        )
        messages = audit.load_messages(sid)
        assert [m["content"] for m in messages] == [
            "Hello",
            "Ollama is not reachable.",
        ]


# ── Session mirror file ───────────────────────────────────────────────


class TestSessionMirror:
    """Tests for the session-id mirror file used across refreshes."""

    def test_save_and_load(self, audit):
        """save_session_id writes a file that load_session_id reads back."""
        audit.save_session_id("abc123")
        assert audit.load_session_id() == "abc123"

    def test_load_missing_file_returns_none(self, audit):
        """No mirror file means no stored session id."""
        assert audit.load_session_id() is None

    def test_clear_session_id(self, audit):
        """clear_session_id removes the mirror file."""
        audit.save_session_id("abc123")
        audit.clear_session_id()
        assert audit.load_session_id() is None

    def test_restore_returns_existing_session(self, tmp_path):
        """restore_session_id returns the id when the session still exists."""
        db_path = tmp_path / "chat_audit.db"
        Database(db_path).upsert_session("existing-session")
        audit = ChatAudit(db_path=db_path, session_file=tmp_path / "active.local")
        audit.save_session_id("existing-session")
        assert audit.restore_session_id() == "existing-session"

    def test_restore_ignores_unknown_session(self, audit):
        """A mirrored id that was cleared is not restored."""
        audit.save_session_id("ghost-session")
        assert audit.restore_session_id() is None


# ── Past sessions listing ─────────────────────────────────────────────


class TestPastSessions:
    """Tests for Database.get_sessions (the Past Sessions sidebar list)."""

    def test_get_sessions_most_recent_first(self, tmp_path):
        """Sessions are listed most recently active first."""
        db = Database(tmp_path / "sessions.db")
        db.upsert_session("s1")
        db.upsert_session("s2")
        sessions = db.get_sessions()
        assert [s["id"] for s in sessions] == ["s2", "s1"]

    def test_session_exists(self, tmp_path):
        """session_exists reflects the sessions table."""
        db = Database(tmp_path / "sessions.db")
        db.upsert_session("known")
        assert db.session_exists("known") is True
        assert db.session_exists("unknown") is False

    def test_get_chat_messages_ordered(self, session):
        """Database.get_chat_messages returns rows oldest → newest."""
        audit, sid = session
        audit.log_user(sid, "one")
        audit.log_user(sid, "two")
        rows = audit.db.get_chat_messages(sid)
        assert [r["content"] for r in rows] == ["one", "two"]
