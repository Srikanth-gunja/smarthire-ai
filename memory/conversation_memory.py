"""Conversation Memory — shared conversation history across agents.

Single responsibility: Provide a consistent interface for adding messages
to and retrieving messages from the shared conversation history in state,
backed by LangGraph's SQLite checkpointer (``SqliteSaver``).

Per PRD Section 11, this stores:
  - Full conversation history (all messages across turns)
  - Previously discussed job descriptions
  - Previously shortlisted candidates
  - Interview preferences

Persistence:
  - LangGraph conversation state is persisted by ``SqliteSaver`` into
    ``db/smarthire.db`` (checkpoints / writes tables).  The graph is compiled
    with :func:`get_checkpointer` and invoked with
    ``config={"configurable": {"thread_id": <session_id>}}``.
  - :class:`ConversationMemory` keeps a per-process in-memory cache (so the
    public API and fast-path lookups behave exactly as before) and mirrors
    each session's data into a dedicated ``conversation_memory`` checkpoint
    namespace, so history survives Streamlit reruns and app restarts.
  - The ``sessions`` table maps each session id (== LangGraph thread_id) to a
    recruiter/candidate mode and activity timestamps.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from db.database import Database, get_db_path

logger = logging.getLogger(__name__)

# Maximum messages kept in the compressed context window injected into prompts.
_CONTEXT_WINDOW_LIMIT = 20

# Checkpoint namespace used for ConversationMemory session data.  It is kept
# separate from the graph's own namespace (""), so the two never collide even
# though they share the same thread_id in the same SQLite file.
_CONVERSATION_MEMORY_NS = "conversation_memory"

# Stable checkpoint id used for the ConversationMemory checkpoint so a thread
# holds exactly one session-data row (INSERT OR REPLACE) per save.
_CONVERSATION_MEMORY_CHECKPOINT_ID = "00000000-0000-0000-0000-000000000000"

# Sessions idle for longer than this are purged (state cleared) the next time
# a new browser opens the app, so closed-browser data does not linger.
_SESSION_TTL_HOURS = 1.0

# Cached SqliteSaver instances keyed by resolved db path.
_checkpointers: dict[str, Any] = {}


@dataclass
class SessionData:
    """All persistent data for a single SmartHire session."""

    conversation_history: list[BaseMessage] = field(default_factory=list)
    job_descriptions: list[dict] = field(default_factory=list)
    shortlisted_candidates: list[dict] = field(default_factory=list)
    interview_preferences: dict[str, Any] = field(default_factory=dict)


def get_thread_config(thread_id: str) -> dict:
    """Build the RunnableConfig used to address a LangGraph thread.

    Args:
        thread_id: The session id (== LangGraph thread_id).

    Returns:
        A config dict suitable for ``graph.invoke(..., config=...)``.
    """
    return {"configurable": {"thread_id": str(thread_id), "checkpoint_ns": ""}}


def get_checkpointer(db_path: str | Path | None = None) -> Any:
    """Return the shared LangGraph ``SqliteSaver`` for this application.

    The checkpointer is backed by ``db/smarthire.db`` (or the supplied path)
    and is cached per path so Streamlit reruns reuse one connection.

    Args:
        db_path: Optional override path (used by tests).  Defaults to
            ``db/smarthire.db``.

    Returns:
        A configured ``SqliteSaver`` whose tables are already created.
    """
    path = str(Path(db_path) if db_path else get_db_path())

    saver = _checkpointers.get(path)
    if saver is None:
        conn = sqlite3.connect(path, check_same_thread=False)
        from langgraph.checkpoint.sqlite import SqliteSaver

        saver = SqliteSaver(conn)
        saver.setup()
        _checkpointers[path] = saver
        logger.info("Initialised SqliteSaver checkpointer at %s", path)

    return saver


class ConversationMemory:
    """Manages conversation history persistence across graph turns.

    The public API is unchanged from the original in-memory implementation:
    a dict cache keyed by session_id, plus a durable mirror written through
    ``SqliteSaver`` so nothing is lost on app restart.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the Conversation Memory store.

        Args:
            db_path: Optional database path override (used by tests).
        """
        self._sessions: dict[str, SessionData] = {}
        self.db = Database(db_path)
        self._saver = get_checkpointer(db_path)

    # ── Checkpoint (durable) helpers ──────────────────────────────────

    def _save(self, session_id: str, session: SessionData) -> None:
        """Write the session's data into the SqliteSaver checkpointer.

        A single, stable checkpoint id is used for the ``conversation_memory``
        namespace so ``INSERT OR REPLACE`` keeps exactly one checkpoint per
        thread.  (The graph's own namespace uses random ids; sorting by a
        random uuid would otherwise not reflect write order.)
        """
        data = {
            "conversation_history": list(session.conversation_history),
            "job_descriptions": list(session.job_descriptions),
            "shortlisted_candidates": list(session.shortlisted_candidates),
            "interview_preferences": dict(session.interview_preferences),
        }
        checkpoint = {
            "v": 1,
            "ts": datetime.now(UTC).isoformat(),
            "id": _CONVERSATION_MEMORY_CHECKPOINT_ID,
            "channel_values": {"session_data": data},
            "channel_versions": {},
            "versions_seen": {},
            "pending_writes": [],
        }
        config = {
            "configurable": {
                "thread_id": str(session_id),
                "checkpoint_ns": _CONVERSATION_MEMORY_NS,
            }
        }
        try:
            self._saver.put(config, checkpoint, {}, {})
        except Exception:
            logger.exception(
                "Failed to persist conversation memory for session %s", session_id
            )

    def _load(self, session_id: str) -> SessionData | None:
        """Read a session's data back from the checkpointer."""
        config = {
            "configurable": {
                "thread_id": str(session_id),
                "checkpoint_ns": _CONVERSATION_MEMORY_NS,
            }
        }
        try:
            checkpoint_tuple = self._saver.get_tuple(config)
        except Exception:
            logger.exception(
                "Failed to load conversation memory for session %s", session_id
            )
            return None

        if checkpoint_tuple is None:
            return None

        data = (checkpoint_tuple.checkpoint or {}).get("channel_values", {}).get(
            "session_data", {}
        ) or {}
        return SessionData(
            conversation_history=list(data.get("conversation_history", [])),
            job_descriptions=list(data.get("job_descriptions", [])),
            shortlisted_candidates=list(data.get("shortlisted_candidates", [])),
            interview_preferences=dict(data.get("interview_preferences", {})),
        )

    # ── Session lifecycle ──────────────────────────────────────────────

    def cleanup_expired_sessions(self, max_age_hours: float = _SESSION_TTL_HOURS) -> int:
        """Delete sessions (and all their data) idle for ``max_age_hours``.

        Runs whenever a new session is created, so each new visitor purges
        closed-browser state from earlier visitors.  Removes the session row,
        its chat/uploads/interview records, and the LangGraph checkpoints.
        """
        cutoff = (
            datetime.now(UTC) - timedelta(hours=max_age_hours)
        ).strftime("%Y-%m-%d %H:%M:%S")
        rows = self.db.fetch_all(
            "SELECT id FROM sessions WHERE last_active_at < ?", (cutoff,)
        )
        count = 0
        for row in rows:
            session_id = row["id"]
            self._sessions.pop(session_id, None)
            try:
                self._saver.delete_thread(session_id)
            except Exception:
                logger.exception(
                    "Failed to clear checkpoints for session %s", session_id
                )
            self.db.delete_session_data(session_id)
            self.db.delete_session(session_id)
            count += 1
        if count:
            logger.info(
                "Expired %d stale session(s) idle over %.1f hour(s)",
                count,
                max_age_hours,
            )
        return count

    def create_session(self, mode: str | None = None) -> str:
        """Create a new empty session, persist it, and return its id.

        Args:
            mode: Optional session mode ('recruiter' | 'candidate').

        Returns:
            A new UUID-based session identifier (== LangGraph thread_id).
        """
        self.cleanup_expired_sessions()
        session_id = uuid.uuid4().hex[:12]
        self._sessions[session_id] = SessionData()
        self.db.upsert_session(session_id, mode)
        self._save(session_id, self._sessions[session_id])
        logger.info("Created session %s", session_id)
        return session_id

    def resume_last_session(self) -> str | None:
        """Return the most recently active persisted session id, if any.

        Used at app startup so a restart continues the previous conversation
        instead of silently starting a brand-new thread.
        """
        return self.db.resume_last_session()

    def set_mode(self, session_id: str, mode: str) -> None:
        """Record the recruiter/candidate mode for a session."""
        self.db.upsert_session(session_id, mode)

    def get_session(self, session_id: str) -> SessionData:
        """Return session data, restoring from the checkpointer if needed.

        Args:
            session_id: The session identifier.

        Returns:
            The SessionData for the given session.
        """
        if session_id not in self._sessions:
            restored = self._load(session_id)
            self._sessions[session_id] = restored or SessionData()
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Erase all data for a session (memory + checkpointer + session row).

        Args:
            session_id: The session to clear.
        """
        self._sessions.pop(session_id, None)
        try:
            self._saver.delete_thread(session_id)
        except Exception:
            logger.exception("Failed to clear checkpoints for session %s", session_id)
        self.db.delete_session_data(session_id)
        self.db.delete_session(session_id)
        logger.info("Cleared session %s", session_id)

    def clear_all_sessions(self) -> None:
        """Erase checkpoints and persisted data for every SmartHire session."""
        session_ids = [row["id"] for row in self.db.get_sessions()]
        for session_id in session_ids:
            self._sessions.pop(session_id, None)
            try:
                self._saver.delete_thread(session_id)
            except Exception:
                logger.exception("Failed to clear checkpoints for session %s", session_id)
        self.db.clear_all_hiring_data()
        logger.info("Cleared all SmartHire sessions: count=%d", len(session_ids))

    def reset_session(self, session_id: str, mode: str | None = None) -> None:
        """Clear a session's working data while retaining its identity.

        This supports the UI's "Clear Current Session" action: the recruiter
        stays in the same session, while conversation and graph state start
        fresh. A separate session is created only on an explicit request.
        """
        self._sessions[session_id] = SessionData()
        try:
            self._saver.delete_thread(session_id)
        except Exception:
            logger.exception("Failed to reset checkpoints for session %s", session_id)
        self.db.upsert_session(session_id, mode)
        self._save(session_id, self._sessions[session_id])
        logger.info("Reset session %s", session_id)

    # ── Conversation history ───────────────────────────────────────────

    def append_turn(
        self,
        session_id: str,
        message: BaseMessage,
    ) -> None:
        """Append a single message to the session's conversation history.

        Args:
            session_id: The session to update.
            message: The message to append.
        """
        session = self.get_session(session_id)
        session.conversation_history.append(message)
        self._save(session_id, session)

    def get_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[BaseMessage]:
        """Retrieve conversation history for a session.

        Args:
            session_id: The session to read from.
            limit: Max recent messages to return (None = all).

        Returns:
            List of messages, oldest first.
        """
        session = self.get_session(session_id)
        history = session.conversation_history
        if limit is not None:
            return list(history[-limit:])
        return list(history)

    # ── Job descriptions ───────────────────────────────────────────────

    def store_job_description(self, session_id: str, jd: dict) -> None:
        """Cache a parsed job description for the session.

        Args:
            session_id: The session to update.
            jd: Parsed JD dict (job_title, required_skills, etc.).
        """
        session = self.get_session(session_id)
        session.job_descriptions.append(jd)
        self._save(session_id, session)

    def get_job_descriptions(self, session_id: str) -> list[dict]:
        """Return all JDs discussed in this session.

        Args:
            session_id: The session to read from.

        Returns:
            List of JD dicts.
        """
        return list(self.get_session(session_id).job_descriptions)

    # ── Shortlisted candidates ─────────────────────────────────────────

    def add_shortlisted_candidate(self, session_id: str, candidate: dict) -> None:
        """Record a shortlisted candidate for the session.

        Args:
            session_id: The session to update.
            candidate: Candidate dict (name, score, etc.).
        """
        session = self.get_session(session_id)
        # Deduplicate by candidate name if present.
        name = candidate.get("candidate_name")
        if name:
            session.shortlisted_candidates = [
                c for c in session.shortlisted_candidates
                if c.get("candidate_name") != name
            ]
        session.shortlisted_candidates.append(candidate)
        self._save(session_id, session)

    def get_shortlisted_candidates(self, session_id: str) -> list[dict]:
        """Return all shortlisted candidates for the session.

        Args:
            session_id: The session to read from.

        Returns:
            List of candidate dicts.
        """
        return list(self.get_session(session_id).shortlisted_candidates)

    # ── Interview preferences ──────────────────────────────────────────

    def update_interview_preferences(
        self, session_id: str, preferences: dict[str, Any]
    ) -> None:
        """Merge new interview preferences into the session.

        Args:
            session_id: The session to update.
            preferences: Dict of preferences (date, time_window, format, etc.).
        """
        session = self.get_session(session_id)
        session.interview_preferences.update(preferences)
        self._save(session_id, session)

    def get_interview_preferences(self, session_id: str) -> dict[str, Any]:
        """Return interview preferences for the session.

        Args:
            session_id: The session to read from.

        Returns:
            Dict of interview preferences.
        """
        return dict(self.get_session(session_id).interview_preferences)

    # ── Context summary for prompt injection ───────────────────────────

    def get_context_summary(self, session_id: str) -> str:
        """Build a compressed summary of the session for prompt injection.

        Instead of injecting the full conversation history (which grows
        unbounded), this produces a bounded string summarising what has
        happened so far: the JD(s), shortlisted candidates, and any
        interview preferences.  The most recent N messages are included
        verbatim for continuity.

        Args:
            session_id: The session to summarise.

        Returns:
            A plaintext summary suitable for LLM prompt context.
        """
        session = self.get_session(session_id)
        parts: list[str] = []

        # Job descriptions discussed
        if session.job_descriptions:
            latest_jd = session.job_descriptions[-1]
            title = latest_jd.get("job_title", "Unknown role")
            skills = latest_jd.get("required_skills", [])
            parts.append(f"Current role: {title}")
            if skills:
                parts.append(f"Required skills: {', '.join(skills)}")

        # Shortlisted candidates
        if session.shortlisted_candidates:
            names = [
                c.get("candidate_name", "Unknown")
                for c in session.shortlisted_candidates
            ]
            parts.append(f"Shortlisted candidates: {', '.join(names)}")

        # Interview preferences
        prefs = session.interview_preferences
        if prefs:
            pref_str = ", ".join(f"{k}={v}" for k, v in prefs.items())
            parts.append(f"Interview preferences: {pref_str}")

        # Recent conversation messages (bounded window)
        recent = session.conversation_history[-_CONTEXT_WINDOW_LIMIT:]
        if recent:
            msg_lines = []
            for msg in recent:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                # Truncate very long messages to keep summary compact.
                content = msg.content[:200]
                if len(msg.content) > 200:
                    content += "…"
                msg_lines.append(f"  {role}: {content}")
            parts.append("Recent conversation:\n" + "\n".join(msg_lines))

        return "\n".join(parts) if parts else "(New session — no history yet.)"
