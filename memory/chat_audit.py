"""Chat message audit log — durable, human-readable conversation transcript.

Backed by the ``chat_messages`` table in ``db/smarthire.db``. Every turn in
the Chat tab and Candidate Chat mode is written here immediately, so a
browser refresh (or app restart) can reload the full conversation.

This is deliberately separate from LangGraph's SqliteSaver checkpoints:
those hold agent state for resuming graph execution, while this table is the
human-readable display/audit transcript.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from db.database import Database

logger = logging.getLogger(__name__)

# Small local file mirroring the active session id so a refresh (or a new
# browser tab) can resume the same conversation.  Ignored by git via
# ``data/*.local``.
SESSION_FILE = Path(__file__).resolve().parents[1] / "data" / "active_session.local"

# Map an agent message content prefix (e.g. "[ResumeScreening] ...") to the
# canonical agent name recorded in chat_messages.agent_name.
_AGENT_PREFIX_MAP: dict[str, str] = {
    "[Supervisor]": "supervisor",
    "[ResumeScreening]": "resume_screening",
    "[CandidateMatching]": "candidate_matching",
    "[InterviewScheduling]": "interview_scheduling",
    "[HRAssistant]": "hr_assistant",
}

# Roles that make up the visible chat transcript.  Agent turns are persisted
# as an audit trail but not rendered in the Streamlit chat UI.
DISPLAY_ROLES = frozenset({"user", "assistant"})


class ChatAudit:
    """Writes and reads the ``chat_messages`` conversation transcript.

    Also owns the tiny session-id mirror file used to resume the active
    conversation across browser refreshes / app restarts.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        session_file: str | Path | None = None,
    ) -> None:
        """Initialise the audit log.

        Args:
            db_path: Optional database path override (used by tests).
            session_file: Optional path for the session-id mirror file.
        """
        self.db = Database(db_path)
        self.session_file = Path(session_file) if session_file else SESSION_FILE

    # ── Writing ───────────────────────────────────────────────────────

    def log_message(
        self,
        session_id: str,
        role: str,
        content: str,
        agent_name: str | None = None,
        mode: str | None = None,
    ) -> str | None:
        """Persist a single message immediately (not batched).

        Args:
            session_id: The session this turn belongs to.
            role: 'user', 'assistant', or 'agent:<agent_name>'.
            content: The message text.
            agent_name: The agent that produced this turn (None for user turns).
            mode: 'recruiter' | 'candidate' transcript owner (None = legacy).
        """
        return self.db.insert_chat_message(
            session_id, role, content, agent_name, mode
        )

    def log_user(
        self, session_id: str, content: str, mode: str | None = None
    ) -> str | None:
        """Persist a user turn."""
        return self.log_message(session_id, "user", content, mode=mode)

    def log_assistant(
        self, session_id: str, content: str, mode: str | None = None
    ) -> str | None:
        """Persist an assistant turn."""
        return self.log_message(session_id, "assistant", content, mode=mode)

    def log_agent(
        self,
        session_id: str,
        agent_name: str,
        content: str,
        mode: str | None = None,
    ) -> str | None:
        """Persist an agent turn with role 'agent:<agent_name>'."""
        return self.log_message(
            session_id,
            f"agent:{agent_name}",
            content,
            agent_name=agent_name,
            mode=mode,
        )

    def log_turn(
        self,
        session_id: str,
        user_content: str | None = None,
        result: dict | None = None,
        answer: str | None = None,
        prior_history_count: int = 0,
        mode: str | None = None,
    ) -> None:
        """Persist one full turn: user prompt, agent notes, final answer.

        Args:
            session_id: The session this turn belongs to.
            user_content: The user's message (None to skip, e.g. for a
                button-triggered pipeline run rather than a chat turn).
            result: Optional graph result state.  When given, intermediate
                agent messages from this turn are logged with the
                'agent:<agent_name>' role and the final_response as assistant.
            answer: Fallback assistant text when no graph result is available
                (e.g. Ollama is unreachable).
            prior_history_count: Number of conversation_history messages that
                already existed before this turn, so only new ones are logged.
            mode: 'recruiter' | 'candidate' transcript owner (None = legacy).
        """
        if user_content:
            self.log_user(session_id, user_content, mode=mode)

        if result is not None:
            history = result.get("conversation_history", [])
            for msg in history[prior_history_count:]:
                agent_name, text = self._extract_agent_turn(msg)
                if agent_name:
                    self.log_agent(session_id, agent_name, text, mode=mode)
            final_val = result.get("final_response") or ""
            if isinstance(final_val, list):
                final_val = " ".join(str(m) for m in final_val)
            final = str(final_val).strip()
            if final:
                self.log_assistant(session_id, final, mode=mode)
        elif answer:
            self.log_assistant(session_id, answer, mode=mode)

    @staticmethod
    def _extract_agent_turn(msg: Any) -> tuple[str | None, str]:
        """Return (agent_name, content) if msg is a prefixed agent note.

        Graph agent nodes write messages whose content starts with a bracketed
        name, e.g. "[ResumeScreening] Screened Alice: match_score=85".  User
        turns and plain assistant text return ``(None, text)``.
        """
        content = str(getattr(msg, "content", "") or "")
        for prefix, agent_name in _AGENT_PREFIX_MAP.items():
            if content.startswith(prefix):
                return agent_name, content
        return None, content

    # ── Reading ───────────────────────────────────────────────────────

    def load_messages(
        self,
        session_id: str,
        display_only: bool = True,
        mode: str | None = None,
    ) -> list[dict]:
        """Return a session's chat messages, oldest first.

        Args:
            session_id: The session to read from.
            display_only: When True (default), only user/assistant turns are
                returned — agent turns stay in the table as an audit trail.
            mode: Optional 'recruiter' | 'candidate' to restrict to one
                transcript. When None, all modes are returned.

        Returns:
            List of dicts with keys 'role', 'content', 'agent_name',
            'created_at'.
        """
        messages: list[dict] = []
        for row in self.db.get_chat_messages(session_id, mode=mode):
            if display_only and row["role"] not in DISPLAY_ROLES:
                continue
            messages.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "agent_name": row["agent_name"],
                    "created_at": row["created_at"],
                }
            )
        return messages

    def clear(self, session_id: str, mode: str | None = None) -> None:
        """Remove chat messages for a session.

        Args:
            session_id: The session to clear.
            mode: Optional 'recruiter' | 'candidate' to clear only one
                transcript. When None, every transcript is removed.
        """
        self.db.delete_chat_messages(session_id, mode=mode)

    # ── Session mirror file ───────────────────────────────────────────

    def save_session_id(self, session_id: str) -> None:
        """Mirror the active session id to a small local file."""
        try:
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            self.session_file.write_text(session_id, encoding="utf-8")
        except Exception:
            logger.exception("Failed to persist active session id")

    def load_session_id(self) -> str | None:
        """Read the mirrored session id, if any."""
        try:
            if self.session_file.exists():
                session_id = self.session_file.read_text(encoding="utf-8").strip()
                return session_id or None
        except Exception:
            logger.exception("Failed to read active session id")
        return None

    def clear_session_id(self) -> None:
        """Remove the mirrored session id file."""
        try:
            if self.session_file.exists():
                self.session_file.unlink()
        except Exception:
            logger.exception("Failed to clear active session id")

    def restore_session_id(self) -> str | None:
        """Return the mirrored session id if it still exists in the DB.

        Used at app startup so a refresh continues the previous conversation.
        Falls back to None (caller decides next step) when the file is missing
        or the referenced session has been cleared.
        """
        session_id = self.load_session_id()
        if session_id and self.db.session_exists(session_id):
            return session_id
        return None
