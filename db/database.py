"""SQLite persistence layer for SmartHire AI.

Wraps Python's built-in ``sqlite3`` module behind a small connection-manager
class (:class:`Database`). Provides:

* Idempotent schema initialisation (:meth:`Database.init_db`) from
  ``db/schema.sql``.
* WAL journal mode so concurrent Streamlit reruns can read/write safely.
* Typed insert / lookup helpers for every recruitment table.
* Defensive error handling — a failed DB write is logged and swallowed so it
  never crashes the Streamlit app.

The LangGraph checkpointer (``SqliteSaver``) shares the same
``db/smarthire.db`` file (its ``checkpoints`` / ``writes`` tables) — see
:func:`memory.conversation_memory.get_checkpointer`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from utils.observability import instrument_tool_methods

logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = DB_DIR / "smarthire.db"
SCHEMA_PATH = DB_DIR / "schema.sql"

# status values allowed by the interviews.status column
INTERVIEW_STATUSES = frozenset({"proposed", "confirmed", "cancelled"})


def get_db_path() -> Path:
    """Return the filesystem path of the shared SQLite database file."""
    return DEFAULT_DB_PATH


def _json(value: Any) -> str | None:
    """Serialise a Python value to a JSON text column (or None)."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _new_id() -> str:
    """Return a fresh UUID hex string used as a primary key."""
    return uuid.uuid4().hex


@instrument_tool_methods
class Database:
    """Small ``sqlite3`` connection manager plus persistence helpers.

    Connections are short-lived (one per operation, via the context manager),
    so concurrent Streamlit reruns never share a single connection.  WAL mode
    permits one writer plus many simultaneous readers.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialise the manager for a given database file.

        Args:
            db_path: Path to the SQLite file.  Defaults to ``db/smarthire.db``.
        """
        self.db_path = Path(db_path) if db_path else get_db_path()
        self._schema_ready = False

    # ── Schema ─────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """Apply ``db/schema.sql`` once per instance (idempotent DDL)."""
        if self._schema_ready:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path), timeout=30) as conn:
            # Run migrations first so new columns exist before any indexes
            # in schema.sql that reference them.
            self._migrate_sessions(conn)
            self._migrate_interviews(conn)
            self._migrate_chat_messages(conn)
            conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._schema_ready = True

    @staticmethod
    def _migrate_sessions(conn: sqlite3.Connection) -> None:
        """Add new columns to the sessions table for existing databases."""
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            )
            if not cursor.fetchone():
                return  # table doesn't exist yet; schema.sql will create it
            cursor = conn.execute("PRAGMA table_info(sessions)")
            existing = {row[1] for row in cursor.fetchall()}
        except Exception:
            return
        if "paused_at_node" not in existing:
            conn.execute("ALTER TABLE sessions ADD COLUMN paused_at_node TEXT")
        if "error_message" not in existing:
            conn.execute("ALTER TABLE sessions ADD COLUMN error_message TEXT")

    @staticmethod
    def _migrate_interviews(conn: sqlite3.Connection) -> None:
        """Add new columns to the interviews table for existing databases."""
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='interviews'"
            )
            if not cursor.fetchone():
                return  # table doesn't exist yet; schema.sql will create it
            cursor = conn.execute("PRAGMA table_info(interviews)")
            existing = {row[1] for row in cursor.fetchall()}
        except Exception:
            return
        if "session_id" not in existing:
            conn.execute("ALTER TABLE interviews ADD COLUMN session_id TEXT")
        if "interview_type" not in existing:
            conn.execute(
                "ALTER TABLE interviews ADD COLUMN interview_type TEXT DEFAULT 'technical'"
            )
        if "interviewer" not in existing:
            conn.execute("ALTER TABLE interviews ADD COLUMN interviewer TEXT")

    @staticmethod
    def _migrate_chat_messages(conn: sqlite3.Connection) -> None:
        """Add the mode column to chat_messages for existing databases."""
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_messages'"
            )
            if not cursor.fetchone():
                return  # table doesn't exist yet; schema.sql will create it
            cursor = conn.execute("PRAGMA table_info(chat_messages)")
            existing = {row[1] for row in cursor.fetchall()}
        except Exception:  # noqa: BLE001
            return
        if "mode" not in existing:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN mode TEXT")

    # ── Connections ────────────────────────────────────────────────────

    def connect(self) -> sqlite3.Connection:
        """Open a new connection (schema + row factory + WAL + FK)."""
        self._ensure_schema()
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def init_db(self) -> None:
        """Create all tables idempotently from ``db/schema.sql``."""
        with self.connect() as conn:
            conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._schema_ready = True
        logger.info("Database initialised at %s", self.db_path)

    # ── Generic queries ────────────────────────────────────────────────

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int | None:
        """Run a write statement and return the rowid (None on failure)."""
        try:
            with self.connect() as conn:
                return conn.execute(sql, tuple(params)).lastrowid
        except sqlite3.Error:
            logger.exception("DB execute failed: %s", sql[:120])
            return None

    def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        """Fetch a single row (None on failure or no match)."""
        try:
            with self.connect() as conn:
                return conn.execute(sql, tuple(params)).fetchone()
        except sqlite3.Error:
            logger.exception("DB fetch_one failed: %s", sql[:120])
            return None

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Fetch all matching rows (empty list on failure)."""
        try:
            with self.connect() as conn:
                return conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.Error:
            logger.exception("DB fetch_all failed: %s", sql[:120])
            return []

    # ── Sessions ───────────────────────────────────────────────────────

    def upsert_session(self, session_id: str, mode: str | None = None) -> None:
        """Record (or refresh) a session row for a LangGraph thread_id."""
        if mode:
            self.execute(
                """
                INSERT INTO sessions (id, mode, last_active_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    mode = excluded.mode,
                    last_active_at = CURRENT_TIMESTAMP
                """,
                (session_id, mode),
            )
        else:
            self.execute(
                """
                INSERT INTO sessions (id, last_active_at)
                VALUES (?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET last_active_at = CURRENT_TIMESTAMP
                """,
                (session_id,),
            )

    def delete_session(self, session_id: str) -> None:
        """Remove a session row (checkpointer data is removed separately)."""
        self.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def resume_last_session(self) -> str | None:
        """Return the most recently active session id, if any."""
        row = self.fetch_one(
            "SELECT id FROM sessions ORDER BY last_active_at DESC, started_at DESC LIMIT 1"
        )
        return row["id"] if row else None

    def session_exists(self, session_id: str) -> bool:
        """Return True if a session row exists for the given id."""
        row = self.fetch_one("SELECT id FROM sessions WHERE id = ?", (session_id,))
        return row is not None

    def get_sessions(self) -> list[sqlite3.Row]:
        """Return all sessions, most recent first (for the Past Sessions list)."""
        return self.fetch_all(
            "SELECT * FROM sessions "
            "ORDER BY last_active_at DESC, started_at DESC, rowid DESC"
        )

    def update_session_paused(
        self, session_id: str, node_name: str, error_message: str
    ) -> None:
        """Mark a session as paused due to a transient error."""
        self.execute(
            """
            UPDATE sessions
            SET paused_at_node = ?, error_message = ?, last_active_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (node_name, error_message, session_id),
        )

    def clear_session_paused(self, session_id: str) -> None:
        """Clear the paused state after a successful resume or retry."""
        self.execute(
            """
            UPDATE sessions
            SET paused_at_node = NULL, error_message = NULL, last_active_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (session_id,),
        )

    def get_session_paused(self, session_id: str) -> dict | None:
        """Return the paused state for a session, or None if not paused."""
        row = self.fetch_one(
            "SELECT paused_at_node, error_message FROM sessions WHERE id = ?",
            (session_id,),
        )
        if row and row["paused_at_node"]:
            return {
                "paused_at_node": row["paused_at_node"],
                "error_message": row["error_message"] or "",
            }
        return None

    # ── Job descriptions ───────────────────────────────────────────────

    def persist_job_description(
        self,
        jd: dict | None = None,
        raw_text: str | None = None,
        source: str | None = None,
    ) -> str | None:
        """Insert a JD row (or reuse an existing one by title).

        Returns the row id, or None if the write failed.
        """
        jd = jd or {}
        if jd.get("id"):
            return jd["id"]

        title = jd.get("job_title") or jd.get("title") or _sniff_title(raw_text)
        if title:
            existing = self.fetch_one(
                "SELECT id FROM job_descriptions WHERE title = ? ORDER BY created_at DESC LIMIT 1",
                (title,),
            )
            if existing:
                return existing["id"]

        body = raw_text or jd.get("raw_text") or json.dumps(jd, ensure_ascii=False)
        jd_id = _new_id()
        self.execute(
            """
            INSERT INTO job_descriptions (id, title, raw_text, source)
            VALUES (?, ?, ?, ?)
            """,
            (jd_id, title, body, source),
        )
        return jd_id

    def get_job_descriptions(self) -> list[sqlite3.Row]:
        """Return all persisted job descriptions (newest first)."""
        return self.fetch_all(
            "SELECT * FROM job_descriptions ORDER BY created_at DESC"
        )

    def get_recent_job_description_id(self) -> str | None:
        """Return the id of the most recently inserted job description."""
        row = self.fetch_one(
            "SELECT id FROM job_descriptions ORDER BY created_at DESC, rowid DESC LIMIT 1"
        )
        return row["id"] if row else None

    # ── Candidates ─────────────────────────────────────────────────────

    def persist_candidate(
        self,
        name: str | None,
        resume_raw_text: str,
        resume_filename: str | None = None,
        skills: list[str] | None = None,
        experience_years: float | None = None,
        email: str | None = None,
    ) -> str | None:
        """Insert a candidate (reuse + refresh the row when the name repeats)."""
        normalized = (name or "").strip().lower()
        if normalized:
            existing = self.fetch_one(
                "SELECT id FROM candidates WHERE LOWER(name) = ? ORDER BY created_at DESC LIMIT 1",
                (normalized,),
            )
            if existing:
                self.execute(
                    """
                    UPDATE candidates
                    SET resume_raw_text = ?, extracted_skills = ?,
                        extracted_experience_years = ?, email = COALESCE(?, email)
                    WHERE id = ?
                    """,
                    (
                        resume_raw_text,
                        _json(skills),
                        experience_years,
                        email,
                        existing["id"],
                    ),
                )
                return existing["id"]

        candidate_id = _new_id()
        self.execute(
            """
            INSERT INTO candidates
                (id, name, email, resume_raw_text, resume_filename,
                 extracted_skills, extracted_experience_years)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                name,
                email,
                resume_raw_text,
                resume_filename,
                _json(skills),
                experience_years,
            ),
        )
        return candidate_id

    def find_candidate_by_name(self, name: str) -> str | None:
        """Look up a candidate id by exact (case-insensitive) name."""
        row = self.fetch_one(
            "SELECT id FROM candidates WHERE LOWER(name) = LOWER(?) ORDER BY created_at DESC LIMIT 1",
            (name,),
        )
        return row["id"] if row else None

    def get_candidates(self) -> list[sqlite3.Row]:
        """Return all candidates (newest first)."""
        return self.fetch_all(
            "SELECT * FROM candidates ORDER BY created_at DESC"
        )

    # ── Screenings ─────────────────────────────────────────────────────

    def insert_screening(
        self,
        candidate_id: str | None,
        jd_id: str | None,
        summary: str | None = None,
        strengths: list[str] | None = None,
        gaps: list[str] | None = None,
    ) -> str | None:
        """Persist a Resume Screening Agent output row."""
        screening_id = _new_id()
        self.execute(
            """
            INSERT INTO screenings (id, candidate_id, jd_id, summary, strengths, gaps)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (screening_id, candidate_id, jd_id, summary, _json(strengths), _json(gaps)),
        )
        return screening_id

    # ── Candidate rankings ─────────────────────────────────────────────

    def insert_candidate_ranking(
        self,
        candidate_id: str | None,
        jd_id: str | None,
        match_score: float,
        rank_position: int | None = None,
        reasoning: str | None = None,
    ) -> str | None:
        """Persist one Candidate Matching Agent output row."""
        ranking_id = _new_id()
        self.execute(
            """
            INSERT INTO candidate_rankings
                (id, candidate_id, jd_id, match_score, rank_position, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ranking_id, candidate_id, jd_id, match_score, rank_position, reasoning),
        )
        return ranking_id

    def update_ranking_reflection(
        self,
        jd_id: str | None,
        validation_passed: bool,
        notes: dict | None = None,
    ) -> None:
        """Stamp reflection results onto the rankings for a JD.

        ``reflection_notes`` stores the full structured notes dict (per-check
        pass/fail, corrections, retry info) as JSON so the UI can render it.

        Args:
            jd_id: The job description whose ranking rows to update.
            validation_passed: The final reflection_validated boolean.
            notes: The structured reflection_notes dict to persist (JSON).
        """
        if not jd_id:
            return
        self.execute(
            """
            UPDATE candidate_rankings
            SET reflection_validated = ?, reflection_notes = ?
            WHERE jd_id = ?
            """,
            (1 if validation_passed else 0, _json(notes), jd_id),
        )

    def get_candidate_rankings(self) -> list[sqlite3.Row]:
        """Return all rankings (newest first)."""
        return self.fetch_all(
            """
            SELECT r.*, c.name AS candidate_name, jd.title AS jd_title
            FROM candidate_rankings r
            LEFT JOIN candidates c ON c.id = r.candidate_id
            LEFT JOIN job_descriptions jd ON jd.id = r.jd_id
            ORDER BY r.created_at DESC
            """
        )

    # ── Interviews ─────────────────────────────────────────────────────

    def insert_interview(
        self,
        candidate_id: str | None,
        jd_id: str | None,
        proposed_start: str | None = None,
        proposed_end: str | None = None,
        status: str = "proposed",
        session_id: str | None = None,
        interview_type: str | None = None,
        interviewer: str | None = None,
    ) -> str | None:
        """Persist one Interview Scheduling Agent output row."""
        if status not in INTERVIEW_STATUSES:
            status = "proposed"
        interview_id = _new_id()
        self.execute(
            """
            INSERT INTO interviews
                (id, candidate_id, jd_id, session_id, proposed_start, proposed_end,
                 interview_type, interviewer, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interview_id, candidate_id, jd_id, session_id,
                proposed_start, proposed_end, interview_type, interviewer, status,
            ),
        )
        return interview_id

    def get_interviews(self) -> list[sqlite3.Row]:
        """Return all interview rows (newest first)."""
        return self.fetch_all(
            """
            SELECT i.*, c.name AS candidate_name, jd.title AS jd_title
            FROM interviews i
            LEFT JOIN candidates c ON c.id = i.candidate_id
            LEFT JOIN job_descriptions jd ON jd.id = i.jd_id
            ORDER BY i.created_at DESC
            """
        )

    def get_interviews_by_session(self, session_id: str) -> list[sqlite3.Row]:
        """Return interview rows for a specific session, soonest first."""
        return self.fetch_all(
            """
            SELECT i.*, c.name AS candidate_name, jd.title AS jd_title
            FROM interviews i
            LEFT JOIN candidates c ON c.id = i.candidate_id
            LEFT JOIN job_descriptions jd ON jd.id = i.jd_id
            WHERE i.session_id = ?
            ORDER BY i.proposed_start ASC
            """,
            (session_id,),
        )

    def update_interview_status(self, interview_id: str, status: str) -> bool:
        """Update the status of an interview. Returns True if a row was changed."""
        if status not in INTERVIEW_STATUSES:
            return False
        cursor = self.execute(
            "UPDATE interviews SET status = ? WHERE id = ?",
            (status, interview_id),
        )
        return cursor.rowcount > 0

    # ── HR answers ─────────────────────────────────────────────────────

    def insert_hr_answer(
        self,
        query: str | None,
        answer: str,
        session_id: str | None = None,
        sources: list[str] | None = None,
        confidence: float | None = None,
        needs_escalation: bool = False,
    ) -> str | None:
        """Persist one HR Assistant Agent output row."""
        answer_id = _new_id()
        self.execute(
            """
            INSERT INTO hr_answers
                (id, session_id, query, answer, sources, confidence, needs_escalation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                answer_id,
                session_id,
                query,
                answer,
                _json(sources),
                confidence,
                1 if needs_escalation else 0,
            ),
        )
        return answer_id

    def get_hr_answers(self) -> list[sqlite3.Row]:
        """Return all HR answers (newest first)."""
        return self.fetch_all(
            "SELECT * FROM hr_answers ORDER BY created_at DESC"
        )

    # ── Chat messages ─────────────────────────────────────────────────

    def insert_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        agent_name: str | None = None,
        mode: str | None = None,
    ) -> str | None:
        """Persist one chat turn for a session (human-readable audit log).

        Args:
            session_id: The session (== LangGraph thread_id) this turn belongs to.
            role: 'user', 'assistant', or 'agent:<agent_name>'.
            content: The message text.
            agent_name: The agent that produced this turn (None for user turns).
            mode: 'recruiter' | 'candidate' owning this transcript (None for
                legacy rows written before mode separation).

        Returns:
            The new row id, or None if the write failed.
        """
        message_id = _new_id()
        self.execute(
            """
            INSERT INTO chat_messages (id, session_id, role, content, agent_name, mode)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, session_id, role, content, agent_name, mode),
        )
        return message_id

    def get_chat_messages(
        self,
        session_id: str,
        mode: str | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        """Return a session's chat messages, oldest first.

        Args:
            session_id: The session to read from.
            mode: Optional 'recruiter' | 'candidate' to restrict to one
                transcript. When None, all modes are returned.
            limit: Max recent messages to return (None = all).

        Returns:
            List of chat_messages rows ordered oldest → newest.
        """
        sql = "SELECT * FROM chat_messages WHERE session_id = ?"
        params: list[Any] = [session_id]
        if mode:
            sql += " AND mode = ?"
            params.append(mode)
        sql += " ORDER BY created_at ASC, rowid ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self.fetch_all(sql, tuple(params))

    def delete_chat_messages(
        self, session_id: str, mode: str | None = None
    ) -> None:
        """Remove chat messages for a session (used by Clear Session).

        Args:
            session_id: The session to clear.
            mode: Optional 'recruiter' | 'candidate' to clear only one
                transcript. When None, every transcript for the session is
                removed.
        """
        if mode:
            self.execute(
                "DELETE FROM chat_messages WHERE session_id = ? AND mode = ?",
                (session_id, mode),
            )
        else:
            self.execute(
                "DELETE FROM chat_messages WHERE session_id = ?", (session_id,)
            )


def _sniff_title(raw_text: str | None) -> str | None:
    """Heuristically extract a job title from raw JD text (first line)."""
    if not raw_text:
        return None
    for line in raw_text.splitlines():
        line = line.strip()
        if line and not line.lower().startswith(("job", "position", "role")):
            return line[:200]
    return None
