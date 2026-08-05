"""SmartHire AI — Streamlit Frontend.

Two modes:
  • Recruiter Dashboard — upload resumes + JD, screen, rank, schedule, chat
  • Candidate Chat — ask HR questions about the recruitment process

The UI is styled as a real recruiter product: an enterprise teal theme
(``.streamlit/config.toml``), bordered cards instead of flat upload boxes,
per-agent progress during multi-agent execution, badge-labelled chat turns,
and a data-driven System Insight tab.
"""

from __future__ import annotations

import datetime
import io
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from pypdf import PdfReader

from db.database import Database
from memory.chat_audit import ChatAudit
from memory.conversation_memory import (
    ConversationMemory,
    get_checkpointer,
    get_thread_config,
)
from memory.state import SmartHireState
from tools.calendar_tool import CalendarTool
from utils.models import InterviewSlot

logger = logging.getLogger(__name__)


def _configure_console_logging() -> None:
    """Ensure execution logs are visible in the Streamlit terminal."""
    execution_logger = logging.getLogger("smarthire.execution")
    execution_logger.setLevel(logging.INFO)
    execution_logger.propagate = False
    if not any(getattr(handler, "_smarthire_console", False) for handler in execution_logger.handlers):
        handler = logging.StreamHandler()
        handler._smarthire_console = True  # type: ignore[attr-defined]
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        execution_logger.addHandler(handler)


_configure_console_logging()

# Initialise the SQLite persistence layer (idempotent CREATE TABLE IF NOT EXISTS).
Database().init_db()

# ── Page config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="SmartHire AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Agent metadata (labels/icons/colors for chat badges + progress) ────

AGENTS: dict[str, dict[str, str]] = {
    "supervisor": {"label": "Supervisor", "icon": "🧭", "color": "#2563EB"},
    "resume_screening": {"label": "Resume Screening", "icon": "📄", "color": "#7C3AED"},
    "candidate_matching": {"label": "Candidate Matching", "icon": "🎯", "color": "#0D9488"},
    "interview_scheduling": {"label": "Interview Scheduling", "icon": "📅", "color": "#EA580C"},
    "hr_assistant": {"label": "HR Assistant", "icon": "💬", "color": "#059669"},
    "memory_update": {"label": "Memory Update", "icon": "🧠", "color": "#64748B"},
    "reflection": {"label": "Reflection", "icon": "🧪", "color": "#64748B"},
}

# Expected execution order used by the progress checklist.
_STAGE_ORDER = [
    "supervisor",
    "resume_screening",
    "candidate_matching",
    "interview_scheduling",
    "hr_assistant",
    "memory_update",
    "reflection",
]

_WORKFLOW_STAGES = {
    "screening": ["supervisor", "resume_screening", "memory_update", "reflection"],
    "matching": ["supervisor", "candidate_matching", "memory_update", "reflection"],
}

_PRIMARY = "#0E7C86"
_MUTED = "#6B7280"
_BORDER = "#E5E7EB"


# ── Small rendering helpers ───────────────────────────────────────────


def _agent_label(name: str) -> str:
    meta = AGENTS.get(name)
    return meta["label"] if meta else name.replace("_", " ").title()


def _agent_badge(name: str) -> str:
    meta = AGENTS.get(name, {"label": _agent_label(name), "icon": "🤖", "color": "#64748B"})
    return (
        f"<span style='background:{meta['color']};color:#fff;padding:2px 10px;"
        f"border-radius:12px;font-size:0.78em;font-weight:600'>"
        f"{meta['icon']} {meta['label']}</span>"
    )


def _score_color(score: float) -> str:
    if score >= 80:
        return _PRIMARY
    if score >= 60:
        return "#D97706"
    return "#B91C1C"


def _score_meter(score: float) -> str:
    pct = max(0, min(100, int(score or 0)))
    color = _score_color(score or 0)
    return (
        f"<div style='background:#E5E7EB;border-radius:6px;height:10px;width:100%'>"
        f"<div style='background:{color};border-radius:6px;height:10px;width:{pct}%'></div></div>"
        f"<div style='color:{color};font-weight:700;font-size:0.95em;margin-top:6px'>"
        f"{pct}% match</div>"
    )


def _stage_checklist(
    completed: list[str], running: str | None, workflow: str | None = None
) -> str:
    """HTML progress checklist showing pending / running / done agents."""
    done = set(completed)
    lines: list[str] = []
    for node in _WORKFLOW_STAGES.get(workflow or "", _STAGE_ORDER):
        meta = AGENTS.get(node, {})
        icon, label = meta.get("icon", "⚙️"), meta.get("label", node)
        if node in done:
            lines.append(f"<span style='color:{_PRIMARY}'>✓ {icon} {label}</span>")
        elif node == running:
            lines.append(
                f"<span style='color:#2563EB;font-weight:600'>⟳ {icon} {label} — running…</span>"
            )
        else:
            lines.append(f"<span style='color:#9CA3AF'>○ {icon} {label}</span>")
    return "<br>".join(lines)


def _parse_routing(content: str) -> list[str]:
    """Extract the 'Routing to: <agents>' list from a Supervisor message."""
    match = re.search(r"Routing to:\s*([^.\n]+)", content)
    if not match:
        return []
    return [x.strip() for x in match.group(1).split(",") if x.strip()]


def _empty_state(icon: str, title: str, body: str) -> None:
    """Friendly placeholder for empty sections."""
    with st.container(border=True):
        st.markdown(
            f"<div style='text-align:center;padding:2.2rem 1rem'>"
            f"<div style='font-size:2.6rem'>{icon}</div>"
            f"<div style='font-size:1.05rem;font-weight:600;margin-top:.4rem'>{title}</div>"
            f"<div style='color:{_MUTED};font-size:0.85rem'>{body}</div></div>",
            unsafe_allow_html=True,
        )


def _metric_card(
    label: str, value: str, help_text: str, active: bool = False
) -> None:
    active_badge = (
        "<span style='display:inline-block;width:8px;height:8px;border-radius:50%;"
        "background:#22C55E;margin-right:6px;"
        "box-shadow:0 0 0 3px rgba(34,197,94,.25)'></span>"
        if active
        else ""
    )
    border_color = "#16A34A" if active else _BORDER
    st.markdown(
        f"<div style='background:#fff;border:1px solid {border_color};border-radius:12px;"
        f"padding:1rem .8rem'>"
        f"<div style='color:{_MUTED};font-size:.78em;text-transform:uppercase;letter-spacing:.04em'>"
        f"{active_badge}{label}</div>"
        f"<div style='font-size:1.7em;font-weight:700;color:{_PRIMARY}'>{value}</div>"
        f"<div style='color:#9CA3AF;font-size:.75em'>{help_text}</div></div>",
        unsafe_allow_html=True,
    )


# ── State helpers ─────────────────────────────────────────────────────


def _restore_state(session_id: str) -> SmartHireState:
    """Load the persisted LangGraph state for a session, if any.

    Reads the latest checkpointer checkpoint (thread_id == session id) so a
    restart continues the previous conversation instead of losing it.
    """
    try:
        checkpoint = get_checkpointer().get_tuple(get_thread_config(session_id))
        if checkpoint is not None:
            values = dict(checkpoint.checkpoint.get("channel_values") or {})
            if values:
                return SmartHireState(**values)
    except Exception:
        logger.exception("Failed to restore state for session %s", session_id)
    return SmartHireState(conversation_history=[])


def _load_past_session(session_id: str) -> None:
    """Switch the app to a previously saved session (chat + ranking results).

    Restores the human-readable transcripts (per recruiter/candidate mode)
    from ``chat_messages`` and the agent outputs (rankings, slots, final
    response) from the LangGraph checkpointer, then reruns so the chosen
    session is rendered.
    """
    audit = ChatAudit()
    st.session_state.session_id = session_id
    audit.save_session_id(session_id)
    row = Database().fetch_one(
        "SELECT mode FROM sessions WHERE id = ?", (session_id,)
    )
    mode = "candidate" if row and row["mode"] == "candidate" else "recruiter"
    st.session_state.chats = _load_all_chats(session_id, audit)
    st.session_state.graph_state = _restore_state(session_id)
    _clear_upload_widget_state()
    _restore_session_uploads(session_id)
    st.session_state.show_results = True
    st.session_state.mode_picker = (
        "Candidate" if mode == "candidate" else "Recruiter"
    )
    # Restore paused state if the session was interrupted mid-pipeline
    paused = Database().get_session_paused(session_id)
    if paused:
        st.session_state.paused_at_node = paused["paused_at_node"]
        st.session_state.paused_error = paused["error_message"]
    else:
        st.session_state.pop("paused_at_node", None)
        st.session_state.pop("paused_error", None)


def _restore_session_uploads(session_id: str) -> None:
    """Restore retained source documents so Screening stays usable per session."""
    resumes: list[dict] = []
    job_description: dict | None = None
    for row in Database().get_session_uploads(session_id):
        record = {
            "name": row["filename"],
            "text": row["extracted_text"],
            "content": bytes(row["content"]),
            "mime": row["mime_type"] or "application/octet-stream",
        }
        if row["kind"] == "resume":
            resumes.append(record)
        elif row["kind"] == "job_description":
            job_description = record

    # Older sessions predate upload retention. Their checkpoint still holds
    # extracted text, so keep that useful fallback and make it downloadable.
    if not resumes:
        resumes = [
            {
                "name": item.get("name") or "resume.txt",
                "text": item.get("text", ""),
                "content": str(item.get("text", "")).encode("utf-8"),
                "mime": "text/plain",
            }
            for item in st.session_state.graph_state.get("resume_inputs", [])
            if item.get("text")
        ]
    if job_description is None:
        raw_text = st.session_state.graph_state.get("job_description", {}).get("raw_text", "")
        if raw_text:
            job_description = {
                "name": "job-description.txt",
                "text": raw_text,
                "content": raw_text.encode("utf-8"),
                "mime": "text/plain",
            }
    st.session_state.resume_texts = resumes
    st.session_state.jd_data = job_description
    st.session_state.pop("jd_paste_text", None)


def _clear_upload_widget_state() -> None:
    """Remove browser widget values before switching the underlying session."""
    for key in list(st.session_state):
        if key in {"resume_uploader", "jd_uploader", "jd_source", "jd_paste_text", "jd_text_preview"} or key.startswith("resume_preview_"):
            st.session_state.pop(key, None)


def _load_all_chats(session_id: str, audit: ChatAudit) -> dict[str, list[dict]]:
    """Load the candidate and recruiter transcripts for a session.

    Candidate and recruiter chats are independent transcripts persisted under
    the same session id (keyed by ``mode``), so switching modes never mixes
    the two conversations.
    """
    return {
        "candidate": audit.load_messages(session_id, mode="candidate"),
        "recruiter": audit.load_messages(session_id, mode="recruiter"),
    }


def _active_chat() -> list[dict]:
    """Return the chat transcript list for the current mode."""
    role = _current_role()
    return st.session_state.setdefault("chats", {}).setdefault(role, [])


def _reset_current_session() -> None:
    """Clear only the active session's workspace, keeping its identifier."""
    session_id = st.session_state.session_id
    audit = ChatAudit()
    mode = "candidate" if st.session_state.mode_picker == "Candidate" else "recruiter"
    ConversationMemory().reset_session(session_id, mode)
    audit.clear(session_id)
    Database().delete_session_uploads(session_id)
    Database().clear_session_paused(session_id)
    _clear_upload_widget_state()
    st.session_state.chats = {"candidate": [], "recruiter": []}
    st.session_state.graph_state = SmartHireState(conversation_history=[])
    st.session_state.resume_texts = []
    st.session_state.jd_data = None
    st.session_state.show_results = False
    st.session_state.sched = None
    st.session_state.recruiter_tab = "Resume Screening"
    st.session_state.pop("paused_at_node", None)
    st.session_state.pop("paused_error", None)
    audit.save_session_id(session_id)


def _clear_all_sessions() -> None:
    """Remove every saved session, upload, workflow result, and chat record."""
    ChatAudit().clear_session_id()
    ConversationMemory().clear_all_sessions()
    _start_new_session()
    st.session_state.flash = "All saved sessions and their uploaded files were cleared."


def _start_new_session() -> None:
    """Create an explicitly requested blank session and switch to it."""
    audit = ChatAudit()
    mode = "candidate" if st.session_state.mode_picker == "Candidate" else "recruiter"
    session_id = ConversationMemory().create_session(mode)
    st.session_state.session_id = session_id
    audit.save_session_id(session_id)
    _clear_upload_widget_state()
    st.session_state.chats = {"candidate": [], "recruiter": []}
    st.session_state.graph_state = SmartHireState(conversation_history=[])
    st.session_state.resume_texts = []
    st.session_state.jd_data = None
    st.session_state.show_results = False
    st.session_state.sched = None
    st.session_state.recruiter_tab = "Resume Screening"
    st.session_state.pop("paused_at_node", None)
    st.session_state.pop("paused_error", None)


# ── Session init ──────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    audit = ChatAudit()
    # Resume the mirrored session id, else the most recent persisted session,
    # else start fresh — so conversation survives browser refreshes.
    sid = audit.restore_session_id() or ConversationMemory().resume_last_session()
    if not sid:
        sid = ConversationMemory().create_session()
    audit.save_session_id(sid)
    st.session_state.session_id = sid
if "graph_state" not in st.session_state:
    st.session_state.graph_state = _restore_state(st.session_state.session_id)
if "chats" not in st.session_state:
    st.session_state.chats = _load_all_chats(
        st.session_state.session_id, ChatAudit()
    )
if "resume_texts" not in st.session_state:
    _restore_session_uploads(st.session_state.session_id)
if "jd_data" not in st.session_state:
    st.session_state.jd_data = None
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "mode_picker" not in st.session_state:
    row = Database().fetch_one(
        "SELECT mode FROM sessions WHERE id = ?", (st.session_state.session_id,)
    )
    st.session_state.mode_picker = (
        "Candidate" if row and row["mode"] == "candidate" else "Recruiter"
    )
if "recruiter_tab" not in st.session_state:
    st.session_state.recruiter_tab = "Resume Screening"
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")


# ── Environment helpers ───────────────────────────────────────────────


def _check_llm() -> bool:
    """Validate the selected model provider without exposing credentials."""
    provider = st.session_state.get("llm_provider") or "ollama"
    if provider == "gemini":
        return bool(st.session_state.get("gemini_api_key", "").strip())
    import urllib.error
    import urllib.request

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        req = urllib.request.Request(
            f"{base_url}/api/tags",
            headers={"Accept": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _extract_pdf_text(uploaded_file) -> str:
    """Extract text from an uploaded PDF."""
    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _extract_docx_text(uploaded_file) -> str:
    """Extract paragraph text from an uploaded DOCX document."""
    from docx import Document

    document = Document(uploaded_file)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _read_uploaded_document(uploaded_file) -> dict:
    """Keep the original bytes and extract text once for workflow/session use."""
    content = uploaded_file.getvalue()
    filename = uploaded_file.name
    stream = io.BytesIO(content)
    if filename.lower().endswith(".pdf"):
        text = _extract_pdf_text(stream)
    elif filename.lower().endswith(".docx"):
        text = _extract_docx_text(stream)
    else:
        text = content.decode("utf-8")
    return {
        "name": filename,
        "text": text,
        "content": content,
        "mime": uploaded_file.type or "application/octet-stream",
    }


def _iter_pipeline(user_input: str):
    """Stream the graph, yielding (event_type, node_name) pairs."""
    from graph import run_smarthire_stream

    state = dict(st.session_state.graph_state)
    return run_smarthire_stream(
        user_input=user_input,
        state=state,
        session_id=st.session_state.session_id,
    )


def _iter_resume_pipeline():
    """Stream the graph resuming from the last checkpoint, yielding (event_type, node_name)."""
    from graph import resume_run

    return resume_run(st.session_state.session_id)


def _show_paused_state(node_name: str, error_msg: str) -> None:
    """Display the paused state with Resume and Retry buttons."""
    state = st.session_state.graph_state
    # Build the list of completed agents from conversation history
    history = state.get("conversation_history", [])
    completed_names = []
    for msg in history:
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.startswith("["):
            # Extract agent name from "[AgentName] ..." messages
            bracket_end = content.index("]") if "]" in content else -1
            if bracket_end > 1:
                agent_tag = content[1:bracket_end].strip().lower().replace(" ", "_")
                if agent_tag in AGENTS:
                    completed_names.append(agent_tag)

    completed_str = (
        ", ".join(_agent_label(a) for a in completed_names)
        if completed_names
        else "None"
    )

    st.warning(
        f"⚠️ Provider is experiencing high demand.\n\n"
        f"**Completed:** {completed_str}\n\n"
        f"**Paused at:** {_agent_label(node_name)}\n\n"
        f"**Error:** {error_msg}",
    )

    col_resume, col_retry = st.columns(2)
    with col_resume:
        if st.button("🔄 Resume", type="primary", use_container_width=True, key="resume_btn"):
            _run_resume_with_progress()
            st.rerun()
    with col_retry:
        if st.button("🔁 Retry from start", type="secondary", use_container_width=True, key="retry_btn"):
            st.session_state.graph_state = _restore_state(st.session_state.session_id)
            st.session_state.pending_input = st.session_state.get("last_user_input", "")
            _run_with_progress("Running multi-agent pipeline…")
            st.rerun()


def _run_resume_with_progress() -> None:
    """Resume a paused pipeline from the last checkpoint, with progress UI."""
    from graph import TransientError

    try:
        workflow = st.session_state.graph_state.get("requested_workflow")
        with st.status("Resuming pipeline…", expanded=True) as status:
            status.update(label="Resuming from last checkpoint…", state="running")
            progress_ph = st.empty()
            completed: list[str] = []
            running: str | None = None
            for event_type, node in _iter_resume_pipeline():
                if event_type == "task":
                    running = node
                    status.update(label=f"Running: {_agent_label(node)}", state="running")
                else:
                    completed.append(node)
                    running = None
                progress_ph.markdown(
                    _stage_checklist(completed, running, workflow), unsafe_allow_html=True
                )
            status.update(label="✓ Pipeline complete", state="complete", expanded=False)
            progress_ph.markdown(_stage_checklist(completed, None, workflow), unsafe_allow_html=True)
    except TransientError as exc:
        st.session_state.graph_state = _restore_state(st.session_state.session_id)
        _show_paused_state(exc.node_name, exc.message)
        return
    except (ValueError, RuntimeError, KeyError) as exc:
        st.error(f"Resume failed: {exc}")
        return

    # Resume completed successfully — clear paused state
    Database().clear_session_paused(st.session_state.session_id)
    st.session_state.pop("paused_at_node", None)
    st.session_state.pop("paused_error", None)
    st.session_state.graph_state = _restore_state(st.session_state.session_id)


# ── Incremental pipeline rendering ────────────────────────────────────


def _iter_pipeline_updates(user_input: str, screening_progress_cb=None):
    """Stream the graph yielding (node_name, state_update) per node.

    Uses ``stream_mode="updates"`` so each completed node's output is
    available immediately for incremental rendering.  ``screening_progress_cb``
    is forwarded to the Resume Screening node and fires as each resume finishes
    so the UI can render progressive results live.
    """
    from graph import run_smarthire_stream_updates

    state = dict(st.session_state.graph_state)
    return run_smarthire_stream_updates(
        user_input=user_input,
        state=state,
        session_id=st.session_state.session_id,
        screening_progress_cb=screening_progress_cb,
    )


def _retry_screening(filename: str) -> None:
    """Button callback: re-screen a single failed resume, then rerun.

    Runs one standalone async screening for the failed document, moves it out
    of ``screening_failures`` (and into ``resumes`` on success), persists the
    corrected state, and reruns so the dropdown reflects the retry.
    """
    if not filename:
        st.warning("Cannot retry: the resume file name is unknown.")
        return
    from graph import persist_graph_state, rescreen_single_resume

    with st.spinner(f"Re-screening {filename}…"):
        result, error = rescreen_single_resume(
            st.session_state.session_id, filename
        )

    state = dict(st.session_state.graph_state)
    state.setdefault("resumes", [])
    state.setdefault("screening_failures", [])
    state["screening_failures"] = [
        f for f in state["screening_failures"] if f.get("filename") != filename
    ]
    if result is not None:
        state["resumes"].append(result)
    else:
        state["screening_failures"].append({
            "candidate_name": filename or "Unknown resume",
            "screening_status": "failed",
            "error": error or "Screening failed",
            "filename": filename,
        })
    st.session_state.graph_state = state
    persist_graph_state(st.session_state.session_id, dict(state))
    st.rerun()


def _render_resume_screening_section(
    resumes: list[dict], failures: list[dict] | None = None
) -> None:
    """Render Resume Screening results with a per-candidate detail view.

    Failed documents stay visible inline in the dropdown with a red
    "Screening failed — retry" state and their own retry button — they are
    never silently dropped from the batch.
    """
    failures = failures or []
    total = len(resumes) + len(failures)
    st.markdown("### 📄 Resume Screening Results")
    failure_note = (
        f" {len(failures)} failed and need attention." if failures else ""
    )
    st.caption(
        f"{total} resume(s) screened. Select a candidate below to view details."
        f"{failure_note}"
    )

    if not resumes and not failures:
        _empty_state("📄", "No resumes screened", "Upload resumes to begin.")
        return

    # Dropdown = successful candidates first, then inline failed entries.
    options: list[tuple[str, dict, bool]] = []
    for r in resumes:
        options.append((r.get("candidate_name", f"Candidate {len(options) + 1}"), r, False))
    for f in failures:
        label = f.get("filename") or f.get("candidate_name") or "Unknown resume"
        options.append((f"{label} — Screening failed", f, True))

    names = [o[0] for o in options]
    selected = st.selectbox(
        "Select candidate",
        names,
        key="screening_candidate_select",
        label_visibility="collapsed",
    )
    idx = names.index(selected) if selected in names else 0
    entry, is_failed = options[idx][1], options[idx][2]

    if is_failed:
        with st.container(border=True):
            st.markdown("#### 🔴 Screening failed")
            st.error(entry.get("error", "Screening failed"))
            st.caption(
                "This resume could not be screened. Fix the file or retry the "
                "screening for just this resume — the rest of the batch is unaffected."
            )
            if st.button(
                "🔁 Retry screening",
                key=f"retry_screening_{entry.get('filename', idx)}",
                use_container_width=True,
            ):
                _retry_screening(entry.get("filename"))
        return

    with st.container(border=True):
        st.markdown(f"#### {entry.get('candidate_name', 'Unknown')}")

        skills = entry.get("skills", [])
        if skills:
            st.markdown("**Skills:**")
            skill_tags = " ".join(
                f"<span style='background:#E0F2FE;color:#0369A1;padding:2px 8px;"
                f"border-radius:10px;font-size:0.82em;margin:2px;display:inline-block'>"
                f"{s}</span>"
                for s in skills
            )
            st.markdown(skill_tags, unsafe_allow_html=True)

        education = entry.get("education", [])
        if education:
            st.markdown("**Education:**")
            for edu in education:
                if isinstance(edu, dict):
                    degree = edu.get("degree", "")
                    inst = edu.get("institution", "")
                    year = edu.get("year", "")
                    st.caption(f"• {degree} — {inst} ({year})" if year else f"• {degree} — {inst}")

        summary = entry.get("summary", "")
        if summary:
            st.markdown("**Screening Summary:**")
            st.info(summary)

        extracted = entry.get("extracted_fields", {})
        certs = extracted.get("certifications", [])
        roles = extracted.get("past_roles", [])
        if certs:
            st.caption(f"**Certifications:** {', '.join(certs)}")
        if roles:
            with st.expander("Past roles"):
                for role in roles:
                    st.caption(f"• {role}")


def _render_matching_section(rankings: list[dict]) -> None:
    """Render Candidate Matching / ranking results."""
    st.markdown("### 🎯 Candidate Rankings")
    if not rankings:
        _empty_state("🎯", "No rankings yet", "Candidate Matching is processing…")
        return
    st.markdown(
        f"**{len(rankings)} candidate(s) ranked.** Click *Schedule Interview* "
        "on any row to jump to Interview Scheduling."
    )
    _display_rankings(rankings)


def _render_scheduling_section(slots: list[dict]) -> None:
    """Render Interview Scheduling results."""
    _display_interview_slots(slots)


def _render_reflection_section(notes: dict) -> None:
    """Render Reflection Node output."""
    _render_reflection_summary(notes)


# ── Staged pipeline runner ────────────────────────────────────────────


def _render_screening_live(placeholder, entries: list[dict], total: int) -> None:
    """Live, mid-run progress view of a screening batch.

    Streamlit renders this placeholder repeatedly as each resume finishes, so
    the recruiter sees a growing dropdown (successes + failures) and a live
    "Screening X of Y resumes…" count instead of one spinner for the whole
    stage.  It is read-only — the interactive per-candidate detail view
    renders after the pipeline completes.
    """
    placeholder.empty()
    done = len(entries)
    with placeholder.container():
        st.markdown(f"**Screening {done} of {total} resumes…**")
        names: list[str] = []
        for entry in entries:
            if entry.get("screening_status") == "failed":
                label = entry.get("filename") or entry.get("candidate_name") or "Unknown resume"
                names.append(f"🔴 {label} — Screening failed")
            else:
                score = entry.get("match_score")
                pct = f" — {score:.0f}% match" if isinstance(score, (int, float)) else ""
                names.append(f"🟢 {entry.get('candidate_name') or 'Unknown'}{pct}")
        if names:
            st.selectbox(
                "Screened so far",
                names,
                key="screening_live_select",
                label_visibility="collapsed",
            )
        remaining = total - done
        if remaining:
            st.caption(f"{remaining} resume(s) still screening…")


def _run_with_progress(progress_title: str) -> None:
    """Run the pipeline with incremental rendering of each agent's results.

    Streams the graph using ``stream_mode="updates"``.  After each node
    completes, its output is stored in ``st.session_state.graph_state`` and
    rendered immediately below the progress indicator — so Resume Screening
    results appear before Candidate Matching starts.

    While the Resume Screening node fans out in parallel, the screening
    progress callback updates the status area live ("Screening 4 of 7
    resumes…") and repopulates the results dropdown as each resume finishes,
    instead of showing one spinner for the whole stage.
    """
    from graph import TransientError

    completed: list[str] = []
    running: str | None = None
    workflow = st.session_state.graph_state.get("requested_workflow")

    try:
        with st.status(progress_title, expanded=True) as status:
            status.update(label="Starting agents…", state="running")
            progress_ph = st.empty()
            screening_ph = st.empty()

            def _on_screening_progress(done, total, name, result, error):
                progress_ph.markdown(
                    f"**Screening {done} of {total} resumes…**",
                    unsafe_allow_html=True,
                )
                if result is None and error is None:
                    return
                live = list(st.session_state.get("screening_live", []))
                if result is not None:
                    live.append(
                        result.model_dump()
                        if hasattr(result, "model_dump")
                        else result
                    )
                else:
                    live.append({
                        "candidate_name": name or "Unknown resume",
                        "screening_status": "failed",
                        "error": str(error),
                        "filename": name,
                    })
                st.session_state["screening_live"] = live
                _render_screening_live(screening_ph, live, total)

            for node_name, update in _iter_pipeline_updates(
                st.session_state.pending_input,
                screening_progress_cb=_on_screening_progress,
            ):
                # Mark this node as completed
                if node_name not in completed:
                    completed.append(node_name)
                running = None

                # Merge the update into graph_state
                state = dict(st.session_state.graph_state)
                for key, value in update.items():
                    if key == "conversation_history":
                        # Append reducer: extend, don't replace
                        state.setdefault("conversation_history", []).extend(value)
                    elif isinstance(state.get(key), list) and isinstance(value, list):
                        # Other list fields: extend if both are lists
                        state.setdefault(key, []).extend(value)
                    else:
                        state[key] = value
                st.session_state.graph_state = state

                # Update progress indicator
                status.update(
                    label=f"✓ {_agent_label(node_name)} complete",
                    state="running",
                )
                progress_ph.markdown(
                    _stage_checklist(completed, running, workflow), unsafe_allow_html=True
                )

            status.update(label="✓ Pipeline complete", state="complete", expanded=False)
            progress_ph.markdown(_stage_checklist(completed, None, workflow), unsafe_allow_html=True)
            screening_ph.empty()
            st.session_state.pop("screening_live", None)

    except TransientError as exc:
        st.session_state.graph_state = _restore_state(st.session_state.session_id)
        st.session_state.pop("screening_live", None)
        _show_paused_state(exc.node_name, exc.message)
        return
    except (ValueError, RuntimeError, KeyError) as exc:
        st.session_state.pop("screening_live", None)
        msg = str(exc)
        if "api_key" in msg.lower() or "auth" in msg.lower() or "permission" in msg.lower():
            st.error(f"Gemini authentication failed: {msg}")
        elif "quota" in msg.lower() or "limit" in msg.lower() or "429" in msg:
            st.error(f"Gemini rate limit exceeded: {msg}")
        elif "network" in msg.lower() or "timeout" in msg.lower() or "connect" in msg.lower():
            st.error(f"Network error reaching Gemini: {msg}")
        else:
            st.error(f"Pipeline error: {msg}")
        return

    Database().clear_session_paused(st.session_state.session_id)
    st.session_state.pop("paused_at_node", None)
    st.session_state.pop("paused_error", None)
    st.session_state.graph_state = _restore_state(st.session_state.session_id)


def _render_stage(node_name: str, state: dict) -> None:
    """Render the output section for a completed pipeline stage."""
    if node_name == "resume_screening":
        resumes = state.get("resumes", [])
        failures = state.get("screening_failures", [])
        if resumes or failures:
            _render_resume_screening_section(resumes, failures)
            st.divider()
    elif node_name == "candidate_matching":
        rankings = state.get("candidate_rankings", [])
        if rankings:
            _render_matching_section(rankings)
            st.divider()
    elif node_name == "interview_scheduling":
        slots = state.get("interview_slots", [])
        if slots:
            _render_scheduling_section(slots)
            st.divider()
    elif node_name == "reflection":
        notes = state.get("reflection_notes", {})
        if notes:
            _render_reflection_section(notes)
            final = state.get("final_response", "")
            if final:
                st.info(f"**Summary:** {final}")


def _schedule_candidate(name: str) -> None:
    """Button callback: preselect a candidate and jump to the scheduling tab.

    Runs before widgets instantiate, so writing widget-keyed session state
    (``recruiter_tab``) is allowed here.
    """
    st.session_state.schedule_candidate = name
    st.session_state.recruiter_tab = "Interview Scheduling"


def _render_reflection_summary(notes: dict) -> None:
    """Surface the Reflection Node's structured validation (Handoff C 2c).

    Shows the overall reflection_validated status, every check with its
    pass/fail result, any corrections made, and whether a correction retry
    was triggered.
    """
    validated = notes.get("reflection_validated", notes.get("validation_passed"))
    with st.expander("🧪 Reflection summary", expanded=False):
        if validated is True:
            st.markdown("✅ **Reflection validated** — all checks passed.")
        elif validated is False:
            st.markdown("⚠️ **Reflection flagged issues** — review the checks below.")
        else:
            st.markdown("_No reflection results yet._")
            return

        if notes.get("correction_attempted"):
            st.caption(
                f"🔄 A correction pass was triggered on "
                f"**{notes.get('retry_agent')}** before this result was returned."
            )

        checks = notes.get("checks", [])
        if checks:
            for c in checks:
                icon = "✅" if c.get("passed") else "⚠️"
                st.markdown(f"{icon} **{c.get('name', c.get('check'))}**")
                for issue in c.get("issues", []):
                    st.markdown(f"  - ⚠️ {issue}")
                for corr in c.get("corrections", []):
                    st.markdown(f"  - ✅ {corr}")
        else:
            issues = notes.get("issues_found", [])
            if issues:
                st.markdown("**Issues:**")
                for issue in issues:
                    st.markdown(f"- ⚠️ {issue}")

        corrections = notes.get("corrections_made", [])
        if corrections:
            st.markdown("**Corrections made:**")
            for corr in corrections:
                st.markdown(f"- ✅ {corr}")


def _display_rankings(rankings: list[dict]) -> None:
    """Render ranked candidates as cards with score meters + validation."""
    notes = st.session_state.graph_state.get("reflection_notes", {})
    issues = notes.get("issues_found", [])
    checks = notes.get("checks_run", [])

    for r in rankings:
        name = r.get("candidate_name", "Unknown")
        rank = r.get("rank", "")
        score = r.get("match_score", 0)
        with st.container(border=True):
            top = st.columns([3, 1])
            with top[0]:
                st.markdown(f"**#{rank} {name}**")
                st.markdown(r.get("justification", "") or "_No justification provided._")
            with top[1]:
                st.markdown(_score_meter(score), unsafe_allow_html=True)

            experience_match = r.get("experience_match")
            if experience_match is not None:
                st.caption(f"Experience match: {'✅ Yes' if experience_match else '⚠️ No'}")

            skills_match = r.get("skills_match", [])
            skills_gap = r.get("skills_gap", [])
            if skills_match:
                st.markdown(f"**Top strengths:** {', '.join(skills_match[:6])}")
            if skills_gap:
                st.markdown(f"**Top gaps:** {', '.join(skills_gap[:4])}")

            with st.expander("Why this ranking?"):
                candidate_issues = [
                    i for i in issues if name.lower() in str(i).lower()
                ]
                if candidate_issues:
                    st.markdown("**Reflection Node flags:**")
                    for issue in candidate_issues:
                        st.markdown(f"- ⚠️ {issue}")
                else:
                    st.markdown(
                        "✅ **Reflection validated** — no issues flagged for this "
                        "candidate against the job description."
                    )
                if checks:
                    st.caption("Checks run: " + ", ".join(checks))

            st.button(
                "Schedule Interview",
                key=f"schedule_{rank}_{name}",
                use_container_width=True,
                on_click=_schedule_candidate,
                args=(name,),
            )


def _display_interview_slots(slots: list[dict]) -> None:
    """Render proposed interview slots from a full pipeline run."""
    if not slots:
        return
    st.subheader("Proposed Interview Slots")
    for s in slots:
        status = s.get("status", "proposed")
        icon = "✅" if status == "confirmed" else "⚠️"
        st.write(
            f"{icon} **{s.get('candidate_name', '?')}** — "
            f"{s.get('date', '?')} {s.get('time_start', '?')}-{s.get('time_end', '?')} "
            f"with {s.get('interviewer', '?')} ({s.get('interview_type', '?')})"
        )


# ── Chat rendering (agent-aware) ──────────────────────────────────────


# Suggested questions shown above the chat input, per role.
# Candidate chat is anonymous, so only GENERIC process questions are offered —
# never personal questions like "What is my application status?" which this
# chat cannot truthfully answer.
_CANDIDATE_SUGGESTIONS = [
    "Explain the hiring process.",
    "How many interview rounds are there?",
    "How long is the interview?",
    "Is my interview online or onsite?",
    "What happens after the interview?",
    "How should I prepare?",
    "What documents are required?",
    "What skills are evaluated?",
    "How long does the hiring process take?",
    "Can I reschedule my interview?",
    "Who should I contact?",
]

_RECRUITER_SUGGESTIONS = [
    "Rank candidates",
    "Screen resumes",
    "Compare candidates",
    "Schedule interviews",
    "Hiring insights",
    "Candidate summary",
]


def _current_role() -> str:
    """Return 'candidate' or 'recruiter' from the active mode picker."""
    return (
        "candidate" if st.session_state.mode_picker == "Candidate" else "recruiter"
    )


def _render_suggestions(role: str) -> None:
    """Render clickable suggested questions matching the user's role."""
    questions = _CANDIDATE_SUGGESTIONS if role == "candidate" else _RECRUITER_SUGGESTIONS
    st.caption("Try asking:")
    cols = st.columns(3)
    for i, question in enumerate(questions):
        with cols[i % 3]:
            if st.button(
                question,
                key=f"suggestion_{role}_{i}",
                use_container_width=True,
            ):
                st.session_state.suggestion_input = question
                st.rerun()


def _render_chat(messages: list[dict]) -> None:
    """Render the transcript, badge-labelling each agent turn."""
    routing: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                if routing:
                    labels = ", ".join(_agent_label(a) for a in routing)
                    st.caption(f"🛰 Routed to: {labels}")
                st.markdown(content)
            routing = []
        elif role == "agent:supervisor":
            routed = _parse_routing(content)
            if routed:
                routing = routed
            labels = ", ".join(_agent_label(a) for a in routed) if routed else "—"
            st.caption(f"🧭 **Supervisor** → routed to {labels}")
        elif role.startswith("agent:"):
            agent_name = msg.get("agent_name") or role.split(":", 1)[1]
            with st.chat_message("assistant"):
                st.markdown(_agent_badge(agent_name), unsafe_allow_html=True)
                st.markdown(content)


def _chat_section(input_placeholder: str) -> None:
    """Shared Chat UI used by both the Chat tab and Candidate Chat mode."""
    role = _current_role()
    chat_messages = _active_chat()

    # Queue the input, process it, then rerun.  This keeps the response in
    # the transcript before the FAQ buttons instead of rendering an answer
    # beneath suggestions (the inconsistent layout shown in the screenshots).
    prompt = st.session_state.pop("pending_chat_prompt", None)
    if not prompt:
        prompt = st.session_state.pop("suggestion_input", None)

    if prompt:
        audit = ChatAudit()
        chat_messages.append({"role": "user", "content": prompt})

        if not _llm_ok:
            _prov = (st.session_state.get("llm_provider") or "ollama").strip().lower()
            if _prov == "gemini":
                answer = "Add a Gemini API key in the sidebar to continue."
            else:
                answer = "Ollama is not reachable. Start `ollama serve` to continue."
            audit.log_turn(
                st.session_state.session_id,
                user_content=prompt,
                answer=answer,
                mode=role,
            )
        else:
            if role == "candidate":
                # Candidate chat is generic KB Q&A — answer directly from the
                # HR Assistant. Never run the recruiter pipeline here.
                from graph import answer_candidate_query

                prior_answers = [
                    m["content"]
                    for m in chat_messages
                    if m["role"] == "assistant"
                ][-5:]
                try:
                    answer = answer_candidate_query(
                        prompt,
                        st.session_state.session_id,
                        {"prior_answers": prior_answers},
                    )
                except Exception:
                    logger.exception("Candidate chat failed")
                    answer = (
                        "Sorry, I couldn't process that request right now. "
                        "Please try again in a moment."
                    )
                audit.log_turn(
                    st.session_state.session_id,
                    user_content=prompt,
                    answer=answer,
                    mode=role,
                )
            else:
                # Recruiter chat answers directly from stored session results
                # (screening/ranking/scheduling/insights) or the HR knowledge
                # base. It never re-runs the multi-agent pipeline — results
                # come from the 'Screen & Rank Candidates' workflow.
                from graph import answer_recruiter_chat

                prior_answers = [
                    m["content"]
                    for m in chat_messages
                    if m["role"] == "assistant"
                ][-5:]
                try:
                    answer = answer_recruiter_chat(
                        prompt,
                        st.session_state.session_id,
                        dict(st.session_state.graph_state),
                        {"prior_answers": prior_answers},
                    )
                except Exception:
                    logger.exception("Recruiter chat failed")
                    answer = (
                        "Sorry, I couldn't process that request right now. "
                        "Please try again in a moment."
                    )
                audit.log_turn(
                    st.session_state.session_id,
                    user_content=prompt,
                    answer=answer,
                    mode=role,
                )

        chat_messages.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()

    _render_chat(chat_messages)
    _render_suggestions(role)

    typed_prompt = st.chat_input(input_placeholder)
    if typed_prompt:
        st.session_state.pending_chat_prompt = typed_prompt
        st.rerun()


# ── Interview scheduling helpers ──────────────────────────────────────


def _candidate_options() -> list[str]:
    names = [
        r.get("candidate_name")
        for r in st.session_state.graph_state.get("candidate_rankings", [])
        if r.get("candidate_name")
    ]
    return list(dict.fromkeys(names))


def _propose_slots(candidate: str, date: str, interviewer: str) -> list[dict]:
    """Build a day grid of slots, flagging conflicts with the calendar.

    Uses the same CalendarTool engine as the Interview Scheduling Agent; a
    slot conflicts only when it overlaps an existing booking.
    """
    calendar = CalendarTool()
    hours = [
        ("09:00", "10:00"),
        ("10:00", "11:00"),
        ("11:00", "12:00"),
        ("13:00", "14:00"),
        ("14:00", "15:00"),
        ("15:00", "16:00"),
    ]
    types = ["technical", "behavioral", "phone", "technical", "behavioral", "phone"]
    slots: list[dict] = []
    for i, (time_start, time_end) in enumerate(hours):
        conflicts = not calendar.check_availability(
            date, time_start, time_end, interviewer
        )
        slots.append(
            {
                "date": date,
                "time_start": time_start,
                "time_end": time_end,
                "interviewer": interviewer,
                "interview_type": types[i],
                "conflict": conflicts,
                "confirmed": False,
            }
        )
    return slots


def _confirm_slot(slot: dict, candidate: str, idx: int) -> None:
    """Book a slot via the calendar + persist to SQLite, then rerun."""
    calendar = CalendarTool()
    result = calendar.book_slot(
        InterviewSlot(
            candidate_name=candidate,
            date=slot["date"],
            time_start=slot["time_start"],
            time_end=slot["time_end"],
            interviewer=slot["interviewer"],
            interview_type=slot["interview_type"],
            status="proposed",
        )
    )
    if result["status"] != "confirmed":
        st.session_state.flash = (
            f"That slot for {candidate} was just taken — please pick another."
        )
        st.rerun()

    db = Database()
    db.insert_interview(
        candidate_id=db.find_candidate_by_name(candidate),
        jd_id=db.get_recent_job_description_id(),
        proposed_start=f"{slot['date']} {slot['time_start']}:00",
        proposed_end=f"{slot['date']} {slot['time_end']}:00",
        status="confirmed",
        session_id=st.session_state.session_id,
        interview_type=slot.get("interview_type"),
        interviewer=slot.get("interviewer"),
    )
    sched = st.session_state.sched
    sched["slots"][idx]["confirmed"] = True
    st.session_state.sched = sched
    st.session_state.flash = (
        f"✅ Interview confirmed: {candidate} · {slot['date']} "
        f"{slot['time_start']}-{slot['time_end']} with {slot['interviewer']}."
    )
    st.rerun()


def _render_session_interviews() -> None:
    """Render the list of interviews scheduled in the current session."""
    session_id = st.session_state.get("session_id", "")
    if not session_id:
        _empty_state(
            "📋",
            "No active session",
            "Start a session to see scheduled interviews here.",
        )
        return

    db = Database()
    rows = db.get_interviews_by_session(session_id)

    if not rows:
        _empty_state(
            "📋",
            "No interviews scheduled yet",
            "Go to 'Schedule New' to propose slots for a shortlisted candidate.",
        )
        return

    # Group by date
    by_date: dict[str, list] = defaultdict(list)
    for row in rows:
        start = row["proposed_start"] or ""
        date_str = start[:10] if len(start) >= 10 else "Unknown"
        by_date[date_str].append(row)

    # Sort dates soonest first
    sorted_dates = sorted(by_date.keys())

    for date_str in sorted_dates:
        if len(sorted_dates) > 1:
            st.markdown(f"**{date_str}**")

        for row in by_date[date_str]:
            interview_id = row["id"]
            candidate = row["candidate_name"] or "Unknown"
            interviewer_val = row["interviewer"] or "—"
            interview_type = row["interview_type"] or "—"
            status = row["status"] or "proposed"

            start = row["proposed_start"] or ""
            end = row["proposed_end"] or ""
            time_display = f"{start[11:16]}–{end[11:16]}" if len(start) > 11 and len(end) > 11 else "—"

            # Status badge colours
            if status == "confirmed":
                badge = (
                    "<span style='color:#15803d;font-weight:600'>"
                    "● Confirmed</span>"
                )
            elif status == "cancelled":
                badge = (
                    "<span style='color:#B91C1C;font-weight:600'>"
                    "● Cancelled</span>"
                )
            else:
                badge = (
                    "<span style='color:#A16207;font-weight:600'>"
                    "● Proposed</span>"
                )

            with st.container(border=True):
                cols = st.columns([5, 1])
                with cols[0]:
                    st.markdown(
                        f"**{candidate}** · {interview_type}<br>"
                        f"<small>{interviewer_val} · {time_display}</small><br>"
                        f"{badge}",
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    if status in ("confirmed", "proposed"):
                        if st.button(
                            "Cancel",
                            key=f"cancel_interview_{interview_id}",
                            use_container_width=True,
                        ):
                            db.update_interview_status(interview_id, "cancelled")
                            st.rerun()


def _render_scheduling_tab() -> None:
    st.markdown("### 📅 Interview Scheduling")

    tab_new, tab_session = st.tabs(["Schedule New", "Session Interviews"])

    with tab_new:
        st.caption(
            "Pick a shortlisted candidate and confirm a slot. Grayed slots "
            "conflict with the interviewer's calendar and are disabled."
        )

        candidates = _candidate_options()
        if not candidates:
            _empty_state(
                "🗓️",
                "No candidates to schedule",
                "Screen and rank resumes first, then return here to book interviews.",
            )
        else:
            default_idx = 0
            preselected = st.session_state.pop("schedule_candidate", None)
            if preselected in candidates:
                default_idx = candidates.index(preselected)

            col_c, col_i, col_d = st.columns([2, 2, 2])
            with col_c:
                candidate = st.selectbox(
                    "Shortlisted candidate", candidates, index=default_idx
                )
            with col_i:
                interviewer = st.selectbox(
                    "Interviewer",
                    ["Bob Tech Lead", "Alice Manager", "Carol Director"],
                )
            with col_d:
                date = st.date_input(
                    "Interview date",
                    value=datetime.datetime.now(datetime.UTC).date(),
                ).isoformat()

            if st.button(
                "Propose available slots",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.sched = {
                    "candidate": candidate,
                    "interviewer": interviewer,
                    "date": date,
                    "slots": _propose_slots(candidate, date, interviewer),
                }

            sched = st.session_state.get("sched")
            if (
                sched
                and sched["candidate"] == candidate
                and sched["interviewer"] == interviewer
            ):
                st.divider()
                st.markdown(
                    f"**{sched['candidate']}** with {sched['interviewer']} "
                    f"on {sched['date']}"
                )
                columns = st.columns(3)
                for idx, slot in enumerate(sched["slots"]):
                    with columns[idx % 3]:
                        if slot["conflict"]:
                            with st.container(border=True):
                                st.markdown(
                                    f"<div style='opacity:.55'>"
                                    f"⛔ **{slot['time_start']}–{slot['time_end']}**"
                                    f"<br><small>{slot['interview_type']} · "
                                    f"{slot['interviewer']}</small>"
                                    f"<br><span style='color:#B91C1C'>Conflict</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                        elif slot["confirmed"]:
                            with st.container(border=True):
                                st.markdown(
                                    f"✅ **{slot['time_start']}–{slot['time_end']}**"
                                    f"<br><small>{slot['interview_type']} · "
                                    f"{slot['interviewer']}</small>"
                                    f"<br><span style='color:{_PRIMARY};"
                                    f"font-weight:600'>Confirmed</span>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            with st.container(border=True):
                                st.markdown(
                                    f"**{slot['time_start']}–{slot['time_end']}**"
                                    f"<br><small>{slot['interview_type']} · "
                                    f"{slot['interviewer']}</small>",
                                    unsafe_allow_html=True,
                                )
                                if st.button(
                                    "Confirm interview",
                                    key=f"confirm_slot_{idx}",
                                    use_container_width=True,
                                ):
                                    _confirm_slot(slot, candidate, idx)

    with tab_session:
        _render_session_interviews()


# ── System Insight ────────────────────────────────────────────────────


def _render_insight_tab() -> None:
    st.markdown("### 📊 Session Insight")
    st.caption(
        f"Showing only data from the loaded session "
        f"({st.session_state.session_id[:8]})."
    )
    state = st.session_state.graph_state
    resumes = state.get("resumes", [])
    uploaded_resumes = st.session_state.resume_texts
    rankings = state.get("candidate_rankings", [])
    interviews = state.get("interview_slots", [])

    scores = [r["match_score"] for r in rankings if r["match_score"] is not None]
    avg_score = f"{sum(scores) / len(scores):.0f}%" if scores else "—"

    cols = st.columns(4)
    with cols[0]:
        _metric_card(
            "Resumes in session",
            str(len(resumes) or len(uploaded_resumes)),
            "Uploaded or processed here",
        )
    with cols[1]:
        _metric_card(
            "Active session",
            st.session_state.session_id[:8],
            "Currently loaded session",
            active=True,
        )
    with cols[2]:
        _metric_card("Avg match score", avg_score, "This session's ranking")
    with cols[3]:
        _metric_card("Interviews", str(len(interviews)), "This session's proposed slots")


# ── Upload & Screen tab ───────────────────────────────────────────────


def _render_upload_tab() -> None:
    st.markdown("### 📄 Resume Screening")
    st.caption(
        "Upload resumes and a job description, then extract a grounded screening "
        "profile for each candidate. Ranking happens separately in Candidate Matching."
    )

    col_resume, col_jd = st.columns(2)

    with col_resume, st.container(border=True):
        st.markdown("#### 📄 Resume Files")
        st.caption(
            "Drop PDF or TXT resumes. The **Resume Screening** agent extracts "
            "skills, experience, education, and an initial match score."
        )
        uploaded_resumes = st.file_uploader(
            "Resume files",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
            key="resume_uploader",
            label_visibility="collapsed",
        )
        if uploaded_resumes:
            st.session_state.resume_texts = [
                _read_uploaded_document(uploaded_file)
                for uploaded_file in uploaded_resumes
            ]
            st.success(f"Loaded {len(st.session_state.resume_texts)} resume(s)")
        if st.session_state.resume_texts:
            with st.expander("Review extracted resume text"):
                for r in st.session_state.resume_texts:
                    st.markdown(f"**{r['name']}**")
                    st.caption(f"{len(r['text']):,} characters extracted")
                    st.text_area(
                        "Extracted text",
                        value=r["text"],
                        height=280,
                        disabled=True,
                        key=f"resume_preview_{r['name']}",
                        label_visibility="collapsed",
                    )
            st.markdown("**Saved resume files**")
            for r in st.session_state.resume_texts:
                file_col, download_col = st.columns([3, 2])
                with file_col:
                    st.caption(f"📎 {r['name']}")
                with download_col:
                    st.download_button(
                        "Download",
                        data=r.get("content") or r["text"].encode("utf-8"),
                        file_name=r["name"],
                        mime=r.get("mime", "text/plain"),
                        key=f"resume_download_{r['name']}",
                        use_container_width=True,
                    )
        else:
            _empty_state(
                "📄",
                "No resumes uploaded yet",
                "Add one or more resumes to start screening.",
            )

    with col_jd, st.container(border=True):
        st.markdown("#### 📝 Job Description")
        st.caption(
            "Upload, paste, or use the bundled sample JD for a Senior "
            "Full-Stack Developer."
        )
        jd_option = st.radio(
            "Source",
            ["Upload file", "Paste text", "Use sample JD"],
            horizontal=True,
            key="jd_source",
        )

        stored_jd = st.session_state.jd_data or {}
        jd_record = dict(stored_jd) if stored_jd else None
        jd_text = stored_jd.get("text", "")
        if jd_option == "Upload file":
            jd_file = st.file_uploader(
                "JD file",
                type=["txt", "pdf", "docx"],
                key="jd_uploader",
                label_visibility="collapsed",
            )
            if jd_file:
                jd_record = _read_uploaded_document(jd_file)
                jd_text = jd_record["text"]
        elif jd_option == "Paste text":
            jd_text = st.text_area(
                "Paste job description",
                height=160,
                value=stored_jd.get("text", ""),
                key="jd_paste_text",
                placeholder="Paste the full job description here…",
                label_visibility="collapsed",
            )
            jd_record = {
                "name": "job-description.txt",
                "text": jd_text,
                "content": jd_text.encode("utf-8"),
                "mime": "text/plain",
            }
        else:
            jd_file = Path("data/sample_jd.txt")
            if jd_file.exists():
                jd_text = jd_file.read_text(encoding="utf-8")
                jd_record = {
                    "name": jd_file.name,
                    "text": jd_text,
                    "content": jd_text.encode("utf-8"),
                    "mime": "text/plain",
                }
                st.info("Using sample JD: Senior Full-Stack Developer")

        if jd_text:
            with st.expander("Review extracted job description text"):
                st.caption(f"{len(jd_text):,} characters extracted")
                st.text_area(
                    "Extracted job description",
                    value=jd_text,
                    height=280,
                    disabled=True,
                    key="jd_text_preview",
                    label_visibility="collapsed",
                )
            if jd_record:
                st.markdown("**Saved job description file**")
                file_col, download_col = st.columns([3, 2])
                with file_col:
                    st.caption(f"📎 {jd_record.get('name', 'job-description.txt')}")
                with download_col:
                    st.download_button(
                        "Download",
                        data=jd_record.get("content") or jd_text.encode("utf-8"),
                        file_name=jd_record.get("name", "job-description.txt"),
                        mime=jd_record.get("mime", "text/plain"),
                        key="jd_download",
                        use_container_width=True,
                    )
        else:
            _empty_state(
                "📝",
                "No job description yet",
                "Add a JD so candidates can be scored against requirements.",
            )

    st.divider()

    run_disabled = not (st.session_state.resume_texts and jd_text)
    if run_disabled:
        st.caption("Upload at least one resume and a job description to enable screening.")

    if st.button(
        "🚀 Run Resume Screening",
        type="primary",
        disabled=run_disabled,
        use_container_width=True,
    ):
        if not _llm_ok:
            _prov = (st.session_state.get("llm_provider") or "ollama").strip().lower()
            if _prov == "gemini":
                st.error("Add a Gemini API key in the sidebar to continue.")
            else:
                st.error("Cannot run: Ollama is not reachable. Start `ollama serve`.")
        else:
            # Persist the raw JD so it survives restarts, and thread its id
            # through state so agents can reference it.
            jd_source = {
                "Upload file": "upload",
                "Paste text": "paste",
                "Use sample JD": "sample",
            }.get(jd_option, "paste")
            jd_id = Database().persist_job_description(
                {}, raw_text=jd_text, source=jd_source
            )
            uploads = [
                {
                    "kind": "resume",
                    "filename": resume["name"],
                    "mime_type": resume.get("mime"),
                    "content": resume.get("content") or resume["text"].encode("utf-8"),
                    "extracted_text": resume["text"],
                }
                for resume in st.session_state.resume_texts
            ]
            if jd_record:
                uploads.append(
                    {
                        "kind": "job_description",
                        "filename": jd_record.get("name", "job-description.txt"),
                        "mime_type": jd_record.get("mime", "text/plain"),
                        "content": jd_record.get("content") or jd_text.encode("utf-8"),
                        "extracted_text": jd_text,
                    }
                )
            Database().replace_session_uploads(st.session_state.session_id, uploads)
            st.session_state.jd_data = jd_record
            st.session_state.graph_state = {
                **st.session_state.graph_state,
                "user_role": "recruiter",
                "requested_workflow": "screening",
                "job_description": {
                    "id": jd_id,
                    "title": jd_text.strip().splitlines()[0][:200]
                    if jd_text.strip()
                    else None,
                    "raw_text": jd_text,
                },
                "resume_inputs": list(st.session_state.resume_texts),
                # A new screening run creates a new source snapshot.  Existing
                # rankings must not appear to belong to it.
                "candidate_rankings": [],
                "screening_failures": [],
                "reflection_notes": {},
                "final_response": "",
                "reflection_attempts": 0,
                "retry_agent": None,
                "reflection_feedback": None,
            }
            request = "Run resume screening for the uploaded resumes."
            st.session_state.pending_input = request
            st.session_state.last_user_input = request
            _run_with_progress("Running resume screening…")
            st.session_state.show_results = True

    if st.session_state.get("show_results"):
        st.divider()
        state = st.session_state.graph_state

        # Screening results stay in this workflow. Ranking has its own tab.
        resumes = state.get("resumes", [])
        failures = state.get("screening_failures", [])
        if resumes or failures:
            _render_resume_screening_section(resumes, failures)
            st.info("Screening is complete. Open **Candidate Matching** to compare and rank this batch.")

        errors = state.get("error", "")
        if errors:
            st.error(f"Pipeline error: {errors}")

        # Show paused state if the session was interrupted mid-pipeline
        paused_node = st.session_state.get("paused_at_node")
        if paused_node and state.get("requested_workflow") == "screening":
            _show_paused_state(paused_node, st.session_state.get("paused_error", ""))


def _render_matching_tab() -> None:
    """Render the independent candidate-comparison workflow."""
    st.markdown("### 🎯 Candidate Matching")
    st.caption(
        "Compare the completed screening batch against its job description. "
        "This does not re-read or re-parse the resumes."
    )

    state = st.session_state.graph_state
    resumes = state.get("resumes", [])
    jd_data = state.get("job_description", {})
    if not resumes or not jd_data.get("raw_text"):
        _empty_state(
            "🎯",
            "Screening required",
            "Complete Resume Screening with a job description before ranking candidates.",
        )
        return

    failures = state.get("screening_failures", [])
    if failures:
        _empty_state(
            "⚠️",
            "Finish resume screening first",
            f"{len(failures)} resume(s) failed screening. Resolve them and run screening again before matching.",
        )
        return

    st.success(f"Ready to compare {len(resumes)} screened candidate(s).")
    st.caption(f"Job description: {jd_data.get('title') or 'Uploaded job description'}")

    if st.button("🚀 Rank Screened Candidates", type="primary", use_container_width=True):
        if not _llm_ok:
            provider = (st.session_state.get("llm_provider") or "ollama").strip().lower()
            st.error(
                "Add a Gemini API key in the sidebar to continue."
                if provider == "gemini"
                else "Cannot run: Ollama is not reachable. Start `ollama serve`."
            )
        else:
            st.session_state.graph_state = {
                **state,
                "user_role": "recruiter",
                "requested_workflow": "matching",
                "candidate_rankings": [],
                "reflection_notes": {},
                "final_response": "",
                "reflection_attempts": 0,
                "retry_agent": None,
                "reflection_feedback": None,
            }
            request = "Rank the completed screening batch against the job description."
            st.session_state.pending_input = request
            st.session_state.last_user_input = request
            _run_with_progress("Running candidate matching…")
            st.session_state.show_results = True

    rankings = st.session_state.graph_state.get("candidate_rankings", [])
    if rankings:
        st.divider()
        _render_matching_section(rankings)
        notes = st.session_state.graph_state.get("reflection_notes", {})
        if notes:
            _render_reflection_section(notes)

    paused_node = st.session_state.get("paused_at_node")
    if paused_node and st.session_state.graph_state.get("requested_workflow") == "matching":
        _show_paused_state(paused_node, st.session_state.get("paused_error", ""))


# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🤖 SmartHire AI")
    st.caption("Multi-Agent Recruitment System")

    mode_label = st.segmented_control(
        "Mode",
        options=["Recruiter", "Candidate"],
        key="mode_picker",
        label_visibility="collapsed",
    )
    is_recruiter = mode_label == "Recruiter"

    ConversationMemory().set_mode(
        st.session_state.session_id,
        "recruiter" if is_recruiter else "candidate",
    )

    st.divider()

    with st.expander("Past Sessions"):
        sessions = Database().get_sessions()
        if not sessions:
            st.caption("No past sessions yet.")
        # Fixed order by original creation time (started_at), so loading a
        # session never re-sorts it to the top.
        sessions = sorted(
            sessions,
            key=lambda s: str(s["started_at"] or ""),
            reverse=True,
        )
        for s in sessions:
            sid = s["id"]
            mode_name = s["mode"].capitalize() if s["mode"] else "Unknown"
            timestamp = str(s["started_at"])[:16]
            label = f"{mode_name} · {timestamp} · {sid[:8]}"
            if sid == st.session_state.session_id:
                st.markdown(
                    f"<div style='background:#ECFDF5;border:1px solid #22C55E;"
                    f"border-radius:8px;padding:8px 10px;margin-bottom:6px'>"
                    f"<span style='color:#16A34A;font-weight:700'>● {label} — "
                    f"active</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.button(
                    label,
                    key=f"past_session_{sid}",
                    use_container_width=True,
                    on_click=_load_past_session,
                    args=(sid,),
                )

    st.divider()

    st.markdown("#### Model Provider")
    _provider = st.segmented_control(
        "Provider",
        options=["Ollama", "Gemini"],
        key="llm_provider",
        label_visibility="collapsed",
    )

    _llm_ok = _check_llm()

    if _provider == "Ollama":
        if _llm_ok:
            st.success("Ollama connected", icon="✅")
        else:
            st.error("Ollama is not reachable", icon="❌")
            st.caption("Start Ollama with `ollama serve` before continuing.")
    else:
        _key = st.text_input(
            "Gemini API key",
            type="password",
            value=st.session_state.gemini_api_key,
            placeholder="Paste your Gemini API key…",
            key="gemini_key_input",
            label_visibility="collapsed",
        )
        if _key != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = _key

        if _key.strip():
            if st.button(
                "Save & Connect",
                use_container_width=True,
                key="gemini_connect_btn",
            ):
                with st.spinner("Validating Gemini API key…"):
                    try:
                        from langchain_google_genai import ChatGoogleGenerativeAI

                        _test_llm = ChatGoogleGenerativeAI(
                            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                            google_api_key=_key.strip(),
                            temperature=0,
                        )
                        _test_llm.invoke("ping")
                        st.success("Gemini connected", icon="✅")
                        st.session_state.gemini_connected = True
                    except (ValueError, RuntimeError, KeyError) as exc:
                        st.error(f"Gemini connection failed: {exc}", icon="❌")
                        st.session_state.gemini_connected = False
            elif st.session_state.get("gemini_connected"):
                st.success("Gemini connected", icon="✅")
            else:
                st.warning("Enter your API key and click Save & Connect.", icon="⚠️")
        else:
            st.caption("Enter your Gemini API key to enable cloud inference.")

    st.divider()

    if st.button("New Session", type="primary", use_container_width=True):
        _start_new_session()
        st.rerun()

    if st.button("Clear Current Session", type="secondary", use_container_width=True):
        _reset_current_session()
        st.rerun()

    if st.button("Clear All Sessions", type="secondary", use_container_width=True):
        _clear_all_sessions()
        st.rerun()


# ── Recruiter Dashboard ───────────────────────────────────────────────

if is_recruiter:
    st.header("📋 Recruiter Dashboard")
    if flash := st.session_state.pop("flash", None):
        st.success(flash)
    st.caption("Workflow: screen resumes → match candidates → schedule → chat → insight.")

    recruiter_tab = st.segmented_control(
        "Recruiter sections",
        options=["Resume Screening", "Candidate Matching", "Interview Scheduling", "Chat", "System Insight"],
        key="recruiter_tab",
        label_visibility="collapsed",
    )

    if recruiter_tab == "Resume Screening":
        _render_upload_tab()
    elif recruiter_tab == "Candidate Matching":
        _render_matching_tab()
    elif recruiter_tab == "Interview Scheduling":
        _render_scheduling_tab()
    elif recruiter_tab == "Chat":
        st.markdown("### 💬 Chat with SmartHire AI")
        st.caption(
            "Ask about your screened resumes, candidate rankings, interview "
            "slots, or hiring insights — or ask HR process questions. Answers "
            "come directly from your session results and the knowledge base."
        )
        _chat_section("Ask SmartHire AI anything…")
    else:
        _render_insight_tab()

# ── Candidate Chat ────────────────────────────────────────────────────

else:
    st.header("💬 Candidate Chat")
    st.caption(
        "Ask general questions about the recruitment process, interview "
        "preparation, and what to expect. Answers come from our knowledge "
        "base. This is an anonymous chat, so individual application details "
        "are not shown here."
    )
    _chat_section("Ask an HR question…")
