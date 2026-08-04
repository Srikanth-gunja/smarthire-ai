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
import logging
import os
import re
from collections import Counter
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


def _stage_checklist(completed: list[str], running: str | None) -> str:
    """HTML progress checklist showing pending / running / done agents."""
    done = set(completed)
    lines: list[str] = []
    for node in _STAGE_ORDER:
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


def _metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"<div style='background:#fff;border:1px solid {_BORDER};border-radius:12px;"
        f"padding:1rem .8rem'>"
        f"<div style='color:{_MUTED};font-size:.78em;text-transform:uppercase;letter-spacing:.04em'>"
        f"{label}</div>"
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

    Restores the human-readable transcript from ``chat_messages`` and the
    agent outputs (rankings, slots, final response) from the LangGraph
    checkpointer, then reruns so the chosen session is rendered.
    """
    audit = ChatAudit()
    st.session_state.session_id = session_id
    audit.save_session_id(session_id)
    st.session_state.chat_messages = audit.load_messages(session_id)
    st.session_state.graph_state = _restore_state(session_id)
    st.session_state.resume_texts = []
    st.session_state.jd_data = None
    st.session_state.show_results = True
    row = Database().fetch_one(
        "SELECT mode FROM sessions WHERE id = ?", (session_id,)
    )
    st.session_state.mode_picker = (
        "Candidate" if row and row["mode"] == "candidate" else "Recruiter"
    )


def _reset_current_session() -> None:
    """Clear only the active session's workspace, keeping its identifier."""
    session_id = st.session_state.session_id
    audit = ChatAudit()
    mode = "candidate" if st.session_state.mode_picker == "Candidate" else "recruiter"
    ConversationMemory().reset_session(session_id, mode)
    audit.clear(session_id)
    st.session_state.chat_messages = []
    st.session_state.graph_state = SmartHireState(conversation_history=[])
    st.session_state.resume_texts = []
    st.session_state.jd_data = None
    st.session_state.show_results = False
    st.session_state.sched = None
    st.session_state.recruiter_tab = "Upload & Screen"
    audit.save_session_id(session_id)


def _start_new_session() -> None:
    """Create an explicitly requested blank session and switch to it."""
    audit = ChatAudit()
    mode = "candidate" if st.session_state.mode_picker == "Candidate" else "recruiter"
    session_id = ConversationMemory().create_session(mode)
    st.session_state.session_id = session_id
    audit.save_session_id(session_id)
    st.session_state.chat_messages = []
    st.session_state.graph_state = SmartHireState(conversation_history=[])
    st.session_state.resume_texts = []
    st.session_state.jd_data = None
    st.session_state.show_results = False
    st.session_state.sched = None
    st.session_state.recruiter_tab = "Upload & Screen"


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
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = ChatAudit().load_messages(
        st.session_state.session_id
    )
if "resume_texts" not in st.session_state:
    st.session_state.resume_texts = []
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
    st.session_state.recruiter_tab = "Upload & Screen"


# ── Environment helpers ───────────────────────────────────────────────


def _check_llm() -> bool:
    """Validate the selected model provider without exposing credentials."""
    if os.getenv("LLM_PROVIDER", "ollama").strip().lower() == "gemini":
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
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


def _iter_pipeline(user_input: str):
    """Stream the graph, yielding (event_type, node_name) pairs."""
    from graph import run_smarthire_stream

    state = dict(st.session_state.graph_state)
    return run_smarthire_stream(
        user_input=user_input,
        state=state,
        session_id=st.session_state.session_id,
    )


def _run_with_progress(progress_title: str) -> None:
    """Run the current graph task, streaming a live stage checklist.

    Reads the final state from the checkpointer after streaming and stores it
    back into ``st.session_state.graph_state``.
    """
    with st.status(progress_title, expanded=True) as status:
        status.update(label="Starting agents…", state="running")
        progress_ph = st.empty()
        completed: list[str] = []
        running: str | None = None
        for event_type, node in _iter_pipeline(st.session_state.pending_input):
            if event_type == "task":
                running = node
                status.update(label=f"Running: {_agent_label(node)}", state="running")
            else:
                completed.append(node)
                running = None
            progress_ph.markdown(
                _stage_checklist(completed, running), unsafe_allow_html=True
            )
        status.update(label="✓ Pipeline complete", state="complete", expanded=False)
        progress_ph.markdown(_stage_checklist(completed, None), unsafe_allow_html=True)

    st.session_state.graph_state = _restore_state(st.session_state.session_id)


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
    _render_chat(st.session_state.chat_messages)

    if prompt := st.chat_input(input_placeholder):
        audit = ChatAudit()
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not _llm_ok:
            answer = "The selected AI provider is not configured or reachable. Check your .env settings."
            audit.log_turn(
                st.session_state.session_id, user_content=prompt, answer=answer
            )
        else:
            prior_count = len(
                st.session_state.graph_state.get("conversation_history", [])
            )
            st.session_state.pending_input = prompt
            _run_with_progress("Running SmartHire agents…")
            result = st.session_state.graph_state
            answer = result.get("final_response", "No response generated.")
            audit.log_turn(
                st.session_state.session_id,
                user_content=prompt,
                result=result,
                prior_history_count=prior_count,
            )

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )
        with st.chat_message("assistant"):
            st.markdown(answer)


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
    )
    sched = st.session_state.sched
    sched["slots"][idx]["confirmed"] = True
    st.session_state.sched = sched
    st.session_state.flash = (
        f"✅ Interview confirmed: {candidate} · {slot['date']} "
        f"{slot['time_start']}-{slot['time_end']} with {slot['interviewer']}."
    )
    st.rerun()


def _render_scheduling_tab() -> None:
    st.markdown("### 📅 Interview Scheduling")
    st.caption(
        "Pick a shortlisted candidate and confirm a slot. Grayed slots conflict "
        "with the interviewer's calendar and are disabled."
    )

    candidates = _candidate_options()
    if not candidates:
        _empty_state(
            "🗓️",
            "No candidates to schedule",
            "Screen and rank resumes first, then return here to book interviews.",
        )
        return

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
            "Interviewer", ["Bob Tech Lead", "Alice Manager", "Carol Director"]
        )
    with col_d:
        date = st.date_input(
            "Interview date",
            value=datetime.datetime.now(datetime.UTC).date(),
        ).isoformat()

    if st.button("Propose available slots", type="primary", use_container_width=True):
        st.session_state.sched = {
            "candidate": candidate,
            "interviewer": interviewer,
            "date": date,
            "slots": _propose_slots(candidate, date, interviewer),
        }

    sched = st.session_state.get("sched")
    if sched and sched["candidate"] == candidate and sched["interviewer"] == interviewer:
        st.divider()
        st.markdown(
            f"**{sched['candidate']}** with {sched['interviewer']} on {sched['date']}"
        )
        columns = st.columns(3)
        for idx, slot in enumerate(sched["slots"]):
            with columns[idx % 3]:
                if slot["conflict"]:
                    with st.container(border=True):
                        st.markdown(
                            f"<div style='opacity:.55'>⛔ **{slot['time_start']}–{slot['time_end']}**"
                            f"<br><small>{slot['interview_type']} · {slot['interviewer']}</small>"
                            f"<br><span style='color:#B91C1C'>Conflict</span></div>",
                            unsafe_allow_html=True,
                        )
                elif slot["confirmed"]:
                    with st.container(border=True):
                        st.markdown(
                            f"✅ **{slot['time_start']}–{slot['time_end']}**"
                            f"<br><small>{slot['interview_type']} · {slot['interviewer']}</small>"
                            f"<br><span style='color:{_PRIMARY};font-weight:600'>Confirmed</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    with st.container(border=True):
                        st.markdown(
                            f"**{slot['time_start']}–{slot['time_end']}**"
                            f"<br><small>{slot['interview_type']} · {slot['interviewer']}</small>",
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            "Confirm interview",
                            key=f"confirm_slot_{idx}",
                            use_container_width=True,
                        ):
                            _confirm_slot(slot, candidate, idx)


# ── System Insight ────────────────────────────────────────────────────


def _routing_log(limit: int = 10) -> list[str]:
    lines = []
    for msg in st.session_state.graph_state.get("conversation_history", []):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.startswith("[Supervisor]"):
            lines.append(content)
    return lines[-limit:]


def _jd_chart(rankings: list) -> plt.Figure | None:
    job_description = st.session_state.graph_state.get("job_description", {})
    active_title = (
        job_description.get("job_title")
        or job_description.get("title")
        or "Current job description"
    )
    counts = Counter(r.get("jd_title") or active_title for r in rankings)
    if not counts:
        return None
    titles = list(counts.keys())
    values = [counts[t] for t in titles]
    fig, ax = plt.subplots(figsize=(8, max(2.2, 0.5 * len(titles))))
    bars = ax.barh(titles, values, color=_PRIMARY)
    ax.set_xlabel("Candidates")
    ax.set_title("Candidates in this session")
    ax.invert_yaxis()
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(value), va="center")
    plt.tight_layout()
    return fig


def _render_insight_tab() -> None:
    st.markdown("### 📊 Session Insight")
    st.caption(
        f"Showing only data from the active session "
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
        _metric_card("Current session", "Active", "Separate from past sessions")
    with cols[2]:
        _metric_card("Avg match score", avg_score, "This session's ranking")
    with cols[3]:
        _metric_card("Interviews", str(len(interviews)), "This session's proposed slots")

    st.divider()

    with st.expander("Candidates in this session (chart)", expanded=True):
        chart = _jd_chart(rankings)
        if chart is None:
            _empty_state(
                "📊",
                "No session ranking data yet",
                "Run Screen & Rank Candidates in this session to populate the chart.",
            )
        else:
            st.pyplot(chart)

    with st.expander("Agent Routing Log (last 10 decisions)", expanded=False):
        log_lines = _routing_log()
        if log_lines:
            st.code("\n".join(log_lines))
        else:
            st.caption(
                "No routing decisions yet — the Supervisor's intent + routing "
                "choices appear here after you chat or screen candidates."
            )

    with st.expander("Raw conversation history", expanded=False):
        history = st.session_state.graph_state.get("conversation_history", [])
        if history:
            for msg in history:
                st.write(f"**{type(msg).__name__}**: {str(msg.content)[:220]}")
        else:
            st.caption("No conversation yet.")


# ── Upload & Screen tab ───────────────────────────────────────────────


def _render_upload_tab() -> None:
    st.markdown("### 📤 Resumes & Job Description")
    st.caption(
        "Upload resumes and a JD, then run the multi-agent pipeline. You'll see "
        "each agent execute live and a ranked results table afterwards."
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
            st.session_state.resume_texts = []
            for f in uploaded_resumes:
                text = (
                    _extract_pdf_text(f)
                    if f.type == "application/pdf"
                    else f.read().decode("utf-8") if f.name.lower().endswith(".txt")
                    else _extract_docx_text(f)
                )
                st.session_state.resume_texts.append({"name": f.name, "text": text})
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

        jd_text = ""
        if jd_option == "Upload file":
            jd_file = st.file_uploader(
                "JD file",
                type=["txt", "pdf", "docx"],
                key="jd_uploader",
                label_visibility="collapsed",
            )
            if jd_file:
                jd_text = (
                    _extract_pdf_text(jd_file)
                    if jd_file.type == "application/pdf"
                    else jd_file.read().decode("utf-8") if jd_file.name.lower().endswith(".txt")
                    else _extract_docx_text(jd_file)
                )
        elif jd_option == "Paste text":
            jd_text = st.text_area(
                "Paste job description",
                height=160,
                placeholder="Paste the full job description here…",
                label_visibility="collapsed",
            )
        else:
            jd_file = Path("data/sample_jd.txt")
            if jd_file.exists():
                jd_text = jd_file.read_text(encoding="utf-8")
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
        "🚀 Screen & Rank Candidates",
        type="primary",
        disabled=run_disabled,
        use_container_width=True,
    ):
        if not _llm_ok:
            st.error("Cannot run: the selected AI provider is not configured or reachable. Check .env.")
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
            st.session_state.graph_state = {
                **st.session_state.graph_state,
                "job_description": {
                    "id": jd_id,
                    "title": jd_text.strip().splitlines()[0][:200]
                    if jd_text.strip()
                    else None,
                    "raw_text": jd_text,
                },
                "resume_inputs": list(st.session_state.resume_texts),
            }

            resume_blocks = "\n\n".join(
                f"=== RESUME: {r['name']} ===\n{r['text']}"
                for r in st.session_state.resume_texts
            )
            combined = (
                f"JOB DESCRIPTION:\n{jd_text}\n\n"
                f"RESUMES:\n{resume_blocks}\n\n"
                "Please screen these resumes against the JD and rank them."
            )
            prior_count = len(
                st.session_state.graph_state.get("conversation_history", [])
            )
            st.session_state.pending_input = combined
            _run_with_progress("Running multi-agent pipeline…")
            ChatAudit().log_turn(
                st.session_state.session_id,
                result=st.session_state.graph_state,
                prior_history_count=prior_count,
            )
            st.session_state.show_results = True

    if st.session_state.get("show_results"):
        st.divider()
        st.markdown("### 📊 Ranking Results")
        state = st.session_state.graph_state
        final = state.get("final_response", "")
        if final:
            st.info(f"**Summary:** {final}")

        rankings = state.get("candidate_rankings", [])
        if rankings:
            st.markdown(
                f"**{len(rankings)} candidate(s) ranked.** Click *Schedule Interview* "
                "on any row to jump to Interview Scheduling."
            )
            _display_rankings(rankings)
        else:
            _empty_state(
                "🎯",
                "No candidates ranked yet",
                "Run Screen & Rank Candidates to populate the ranking table.",
            )
        _render_reflection_summary(state.get("reflection_notes", {}))
        _display_interview_slots(state.get("interview_slots", []))
        errors = state.get("error", "")
        if errors:
            st.error(f"Pipeline error: {errors}")


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
        for s in sessions:
            sid = s["id"]
            mode_name = s["mode"].capitalize() if s["mode"] else "Unknown"
            timestamp = str(s["last_active_at"])[:16]
            label = f"{mode_name} · {timestamp} · {sid[:8]}"
            if sid == st.session_state.session_id:
                label += " (active)"
            st.button(
                label,
                key=f"past_session_{sid}",
                use_container_width=True,
                on_click=_load_past_session,
                args=(sid,),
            )

    st.divider()

    _llm_ok = _check_llm()
    _provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if _llm_ok:
        st.success(f"{_provider.title()} ready", icon="✅")
    else:
        st.error(f"{_provider.title()} is not ready", icon="❌")
        st.caption("Set provider settings in .env; for local Ollama, start `ollama serve`.")

    st.divider()

    if st.button("New Session", type="primary", use_container_width=True):
        _start_new_session()
        st.rerun()

    if st.button("Clear Current Session", type="secondary", use_container_width=True):
        _reset_current_session()
        st.rerun()


# ── Recruiter Dashboard ───────────────────────────────────────────────

if is_recruiter:
    st.header("📋 Recruiter Dashboard")
    if flash := st.session_state.pop("flash", None):
        st.success(flash)
    st.caption("Pipeline: upload → screen & rank → schedule → chat → insight.")

    recruiter_tab = st.segmented_control(
        "Recruiter sections",
        options=["Upload & Screen", "Interview Scheduling", "Chat", "System Insight"],
        key="recruiter_tab",
        label_visibility="collapsed",
    )

    if recruiter_tab == "Upload & Screen":
        _render_upload_tab()
    elif recruiter_tab == "Interview Scheduling":
        _render_scheduling_tab()
    elif recruiter_tab == "Chat":
        st.markdown("### 💬 Chat with SmartHire AI")
        st.caption(
            "Messages are labelled with the agent that produced them, so you can "
            "see the multi-agent routing in action."
        )
        _chat_section("Ask SmartHire AI anything…")
    else:
        _render_insight_tab()

# ── Candidate Chat ────────────────────────────────────────────────────

else:
    st.header("💬 Candidate Chat")
    st.caption(
        "Ask questions about the recruitment process, interview prep, or "
        "application status."
    )
    _chat_section("Ask an HR question…")
