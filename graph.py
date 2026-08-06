"""LangGraph orchestration — builds and compiles the SmartHire AI graph.

This module defines the StateGraph, registers all agent nodes, wires
conditional edges based on Supervisor routing, and compiles the graph
for execution.

Graph shape:
    START → supervisor → {resume_screening, candidate_matching,
                          interview_scheduling, hr_assistant}
                       → memory_update (persists session data)
                       → reflection (validates & polishes output)
                       → END

Conditional edges out of the supervisor route to one or more agent nodes
in sequence. Each agent node pops itself off the active_agents list and
routes to the next agent, or to memory_update when the queue is empty.
"""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from memory.state import SmartHireState
from supervisor import Supervisor

logger = logging.getLogger(__name__)
execution_logger = logging.getLogger("smarthire.execution")


# ── Transient error handling ──────────────────────────────────────────


class TransientError(Exception):
    """Raised when a node fails due to a transient provider error (503, overload, timeout)."""

    def __init__(self, node_name: str, message: str) -> None:
        self.node_name = node_name
        self.message = message
        super().__init__(f"[{node_name}] {message}")


def _is_transient_error(exc: Exception) -> bool:
    """Return True if the exception represents a transient provider error."""
    msg = str(exc).lower()
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    status_code = getattr(exc, "status_code", None)

    # Check numeric status codes
    for val in (code, status, status_code):
        if val == 503 or (isinstance(val, int) and 500 <= val < 600):
            return True

    # Check error message patterns
    transient_signals = [
        "503", "overloaded", "overloaded_error", "resource_exhausted",
        "rate limit", "429", "unavailable", "deadline_exceeded",
        "timeout", "timed out", "connection reset", "connection refused",
        "server returned", "model is currently overloaded",
        "internal server error",
    ]
    return any(signal in msg for signal in transient_signals)


def _is_auth_error(exc: Exception) -> bool:
    """Return True if the exception represents an auth/key error (should NOT be retried)."""
    msg = str(exc).lower()
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    status_code = getattr(exc, "status_code", None)

    for val in (code, status, status_code):
        if val in (401, 403):
            return True

    auth_signals = ["api_key", "auth", "permission", "invalid_key", "deprecated"]
    return any(signal in msg for signal in auth_signals)


def _wrap_node(node_fn: Any, node_name: str, session_id: str = "") -> Any:
    """Wrap a graph node function with transient error handling.

    On a transient error the node:
      1. Records the pause state in the sessions DB table.
      2. Re-raises as TransientError so the stream stops cleanly.
    Auth errors are re-raised as-is (not resumeable).
    """

    def wrapper(state: SmartHireState) -> dict:
        started = perf_counter()
        execution_logger.info("event=start kind=agent agent=%s", node_name)
        try:
            result = node_fn(state)
            execution_logger.info(
                "event=complete kind=agent agent=%s duration_ms=%.1f",
                node_name,
                (perf_counter() - started) * 1000,
            )
            return result
        except TransientError:
            execution_logger.exception(
                "event=failed kind=agent agent=%s duration_ms=%.1f",
                node_name,
                (perf_counter() - started) * 1000,
            )
            raise
        except Exception as exc:
            if _is_transient_error(exc):
                _record_pause(session_id, node_name, exc)
                execution_logger.exception(
                    "event=failed kind=agent agent=%s duration_ms=%.1f",
                    node_name,
                    (perf_counter() - started) * 1000,
                )
                raise TransientError(node_name, str(exc)) from exc
            execution_logger.exception(
                "event=failed kind=agent agent=%s duration_ms=%.1f",
                node_name,
                (perf_counter() - started) * 1000,
            )
            raise

    return wrapper


def _record_pause(session_id: str, node_name: str, exc: Exception) -> None:
    """Persist the paused state to the sessions table."""
    if not session_id:
        return
    try:
        from db.database import Database
        Database().update_session_paused(session_id, node_name, str(exc))
        logger.warning("Session %s paused at %s: %s", session_id, node_name, exc)
    except Exception:
        logger.exception("Failed to record pause state for session %s", session_id)


# ── Routing helpers ───────────────────────────────────────────────────


def _route_to_next_agent(state: SmartHireState) -> str:
    """Read the first agent in the active_agents queue and route to it.

    After each agent executes it pops itself off the list.  When the
    queue is empty we fall through to memory_update.
    """
    agents = state.get("active_agents", [])
    if agents:
        return agents[0]
    return "memory_update"


def _route_after_reflection(state: SmartHireState) -> str:
    """Decide what happens after the Reflection Node runs.

    If validation failed AND this is the first reflection pass, loop back to
    the responsible agent for a single correction attempt.  Otherwise the
    pipeline ends — the result is returned to the user with any remaining
    issues surfaced in ``reflection_notes`` rather than silently dropped.
    """
    retry_agent = state.get("retry_agent")
    attempts = state.get("reflection_attempts", 0)
    if retry_agent and attempts <= 1:
        return retry_agent
    return "END"


# ── Graph builder ─────────────────────────────────────────────────────


def build_graph(
    supervisor: Supervisor,
    resume_screening_agent: Any,
    candidate_matching_agent: Any,
    interview_scheduler_agent: Any,
    hr_assistant_agent: Any,
    session_id: str = "",
    llm: Any = None,
    checkpointer: Any = None,
    screening_progress_cb: Any = None,
) -> Any:
    """Build and compile the LangGraph StateGraph for SmartHire AI.

    Agent instances are captured via closure so the node functions have
    no module-level singletons — easy to test with mocks.

    Args:
        supervisor: The Supervisor instance for intent classification.
        resume_screening_agent: ResumeScreeningAgent instance.
        candidate_matching_agent: CandidateMatchingAgent instance.
        interview_scheduler_agent: InterviewSchedulerAgent instance.
        hr_assistant_agent: HRAssistantAgent instance.
        session_id: Session identifier for conversation memory persistence.
        llm: Optional LLM for the reflection polish pass.
        checkpointer: Optional LangGraph checkpointer.  When not supplied but
            ``session_id`` is set, the shared SQLite checkpointer
            (``SqliteSaver`` over ``db/smarthire.db``) is used so agent state
            survives Streamlit reruns and app restarts.
        screening_progress_cb: Optional callback forwarded to the Resume
            Screening agent's ``screen_batch_async`` so the UI can render each
            resume's result live (progressive dropdown + count) instead of
            waiting for the whole batch.

    Returns:
        A compiled StateGraph ready for invocation via ``.invoke()``.
    """

    # ── Node: Supervisor ──────────────────────────────────────────────

    def supervisor_node(state: SmartHireState) -> dict:
        from utils.models import SupervisorInput

        history_msgs = state.get("conversation_history", [])
        user_query = history_msgs[-1].content if history_msgs else ""
        user_role = state.get("user_role", "recruiter")

        # Button-driven recruiter workflows are unambiguous.  Route them
        # directly instead of spending a Gemini call on intent classification.
        requested_workflow = state.get("requested_workflow")
        direct_routes = {
            "screening": ["resume_screening"],
            "matching": ["candidate_matching"],
        }
        if requested_workflow in direct_routes:
            agents = direct_routes[requested_workflow]
            label = "Resume Screening" if requested_workflow == "screening" else "Candidate Matching"
            logger.info("Direct workflow route=%s agents=%s", requested_workflow, agents)
            return {
                "current_intent": requested_workflow,
                "active_agents": agents,
                "conversation_history": [
                    AIMessage(content=f"[Workflow] Started {label}.")
                ],
            }

        history = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in history_msgs
        ]

        input_data = SupervisorInput(
            user_query=user_query,
            conversation_history=history,
            user_role=user_role,
        )
        plan = supervisor.classify_intent(input_data)

        logger.info(
            "Supervisor classified intent=%s agents=%s",
            plan.intent,
            plan.agents_to_invoke,
        )

        return {
            "current_intent": plan.intent,
            "active_agents": plan.agents_to_invoke,
            "conversation_history": [
                AIMessage(
                    content=(
                        f"[Supervisor] Intent: {plan.intent}. "
                        f"Routing to: {', '.join(plan.agents_to_invoke)}. "
                        f"Reasoning: {plan.reasoning}"
                    )
                ),
            ],
        }

    # ── Node: Resume Screening ────────────────────────────────────────

    def resume_screening_node(state: SmartHireState) -> dict:
        from tools.jd_analyzer import JDAnalyzer

        history_msgs = state.get("conversation_history", [])
        jd_data = dict(state.get("job_description", {}))
        resume_inputs = state.get("resume_inputs", [])
        # Backward-compatible single-document flow for chat/API callers.
        if not resume_inputs:
            resume_text = history_msgs[-1].content if history_msgs else ""
            resume_inputs = [{"name": None, "text": resume_text}]

        # The UI stores the original JD immediately.  Extract its requirements
        # here, at execution time, so all downstream agents score against the
        # actual uploaded/pasted role rather than placeholder metadata.  The
        # JD analysis runs as an asyncio task concurrently with the resume
        # parses; each resume is scored only after both its parse and the JD
        # task have resolved.
        analyzer = JDAnalyzer(resume_screening_agent.llm)

        async def run_screening_batch():
            jd_task = None
            if jd_data.get("raw_text") and not jd_data.get("required_skills"):
                jd_task = asyncio.create_task(
                    analyzer.analyze_async(jd_data["raw_text"])
                )
            results = await resume_screening_agent.screen_batch_async(
                resume_inputs,
                jd_data,
                on_progress=screening_progress_cb,
                jd_analysis_task=jd_task,
            )
            analyzed = {}
            jd_analysis_status = (
                "not_parsed"
                if not jd_data.get("raw_text")
                else "already_parsed" if jd_data.get("required_skills") else "pending"
            )
            if jd_task is not None:
                try:
                    analyzed = await jd_task
                    jd_analysis_status = (
                        "ok"
                        if analyzed.get("required_skills") or analyzed.get("job_title")
                        else "no_data"
                    )
                except Exception:
                    logger.exception("JD analysis failed; keeping raw JD")
                    jd_analysis_status = "error"
            return results, {**jd_data, **analyzed, "jd_analysis_status": jd_analysis_status}

        batch_results, jd_data = asyncio.run(run_screening_batch())
        screened, failures = [], []
        for resume, result in zip(resume_inputs, batch_results):
            if isinstance(result, Exception):
                failures.append({
                    "candidate_name": resume.get("name") or "Unknown resume",
                    "screening_status": "failed",
                    "error": str(result),
                    "filename": resume.get("name"),
                })
            else:
                screened.append(result.model_dump())

        remaining = state.get("active_agents", [])[1:]
        logger.info("ResumeScreening completed: %d succeeded, %d failed", len(screened), len(failures))

        return {
            "resumes": screened,
            "screening_failures": failures,
            "job_description": jd_data,
            "active_agents": remaining,
            "conversation_history": [
                AIMessage(
                    content=(
                        f"[ResumeScreening] Screened {len(screened)} resume(s); {len(failures)} failed. "
                        f"against {jd_data.get('job_title') or 'the uploaded role'}."
                    )
                ),
            ],
        }

    # ── Node: Candidate Matching ──────────────────────────────────────

    def candidate_matching_node(state: SmartHireState) -> dict:
        from utils.models import CandidateMatchingInput

        resumes = state.get("resumes", [])
        jd_data = state.get("job_description", {})

        input_data = CandidateMatchingInput(
            resumes=resumes,
            job_description=jd_data,
            reflection_feedback=state.get("reflection_feedback"),
        )
        result = asyncio.run(candidate_matching_agent.rank_candidates_async(input_data))

        remaining = state.get("active_agents", [])[1:]
        logger.info(
            "CandidateMatching completed: %d candidates ranked",
            result.total_candidates_evaluated,
        )

        return {
            "candidate_rankings": [r.model_dump() for r in result.ranked_candidates],
            "active_agents": remaining,
            "conversation_history": [
                AIMessage(
                    content=(
                        f"[CandidateMatching] Ranked {result.total_candidates_evaluated} "
                        f"candidates. Summary: {result.summary}"
                    )
                ),
            ],
        }

    # ── Node: Interview Scheduling ────────────────────────────────────

    def interview_scheduling_node(state: SmartHireState) -> dict:
        from utils.models import InterviewSchedulingInput

        rankings = state.get("candidate_rankings", [])
        candidates = (
            [r.get("candidate_name", "") for r in rankings[:3]]
            if rankings
            else ["Unknown"]
        )

        availability = state.get("candidate_availability", [])
        if not availability:
            # A dynamic next-business-day fallback keeps conversational
            # scheduling useful without embedding a demo-only calendar date.
            from datetime import UTC, datetime, timedelta

            next_day = datetime.now(UTC).date() + timedelta(days=1)
            while next_day.weekday() >= 5:
                next_day += timedelta(days=1)
            availability = [
                {
                    "candidate_name": candidate,
                    "date": next_day.isoformat(),
                    "preferred_times": [{"time_start": "09:00", "time_end": "17:00"}],
                }
                for candidate in candidates
            ]

        input_data = InterviewSchedulingInput(
            candidates=candidates,
            availability=availability,
            reflection_feedback=state.get("reflection_feedback"),
        )
        result = interview_scheduler_agent.propose_schedule(
            input_data, session_id=session_id
        )

        remaining = state.get("active_agents", [])[1:]
        logger.info("InterviewScheduling completed: %s", result.summary)

        return {
            "interview_slots": [s.model_dump() for s in result.proposed_slots],
            "active_agents": remaining,
            "conversation_history": [
                AIMessage(content=f"[InterviewScheduling] {result.summary}"),
            ],
        }

    # ── Node: HR Assistant ────────────────────────────────────────────

    def hr_assistant_node(state: SmartHireState) -> dict:
        from utils.models import HRAssistantInput

        history_msgs = state.get("conversation_history", [])
        query = ""
        for msg in reversed(history_msgs):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        user_role = state.get("user_role", "recruiter")
        # Candidate context is deliberately recruiter-data-free. The HR
        # Assistant's live workflow context must never expose ranking or
        # scheduling details to a candidate chat.
        workflow_context: dict = {
            "user_role": user_role,
            "prior_answers": state.get("hr_answers", []),
        }
        if user_role != "candidate":
            workflow_context.update(
                {
                    "job_description": state.get("job_description", {}),
                    "candidate_rankings": state.get("candidate_rankings", []),
                    "interview_slots": state.get("interview_slots", []),
                }
            )

        input_data = HRAssistantInput(
            query=query,
            user_role=user_role,
            context=workflow_context,
            reflection_feedback=state.get("reflection_feedback"),
        )
        result = hr_assistant_agent.answer_query(input_data, session_id=session_id)

        remaining = state.get("active_agents", [])[1:]
        logger.info("HRAssistant completed: confidence=%.2f", result.confidence)

        return {
            "hr_answers": [result.model_dump()],
            "active_agents": remaining,
            "conversation_history": [
                AIMessage(content=f"[HRAssistant] {result.answer}"),
            ],
        }

    # ── Node: Memory Update ───────────────────────────────────────────

    def memory_update_node(state: SmartHireState) -> dict:
        """Persist conversation memory and session data after each turn.

        Updates the ConversationMemory store with:
        - All new messages from this turn
        - Any job descriptions discussed
        - Any shortlisted candidates from rankings
        - Interview preferences if available
        """
        from memory.conversation_memory import ConversationMemory

        mem = ConversationMemory()
        sid = session_id or "default"

        # Append all new messages from this turn's conversation_history
        history = state.get("conversation_history", [])
        existing = mem.get_history(sid)
        existing_count = len(existing)
        for msg in history[existing_count:]:
            mem.append_turn(sid, msg)

        # Cache job description if present
        jd = state.get("job_description")
        if jd:
            mem.store_job_description(sid, jd)

        # Cache shortlisted candidates from rankings
        for candidate in state.get("candidate_rankings", []):
            mem.add_shortlisted_candidate(sid, candidate)

        # Cache interview preferences if present
        slots = state.get("interview_slots", [])
        if slots:
            mem.update_interview_preferences(
                sid,
                {"last_scheduled_count": len(slots)},
            )

        logger.info(
            "Memory update: session=%s history_len=%d jd=%s candidates=%d",
            sid,
            len(mem.get_history(sid)),
            bool(jd),
            len(mem.get_shortlisted_candidates(sid)),
        )
        return {}

    # ── Node: Reflection ──────────────────────────────────────────────

    def reflection_node(state: SmartHireState) -> dict:
        """Run the 4-point validation checklist and produce final output.

        Checks:
          a) verify_candidate_recommendations_match_jd
          b) check_interview_schedule_conflicts
          c) check_all_questions_answered
          d) improve_clarity_and_consistency

        Issues are surfaced in reflection_notes rather than hidden.
        """
        # Screening is an extraction-and-evidence workflow, not a comparison.
        # Save the Gemini polish/validation pass for Matching, where it checks
        # cross-candidate claims and ranking consistency.
        if state.get("requested_workflow") == "screening":
            count = len(state.get("resumes", []))
            return {
                "reflection_notes": {},
                "reflection_validated": True,
                "reflection_attempts": 0,
                "retry_agent": None,
                "reflection_feedback": None,
                "final_response": f"Screening complete for {count} resume(s).",
            }

        from agents.reflection_node import run_reflection

        return run_reflection(state, llm=llm)

    # ── Assemble the graph ────────────────────────────────────────────

    graph = StateGraph(SmartHireState)

    # Nodes — wrapped with transient error handling for resumable execution
    graph.add_node("supervisor", _wrap_node(supervisor_node, "supervisor", session_id))
    graph.add_node("resume_screening", _wrap_node(resume_screening_node, "resume_screening", session_id))
    graph.add_node("candidate_matching", _wrap_node(candidate_matching_node, "candidate_matching", session_id))
    graph.add_node("interview_scheduling", _wrap_node(interview_scheduling_node, "interview_scheduling", session_id))
    graph.add_node("hr_assistant", _wrap_node(hr_assistant_node, "hr_assistant", session_id))
    graph.add_node("memory_update", _wrap_node(memory_update_node, "memory_update", session_id))
    graph.add_node("reflection", _wrap_node(reflection_node, "reflection", session_id))

    # Entry point
    graph.set_entry_point("supervisor")

    # Conditional edges out of supervisor → first agent or memory_update
    graph.add_conditional_edges(
        "supervisor",
        _route_to_next_agent,
        {
            "resume_screening": "resume_screening",
            "candidate_matching": "candidate_matching",
            "interview_scheduling": "interview_scheduling",
            "hr_assistant": "hr_assistant",
            "memory_update": "memory_update",
        },
    )

    # Conditional edges out of each agent → next agent or memory_update
    for agent_name in [
        "resume_screening",
        "candidate_matching",
        "interview_scheduling",
        "hr_assistant",
    ]:
        graph.add_conditional_edges(
            agent_name,
            _route_to_next_agent,
            {
                "resume_screening": "resume_screening",
                "candidate_matching": "candidate_matching",
                "interview_scheduling": "interview_scheduling",
                "hr_assistant": "hr_assistant",
                "memory_update": "memory_update",
            },
        )

    # Linear tail: memory_update → reflection
    graph.add_edge("memory_update", "reflection")

    # Reflection → END, or loop back once to the responsible agent when
    # validation fails so it can attempt a correction (see
    # ``_route_after_reflection``).  The retried agent drains the empty
    # ``active_agents`` queue back to memory_update → reflection for a
    # second, final validation pass.
    graph.add_conditional_edges(
        "reflection",
        _route_after_reflection,
        {
            "resume_screening": "resume_screening",
            "candidate_matching": "candidate_matching",
            "interview_scheduling": "interview_scheduling",
            "hr_assistant": "hr_assistant",
            "memory_update": "memory_update",
            "END": END,
        },
    )

    # Persist conversation state with LangGraph's SQLite checkpointer
    # (SqliteSaver over db/smarthire.db), keyed by thread_id == session id.
    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    elif session_id:
        from memory.conversation_memory import get_checkpointer

        compile_kwargs["checkpointer"] = get_checkpointer()

    return graph.compile(**compile_kwargs)


# ── Public entrypoint ─────────────────────────────────────────────────


def _build_components(session_id: str, screening_progress_cb: Any = None):
    """Create the LLM, agents, checkpointer, and compiled graph for a session."""
    from agents.candidate_matching_agent import CandidateMatchingAgent
    from agents.hr_assistant_agent import HRAssistantAgent
    from agents.interview_scheduler_agent import InterviewSchedulerAgent
    from agents.resume_screening_agent import ResumeScreeningAgent
    from memory.conversation_memory import get_checkpointer, get_thread_config
    from utils.llm_factory import get_llm

    llm = get_llm()

    sup = Supervisor(llm)
    resume_agent = ResumeScreeningAgent(llm)
    matching_agent = CandidateMatchingAgent(llm)
    scheduler_agent = InterviewSchedulerAgent(llm)
    hr_agent = HRAssistantAgent(llm)

    checkpointer = get_checkpointer()
    compiled = build_graph(
        sup, resume_agent, matching_agent, scheduler_agent, hr_agent,
        session_id=session_id, llm=llm, checkpointer=checkpointer,
        screening_progress_cb=screening_progress_cb,
    )
    return compiled, checkpointer, get_thread_config(session_id)


def _prepare_input_state(
    user_input: str,
    state: SmartHireState | None,
    checkpointer: Any,
    config: dict,
) -> dict:
    """Build the input state for a graph invocation.

    When a prior checkpoint exists, only the new user message is passed as
    input (the checkpointer restores prior history) to avoid duplicating the
    conversation via the append reducer.  Fresh threads pass state through.
    """
    if state is None:
        state = {}

    prior = checkpointer.get_tuple(config)
    if prior is not None:
        base = dict(prior.checkpoint.get("channel_values") or {})
        for key, value in state.items():
            if key != "conversation_history":
                base[key] = value

        new_messages = []
        if user_input:
            prev_msgs = base.get("conversation_history", [])
            already_last = (
                prev_msgs
                and isinstance(prev_msgs[-1], HumanMessage)
                and prev_msgs[-1].content == user_input
            )
            if not already_last:
                new_messages = [HumanMessage(content=user_input)]
        base["conversation_history"] = new_messages
        return base

    if "conversation_history" not in state:
        state["conversation_history"] = []
    history = state["conversation_history"]
    if user_input and not (
        history
        and isinstance(history[-1], HumanMessage)
        and history[-1].content == user_input
    ):
        state["conversation_history"] = list(history) + [
            HumanMessage(content=user_input)
        ]
    return state


def _resolve_session(session_id: str | None) -> str:
    """Create (or refresh) a session row and return its id."""
    from memory.conversation_memory import ConversationMemory

    mem = ConversationMemory()
    if session_id is None:
        session_id = mem.create_session()
    else:
        # Refresh activity so the session can be resumed after a restart.
        mem.db.upsert_session(session_id)
    return session_id


def run_smarthire(
    user_input: str,
    state: SmartHireState | None = None,
    session_id: str | None = None,
) -> SmartHireState:
    """Run the full SmartHire AI pipeline for a single user input.

    Creates LLM, agents, and graph on each call (suitable for CLI / Phase 5
    integration).  For live per-agent progress, use :func:`run_smarthire_stream`.

    Args:
        user_input: The raw user message.
        state: Optional existing state to continue from.
        session_id: Optional session id for memory persistence. Generated
            if not provided.

    Returns:
        The updated SmartHireState after all agents have run.
    """
    session_id = _resolve_session(session_id)
    compiled, checkpointer, config = _build_components(session_id)
    input_state = _prepare_input_state(user_input, state, checkpointer, config)
    return compiled.invoke(input_state, config=config)


def run_smarthire_stream(
    user_input: str,
    state: SmartHireState | None = None,
    session_id: str | None = None,
):
    """Stream the SmartHire pipeline, yielding ``(event_type, node_name)``.

    The graph runs with ``stream_mode="debug"`` so the caller sees each node
    *start* (``("task", <node>)``) and *finish* (``("task_result", <node>)``)
    — ideal for live progress UIs while the local Ollama models respond
    (typically 10-30s).

    The final state is not yielded; the caller reads it back from the SQLite
    checkpointer afterwards (see ``memory.conversation_memory``).

    Yields:
        Pairs of ``(event_type, node_name)``.
    """
    session_id = _resolve_session(session_id)
    compiled, checkpointer, config = _build_components(session_id)
    input_state = _prepare_input_state(user_input, state, checkpointer, config)

    for event in compiled.stream(input_state, config=config, stream_mode="debug"):
        event_type = event.get("type")
        node_name = (event.get("payload") or {}).get("name")
        if event_type in ("task", "task_result") and node_name:
            yield event_type, node_name


def run_smarthire_stream_updates(
    user_input: str,
    state: SmartHireState | None = None,
    session_id: str | None = None,
    screening_progress_cb=None,
):
    """Stream the pipeline yielding ``(node_name, state_update)`` per node.

    Uses ``stream_mode="updates"`` so each completed node's output dict is
    yielded immediately — enabling incremental rendering of results in the UI.

    Args:
        user_input: The raw user message.
        state: Optional existing state to continue from.
        session_id: Optional session id for memory persistence.
        screening_progress_cb: Optional callback passed through to the Resume
            Screening node for live per-resume progress rendering.

    Yields:
        Tuples of ``(node_name, state_update_dict)`` where state_update_dict
        contains the keys written by that node (e.g. ``{"resumes": [...]}``).
    """
    session_id = _resolve_session(session_id)
    compiled, checkpointer, config = _build_components(session_id, screening_progress_cb)
    input_state = _prepare_input_state(user_input, state, checkpointer, config)

    for chunk in compiled.stream(input_state, config=config, stream_mode="updates"):
        # stream_mode="updates" yields dicts like {"node_name": {state_update}}
        for node_name, update in chunk.items():
            if update:
                yield node_name, dict(update)


def resume_run(session_id: str):
    """Resume a paused pipeline from the last successful checkpoint.

    Re-invokes the compiled graph with no new input on the same thread_id,
    causing LangGraph to pick up from the last saved checkpoint.  Completed
    nodes will NOT re-execute; only the failed node and subsequent ones run.

    Yields:
        Pairs of ``(event_type, node_name)`` just like ``run_smarthire_stream``.
    """
    from db.database import Database

    Database().clear_session_paused(session_id)
    compiled, _checkpointer, config = _build_components(session_id)

    # Empty input + existing checkpoint = resume from last checkpoint
    for event in compiled.stream({}, config=config, stream_mode="debug"):
        event_type = event.get("type")
        node_name = (event.get("payload") or {}).get("name")
        if event_type in ("task", "task_result") and node_name:
            yield event_type, node_name


def persist_graph_state(session_id: str, values: dict) -> None:
    """Best-effort write of a full channel-value dict to the session checkpoint.

    Used after a standalone per-resume retry so the corrected screening
    results survive app restarts (the same shape of checkpoint dict that
    ``ConversationMemory._save`` writes, so it is safe for ``SqliteSaver``).
    """
    from memory.conversation_memory import get_checkpointer, get_thread_config

    try:
        saver = get_checkpointer()
        config = get_thread_config(session_id)
        current = saver.get_tuple(config)
        if current is None:
            return
        old = current.checkpoint or {}
        import uuid
        from datetime import UTC, datetime

        checkpoint = {
            "v": old.get("v", 1),
            "ts": datetime.now(UTC).isoformat(),
            "id": str(uuid.uuid4()),
            "channel_values": values,
            "channel_versions": old.get("channel_versions", {}),
            "versions_seen": old.get("versions_seen", {}),
            "pending_writes": [],
        }
        saver.put(config, checkpoint, old.get("metadata", {}), {})
    except Exception:
        logger.exception("Failed to persist graph state for session %s", session_id)


def rescreen_single_resume(session_id: str, filename: str):
    """Re-screen one previously-failed resume against the session's job description.

    Standalone retry used by the Resume Screening UI's per-resume "retry"
    button.  It reuses the same async screening path as the batch fan-out so a
    single document can be re-run without touching the other results.  Returns
    a ``(result_dict, error_message)`` tuple.

    Args:
        session_id: The session whose JD + uploaded files are used.
        filename: Original uploaded resume file name to retry.

    Returns:
        ``(screening_output_dict, None)`` on success or ``(None, error_message)``
        on failure.
    """
    from agents.resume_screening_agent import ResumeScreeningAgent
    from db.database import Database
    from memory.conversation_memory import get_checkpointer, get_thread_config
    from utils.llm_factory import get_llm
    from utils.models import ResumeScreeningInput

    checkpoint = get_checkpointer().get_tuple(get_thread_config(session_id))
    values = dict((checkpoint.checkpoint or {}).get("channel_values") or {}) if checkpoint else {}
    jd_data = values.get("job_description") or {}
    if not jd_data.get("raw_text") and not jd_data.get("required_skills"):
        return None, "Job description is not available for this session."

    resume = next(
        (
            row
            for row in Database().get_session_uploads(session_id)
            if row["kind"] == "resume" and row["filename"] == filename
        ),
        None,
    )
    if resume is None:
        return None, f"Uploaded resume '{filename}' was not found in this session."

    agent = ResumeScreeningAgent(get_llm())
    try:
        output = asyncio.run(
            agent.screen_resume_async(
                ResumeScreeningInput(
                    resume_text=resume["extracted_text"],
                    job_description=jd_data,
                ),
                resume_filename=filename,
            )
        )
        return output.model_dump(), None
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI for retry
        return None, str(exc)


def answer_candidate_query(
    user_input: str,
    session_id: str,
    prior_context: dict | None = None,
) -> str:
    """Answer a candidate question directly from the HR knowledge base.

    Candidate chat is a generic, anonymous Q&A. It must NOT run the
    multi-agent pipeline (supervisor, screening, matching, scheduling) —
    those are recruiter tools. This bypasses the graph entirely and answers
    straight from the HR Assistant's role-aware knowledge base.

    Args:
        user_input: The candidate's question.
        session_id: Session id used to tag persisted answers.
        prior_context: Optional dict with a ``prior_answers`` list for
            lightweight follow-up context.

    Returns:
        The HR Assistant's answer string.
    """
    from agents.hr_assistant_agent import HRAssistantAgent
    from utils.llm_factory import get_llm
    from utils.models import HRAssistantInput

    context = dict(prior_context or {})
    context.setdefault("prior_answers", [])

    llm = get_llm()
    hr = HRAssistantAgent(llm)
    result = hr.answer_query(
        HRAssistantInput(
            query=user_input,
            user_role="candidate",
            context=context,
        ),
        session_id=session_id,
    )
    return result.answer


# ── Recruiter chat: answer from stored session results ─────────────────

# Words/phrases that indicate a process/policy question rather than a request
# for stored results. When present, the recruiter chat answers from the HR
# knowledge base instead of the results tables.
_HR_SIGNAL_WORDS = (
    "process", "policy", "procedure", "best practice", "practices",
    "guideline", "how to", "how do i", "how do we", "what is", "explain",
    "workflow", "approach",
)

# Results intents matched in priority order, with their trigger keywords.
_RECRUITER_RESULTS_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rankings", ("rank", "match", "compare", "shortlist", "top candidate",
                  "best candidate", "candidate summary", "scored")),
    ("screening", ("screen", "resume", "cv")),
    ("scheduling", ("schedule", "slot", "book", "availability", "propose")),
    ("insights", ("insight", "analytics", "stats", "trend", "report")),
)


def _parse_recruiter_results_intent(query: str) -> str | None:
    """Classify a recruiter chat query as a results lookup, or None for HR.

    Returns 'screening', 'rankings', 'scheduling', or 'insights' when the
    query asks about stored results, otherwise None so the caller answers
    from the HR knowledge base.
    """
    q = query.lower()
    if any(word in q for word in _HR_SIGNAL_WORDS):
        return None
    for intent, keywords in _RECRUITER_RESULTS_RULES:
        if any(keyword in q for keyword in keywords):
            return intent
    return None


def _fmt_number(value) -> str:
    """Format a numeric score as a percentage string, else '—'."""
    try:
        return f"{float(value):.0f}%"
    except (TypeError, ValueError):
        return "—"


def _rank_key(candidate: dict) -> int:
    try:
        return int(candidate.get("rank", 999))
    except (TypeError, ValueError):
        return 999


def _fmt_screening_results(state: dict) -> str:
    """Format screened resumes as a chat-friendly message."""
    resumes = state.get("resumes") or []
    if not resumes:
        return (
            "No resumes have been screened yet. Upload resumes and a job "
            "description, then run 'Screen & Rank Candidates'."
        )
    lines = [f"I found {len(resumes)} screened resume(s):"]
    for r in resumes:
        name = r.get("candidate_name", "Unknown")
        score = _fmt_number(r.get("match_score"))
        skills = ", ".join(r.get("skills") or [])
        entry = f"- **{name}** ({score} match)"
        if skills:
            entry += f": {skills}"
        lines.append(entry)
    return "\n".join(lines)


def _fmt_rankings(state: dict) -> str:
    """Format ranked candidates as a chat-friendly message."""
    rankings = state.get("candidate_rankings") or []
    if not rankings:
        return (
            "No candidates have been ranked yet. Screen resumes first, then "
            "run 'Screen & Rank Candidates'."
        )
    ordered = sorted(rankings, key=_rank_key)
    lines = [f"Here are the ranked candidates ({len(ordered)} total):"]
    for r in ordered:
        name = r.get("candidate_name", "Unknown")
        score = _fmt_number(r.get("match_score"))
        justification = (r.get("justification") or "").strip()
        entry = f"- **#{r.get('rank', '?')} {name}** — {score} match"
        if justification:
            entry += f". {justification}"
        lines.append(entry)
    return "\n".join(lines)


def _fmt_scheduling(state: dict) -> str:
    """Format proposed interview slots as a chat-friendly message."""
    slots = state.get("interview_slots") or []
    if not slots:
        return (
            "No interviews have been scheduled yet. Rank candidates, then "
            "propose and confirm interview slots."
        )
    lines = [f"I found {len(slots)} interview slot(s):"]
    for s in slots:
        name = s.get("candidate_name", "?")
        date = s.get("date", "?")
        window = f"{s.get('time_start')}-{s.get('time_end')}"
        interviewer = s.get("interviewer", "?")
        interview_type = s.get("interview_type", "")
        status = s.get("status", "proposed")
        lines.append(
            f"- **{name}** — {date} {window} with {interviewer} "
            f"({interview_type}) — {status}"
        )
    return "\n".join(lines)


def _fmt_insights(state: dict) -> str:
    """Format high-level recruiter insights from the session data."""
    resumes = state.get("resumes") or []
    rankings = state.get("candidate_rankings") or []
    slots = state.get("interview_slots") or []
    if not resumes and not rankings and not slots:
        return (
            "No hiring data yet. Run 'Screen & Rank Candidates' to populate "
            "your insights."
        )
    scores = [r["match_score"] for r in rankings if r.get("match_score") is not None]
    avg = f"{sum(scores) / len(scores):.0f}%" if scores else "—"
    top = rankings[0].get("candidate_name") if rankings else "—"
    jd = state.get("job_description") or {}
    lines = [
        "**Recruiter insights**",
        f"- Resumes screened: {len(resumes)}",
        f"- Candidates ranked: {len(rankings)}",
        f"- Average match score: {avg}",
        f"- Top candidate: {top}",
        f"- Interviews proposed: {len(slots)}",
    ]
    if jd.get("job_title") or jd.get("title"):
        lines.append(f"- Job description: {jd.get('job_title') or jd.get('title')}")
    return "\n".join(lines)


def answer_recruiter_chat(
    user_input: str,
    session_id: str,
    state: dict | None = None,
    prior_context: dict | None = None,
) -> str:
    """Answer a recruiter chat query without re-running the pipeline.

    Results-style queries ("screen resumes", "rank candidates", "schedule
    interviews", "hiring insights") are answered directly from the stored
    session results. Process/policy questions fall back to the HR Assistant
    knowledge base (recruiter role). Never re-invokes the agents.

    Args:
        user_input: The recruiter's message.
        session_id: Session id used to tag persisted HR answers.
        state: The current graph state containing stored results.
        prior_context: Optional dict with a ``prior_answers`` list.

    Returns:
        The chat-ready answer string.
    """
    state = dict(state or {})
    intent = _parse_recruiter_results_intent(user_input)

    if intent == "screening":
        return _fmt_screening_results(state)
    if intent == "rankings":
        return _fmt_rankings(state)
    if intent == "scheduling":
        return _fmt_scheduling(state)
    if intent == "insights":
        return _fmt_insights(state)

    # Process/policy/greeting → HR Assistant knowledge base (recruiter role).
    from agents.hr_assistant_agent import HRAssistantAgent
    from utils.llm_factory import get_llm
    from utils.models import HRAssistantInput

    context = dict(prior_context or {})
    context.setdefault("prior_answers", [])
    workflow = {
        key: state.get(key)
        for key in ("resumes", "candidate_rankings", "interview_slots", "job_description")
        if state.get(key)
    }
    if workflow:
        context["workflow"] = workflow

    llm = get_llm()
    hr = HRAssistantAgent(llm)
    result = hr.answer_query(
        HRAssistantInput(query=user_input, user_role="recruiter", context=context),
        session_id=session_id,
    )
    return result.answer


# ── CLI smoke test ────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json

    from agents.candidate_matching_agent import CandidateMatchingAgent
    from agents.hr_assistant_agent import HRAssistantAgent
    from agents.interview_scheduler_agent import InterviewSchedulerAgent
    from agents.resume_screening_agent import ResumeScreeningAgent
    from utils.llm_factory import get_llm

    logging.basicConfig(level=logging.INFO)

    print("=== SmartHire AI Graph Smoke Test ===\n")

    llm = get_llm()

    sup = Supervisor(llm)
    resume_agent = ResumeScreeningAgent(llm)
    matching_agent = CandidateMatchingAgent(llm)
    scheduler_agent = InterviewSchedulerAgent(llm)
    hr_agent = HRAssistantAgent(llm)

    compiled = build_graph(
        sup, resume_agent, matching_agent, scheduler_agent, hr_agent
    )

    # Run an HR question scenario against the real LLM
    initial_state: SmartHireState = {
        "conversation_history": [
            HumanMessage(
                content="What are the stages of the hiring process?"
            )
        ],
    }

    print("Input: What are the stages of the hiring process?\n")

    result = compiled.invoke(initial_state)

    print(f"Intent:           {result.get('current_intent', 'N/A')}")
    print(f"Active agents:    {result.get('active_agents', [])}")
    print(f"HR answers:       {_json.dumps(result.get('hr_answers', []), indent=2)}")
    print(f"Final response:   {result.get('final_response', 'N/A')}")
    print(f"Reflection notes: {_json.dumps(result.get('reflection_notes', {}), indent=2)}")
    print("\n=== Smoke Test Complete ===")
