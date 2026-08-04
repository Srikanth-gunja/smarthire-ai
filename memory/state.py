"""Shared LangGraph state schema for SmartHire AI.

This module defines the single source of truth for all data flowing through
the multi-agent graph. Every node reads from and writes to this state.

Design decision: TypedDict over Pydantic BaseModel for the graph state.
LangGraph's StateGraph uses state as a lightweight dict-like channel — nodes
return partial dicts to update state, and LangGraph merges them. TypedDict
aligns naturally with this pattern and avoids unnecessary runtime validation
on every node update. We use Annotated reducers for list fields that should
append rather than overwrite across nodes. Agent-level input/output validation
is handled by separate Pydantic models in utils/models.py.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class SmartHireState(TypedDict, total=False):
    """Core state shared across all agents in the SmartHire AI graph.

    Fields are designed so each agent writes to its own slice, and the
    Reflection Node reads all slices before producing the final response.
    The conversation_history field uses an append reducer so messages
    accumulate across turns rather than being overwritten.
    """

    # ── Conversation Memory ──────────────────────────────────────────
    # Append reducer: each node's messages are added, not replaced.
    conversation_history: Annotated[list[BaseMessage], operator.add]

    # ── Supervisor Output ────────────────────────────────────────────
    # Classified intent from the Supervisor for this turn.
    current_intent: str

    # Which agents were invoked this turn (e.g. ["resume_screening", "candidate_matching"]).
    active_agents: list[str]

    # ── Resume Screening Agent Output ────────────────────────────────
    # List of parsed resume dicts, each containing structured fields
    # like skills, experience, education, summary.
    resumes: list[dict]

    # Raw resume documents supplied for the current workflow.  Keeping each
    # document separate is essential: a batch upload must be parsed as N
    # candidates, never as one concatenated "resume".
    resume_inputs: list[dict]

    # Optional, user-supplied scheduling constraints.  Each item contains a
    # candidate name, date, and one or more preferred time windows.
    candidate_availability: list[dict]

    # ── Candidate Matching Agent Output ──────────────────────────────
    # Ranked list of candidates with scores and justifications.
    candidate_rankings: list[dict]

    # ── Job Description (shared input) ──────────────────────────────
    # Parsed JD data used by Resume Screening and Candidate Matching.
    job_description: dict

    # ── Interview Scheduling Agent Output ────────────────────────────
    # Proposed and booked interview slots.
    interview_slots: list[dict]

    # ── HR Assistant Agent Output ────────────────────────────────────
    # Answers and guidance from the HR Assistant.
    hr_answers: list[dict]

    # ── Reflection Node Output ───────────────────────────────────────
    # Validation notes, issues found, and revised data from the
    # Reflection Node that runs before the final response.
    reflection_notes: dict

    # Whether the Reflection Node validated the combined agent outputs.
    reflection_validated: bool

    # How many times the Reflection Node has run this turn (1 = first pass,
    # 2 = after a correction retry).  Used to bound the retry loop to once.
    reflection_attempts: int

    # Agent to loop back to for a single correction attempt when validation
    # fails (e.g. "candidate_matching" or "interview_scheduling").  None when
    # no retry is required.
    retry_agent: str | None

    # Human-readable feedback produced by the Reflection Node and consumed by
    # the retried agent so its correction attempt is grounded in the issues.
    reflection_feedback: str | None

    # ── Final Response ───────────────────────────────────────────────
    # The polished response shown to the user after reflection.
    final_response: str

    # ── Error Handling ───────────────────────────────────────────────
    # Non-empty if any agent encounters an error during processing.
    error: str
