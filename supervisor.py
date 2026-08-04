"""Supervisor Agent — intent detection and routing for SmartHire AI.

Single responsibility: Classify the user's intent and decide which
specialist agents to invoke. The Supervisor never processes data itself;
it only routes to other agents via the graph's conditional edges.

Uses structured output (Pydantic ExecutionPlan) so the routing decision
is deterministic to branch on — no free-text parsing needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate

from utils.models import ExecutionPlan, SupervisorInput

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# Valid agent names used in routing
VALID_AGENTS = frozenset({
    "resume_screening",
    "candidate_matching",
    "interview_scheduling",
    "hr_assistant",
})

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a supervisor agent for SmartHire AI, a recruitment automation system. "
        "Your job is to classify the user's intent and determine which specialist agents "
        "should handle the request.\n\n"
        "Available specialist agents:\n"
        "- resume_screening: Parses and screens resumes against a job description. "
        "Use when the user uploads or asks to analyze/screen resumes or CVs.\n"
        "- candidate_matching: Ranks candidates by match score against a job description. "
        "Use when the user wants to compare, rank, or shortlist candidates.\n"
        "- interview_scheduling: Proposes and manages interview scheduling slots. "
        "Use when the user wants to schedule interviews or check availability.\n"
        "- hr_assistant: Answers HR questions and recruitment FAQs. "
        "Use for general HR queries, policy questions, or application status.\n\n"
        "Classification rules:\n"
        "- If the user mentions resumes, CVs, or screening → resume_screening\n"
        "- If the user mentions ranking, matching, comparing, or shortlisting → candidate_matching\n"
        "- If the user mentions scheduling, interviews, availability, or booking → interview_scheduling\n"
        "- If the user asks a question about policies, status, process, or general HR → hr_assistant\n"
        "- If the user wants multiple things (e.g., 'screen resumes AND schedule interviews'), "
        "classify as multi_intent and list all relevant agents in execution order.\n"
        "- If the user is greeting or making small talk → greeting (use hr_assistant)\n\n"
        "Execution order for multi_intent:\n"
        "1. resume_screening first (to parse/screen resumes)\n"
        "2. candidate_matching second (to rank screened resumes)\n"
        "3. interview_scheduling third (to schedule for top candidates)\n"
        "4. hr_assistant if there are also questions\n\n"
        "Return your response as a JSON object matching the ExecutionPlan schema."
    )),
    ("human", (
        "User query: {query}\n\n"
        "Conversation history:\n{history}"
    )),
])

# Keyword-based fallback when LLM classification fails or is unavailable
_FALLBACK_KEYWORD_MAP: dict[str, str] = {
    "resume": "resume_screening",
    "cv": "resume_screening",
    "screen": "resume_screening",
    "parse": "resume_screening",
    "rank": "candidate_matching",
    "match": "candidate_matching",
    "compare": "candidate_matching",
    "shortlist": "candidate_matching",
    "schedule": "interview_scheduling",
    "interview": "interview_scheduling",
    "availability": "interview_scheduling",
    "slot": "interview_scheduling",
    "book": "interview_scheduling",
    "status": "hr_assistant",
    "application": "hr_assistant",
    "policy": "hr_assistant",
    "question": "hr_assistant",
    "help": "hr_assistant",
    "process": "hr_assistant",
    "stage": "hr_assistant",
}


class Supervisor:
    """Classifies user intent and routes to specialist agents."""

    def __init__(self, llm: ChatOllama) -> None:
        """Initialize the Supervisor.

        Args:
            llm: The Ollama LLM instance used for intent classification.
        """
        self.llm = llm

    def classify_intent(self, input_data: SupervisorInput) -> ExecutionPlan:
        """Classify the user's intent and produce an execution plan.

        Uses structured output (Pydantic) for deterministic routing —
        no free-text parsing or string matching on LLM output.

        Args:
            input_data: The user query and conversation history.

        Returns:
            An ExecutionPlan with intent, agents to invoke, and reasoning.
        """
        history_str = (
            "\n".join(
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in input_data.conversation_history
            )
            if input_data.conversation_history
            else "None"
        )

        try:
            structured_llm = self.llm.with_structured_output(ExecutionPlan)
            chain = SUPERVISOR_PROMPT | structured_llm
            result = chain.invoke({
                "query": input_data.user_query,
                "history": history_str,
            })

            # Validate agents against allowlist
            valid_agents = [a for a in result.agents_to_invoke if a in VALID_AGENTS]
            if not valid_agents:
                logger.warning(
                    "LLM returned no valid agents (got %s), using fallback",
                    result.agents_to_invoke,
                )
                return self._fallback_classify(input_data.user_query)

            result.agents_to_invoke = valid_agents
            return result

        except Exception:
            logger.exception("LLM classification failed, using keyword fallback")
            return self._fallback_classify(input_data.user_query)

    def _fallback_classify(self, query: str) -> ExecutionPlan:
        """Keyword-based fallback when LLM classification fails.

        Args:
            query: The raw user query.

        Returns:
            An ExecutionPlan derived from keyword matching.
        """
        query_lower = query.lower()
        matched_agents: list[str] = []
        seen: set[str] = set()

        for keyword, agent in _FALLBACK_KEYWORD_MAP.items():
            if keyword in query_lower and agent not in seen:
                matched_agents.append(agent)
                seen.add(agent)

        if not matched_agents:
            return ExecutionPlan(
                intent="hr_question",
                agents_to_invoke=["hr_assistant"],
                reasoning="No specific intent detected; defaulting to HR assistant.",
            )

        intent = "multi_intent" if len(matched_agents) > 1 else matched_agents[0]
        return ExecutionPlan(
            intent=intent,
            agents_to_invoke=matched_agents,
            reasoning=f"Fallback classification based on keywords: {matched_agents}",
        )
