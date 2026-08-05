"""HR Assistant Agent — answers candidate and recruiter questions.

Single responsibility: Answer candidate OR recruiter questions about
recruitment processes, policies, and general HR knowledge, behaving
differently depending on the user's role.

Role-aware behaviour:
  • Candidate — answers only candidate-relevant topics (application status,
    interview rounds, preparation, documents, rescheduling, process) grounded
    in the dynamic knowledge base. Never invents personal details.
  • Recruiter — answers HR/process questions grounded in the knowledge base
    plus the live workflow context (rankings, slots) passed by the graph.

Grounded in ``knowledge/recruitment.json``. If a question does not match any
knowledge topic, the agent politely says the information is unavailable and
never hallucinates. Escalates legal/sensitive topics to human HR.
Does NOT make hiring decisions, process resumes, rank candidates, or schedule
interviews — those belong to the specialist agents.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tools.candidate_database import CandidateDatabase
from utils.models import HRAssistantInput, HRAssistantOutput

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_PATH = Path("knowledge/recruitment.json")

# Recruiter-only knowledge topic — excluded from candidate retrieval so
# candidate answers never leak recruiter-side workflows.
_RECRUITER_ONLY_TOPICS = frozenset({"recruiter_guidelines"})

# Keys in the live workflow context that are recruiter-side and must never be
# surfaced to candidates.
_RECRUITER_CONTEXT_KEYS = frozenset(
    {"candidate_rankings", "interview_slots", "job_description", "resumes"}
)

ESCALATION_KEYWORDS = frozenset({
    "legal", "lawyer", "attorney", "sue", "lawsuit", "discrimination",
    "harassment", "accommodation", "disability", "religious",
    "non-compete", "nda", "contract", "termination", "fired",
    "salary negotiation", "offer negotiation",
})

GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|good\s*(morning|afternoon|evening)|"
    r"howdy|greetings|namaste|hola)\b[\s!?.,]*$",
    re.IGNORECASE,
)

UNAVAILABLE_ANSWER = (
    "I'm sorry, I don't have enough information to answer that question. "
    "Please contact our HR team directly for assistance."
)

CANDIDATE_SYSTEM_PROMPT = (
    "You are an HR assistant for TalentBridge HR Solutions talking to a "
    "CANDIDATE about the recruitment process. Be friendly, professional, and concise.\n"
    "RULES:\n"
    "- Answer using ONLY the knowledge base content below and the conversation context.\n"
    "- This is an anonymous, general Q&A chat. NEVER invent or claim to know the "
    "candidate's individual application status, interview time, score, or result — "
    "you have no access to any personal record.\n"
    "- If the candidate asks about their own private details (status, their interview "
    "date, their result), do not guess. Explain that individual application details "
    "are shared directly with each candidate by the recruitment team.\n"
    "- Do not discuss recruiter-side activities such as screening or ranking other "
    "candidates.\n"
    "- If the knowledge base does not cover the question, say the information is "
    "currently unavailable and suggest contacting HR.\n"
    "- Keep answers concise (2-4 sentences) and friendly."
)

RECRUITER_SYSTEM_PROMPT = (
    "You are an HR assistant for TalentBridge HR Solutions talking to a "
    "RECRUITER. Be professional, concise, and action-oriented.\n"
    "RULES:\n"
    "- Answer using ONLY the knowledge base content below and the live workflow context.\n"
    "- For screening, ranking, matching, or scheduling questions, use the live workflow "
    "context; if the data is absent, tell the recruiter to run those steps first.\n"
    "- NEVER invent candidate data, scores, or interview details that are not present "
    "in the context.\n"
    "- If the knowledge base does not cover the question, say the information is "
    "currently unavailable and suggest contacting the HR team.\n"
    "- Keep answers concise."
)


class HRAnswer(BaseModel):
    """LLM output schema for an HR answer."""

    answer: str = Field(description="The answer to the HR question.")
    confidence: float = Field(ge=0, le=1, description="Confidence in the answer.")
    needs_escalation: bool = Field(description="Whether human HR intervention is needed.")
    relevant_sections: list[str] = Field(
        default_factory=list,
        description="Sections of the knowledge base used.",
    )


class HRAssistantAgent:
    """Answers candidate and recruiter HR questions with role awareness."""

    def __init__(
        self,
        llm: ChatOllama,
        candidate_db: CandidateDatabase | None = None,
        knowledge_base_path: str | Path | None = None,
    ) -> None:
        """Initialize the HR Assistant Agent.

        Args:
            llm: The Ollama LLM instance used for generating answers.
            candidate_db: CandidateDatabase for looking up candidate info.
            knowledge_base_path: Path to the HR knowledge base (JSON file with
                a ``topics`` map, or a markdown file for backward compatibility).
        """
        self.llm = llm
        self.candidate_db = candidate_db or CandidateDatabase()
        self.kb_path = Path(knowledge_base_path) if knowledge_base_path else KNOWLEDGE_BASE_PATH
        self._kb_data: dict | None = None
        self._kb_content: str | None = None

    def answer_query(
        self,
        input_data: HRAssistantInput,
        session_id: str | None = None,
    ) -> HRAssistantOutput:
        """Answer a candidate or recruiter HR question.

        Args:
            input_data: The query, role, and optional conversation context.
            session_id: Optional session id used to tag the persisted answer.

        Returns:
            The answer with sources, confidence, and escalation flag.
        """
        role = input_data.user_role or "recruiter"

        if self._should_escalate(input_data.query):
            return self._persist_and_return(
                input_data,
                HRAssistantOutput(
                    answer=(
                        "This question requires specialized HR attention. "
                        "Please contact our HR team directly for assistance "
                        "with this matter."
                    ),
                    sources=[],
                    confidence=0.0,
                    needs_escalation=True,
                ),
                session_id,
            )

        if GREETING_PATTERNS.match(input_data.query.strip()):
            greeting = (
                "Hello! I'm the SmartHire HR Assistant. I can help with the "
                "recruitment process, interview preparation, required documents, "
                "and more. What would you like to know?"
                if role == "candidate"
                else "Hello! I'm the SmartHire HR Assistant. I can help with "
                "recruitment process questions and recruiter guidelines. "
                "How can I help?"
            )
            return self._persist_and_return(
                input_data,
                HRAssistantOutput(
                    answer=greeting,
                    sources=[],
                    confidence=1.0,
                    needs_escalation=False,
                ),
                session_id,
            )

        relevant = self._retrieve_relevant_sections(input_data.query, role)
        if not relevant:
            return self._persist_and_return(
                input_data,
                HRAssistantOutput(
                    answer=UNAVAILABLE_ANSWER,
                    sources=[],
                    confidence=0.0,
                    needs_escalation=False,
                ),
                session_id,
            )

        knowledge_base = "\n\n".join(
            f"### {section['topic']}\n{section['content']}" for section in relevant
        )
        sources = [section["topic"] for section in relevant]
        context_str = self._render_context(input_data, role)
        if input_data.reflection_feedback:
            context_str = (
                f"{context_str}\n\nREFLECTION FEEDBACK (answer the part that was "
                f"previously left unaddressed):\n{input_data.reflection_feedback}"
            )

        prompt = self._build_prompt(role)
        structured_llm = self.llm.with_structured_output(HRAnswer)
        chain = prompt | structured_llm
        result = chain.invoke({
            "knowledge_base": knowledge_base,
            "query": input_data.query,
            "context": context_str,
        })

        return self._persist_and_return(
            input_data,
            HRAssistantOutput(
                answer=result.answer,
                sources=sources,
                confidence=result.confidence,
                needs_escalation=result.needs_escalation,
            ),
            session_id,
        )

    def _build_prompt(self, role: str) -> ChatPromptTemplate:
        """Build a role-appropriate prompt for the HR Assistant."""
        role_system = CANDIDATE_SYSTEM_PROMPT if role == "candidate" else RECRUITER_SYSTEM_PROMPT
        return ChatPromptTemplate.from_messages([
            ("system", role_system),
            ("system", "KNOWLEDGE BASE:\n{knowledge_base}"),
            ("human", "Question: {query}\n\nCONVERSATION CONTEXT: {context}"),
        ])

    def _render_context(self, input_data: HRAssistantInput, role: str) -> str:
        """Render conversation context, scoped to the user's role.

        Candidate context never includes recruiter-side workflow data
        (rankings, slots, resumes, JD) — a candidate must never see
        "we identified two top candidates" style content.
        """
        context = dict(input_data.context or {})
        if role == "candidate":
            context = {
                k: v for k, v in context.items() if k not in _RECRUITER_CONTEXT_KEYS
            }
        if not context:
            return "None"
        return str(context)

    def _persist_and_return(
        self,
        input_data: HRAssistantInput,
        output: HRAssistantOutput,
        session_id: str | None,
    ) -> HRAssistantOutput:
        """Persist an HR answer to SQLite (best-effort) and return it."""
        try:
            from db.database import Database

            Database().insert_hr_answer(
                query=input_data.query,
                answer=output.answer,
                session_id=session_id,
                sources=output.sources,
                confidence=output.confidence,
                needs_escalation=output.needs_escalation,
            )
        except Exception:
            logger.exception("Failed to persist HR assistant answer")
        return output

    def _load_kb_data(self) -> dict:
        """Load the structured knowledge base from disk (cached).

        Returns:
            A dict with a ``topics`` map of topic -> {keywords, content}.
            Empty dict if the file is missing or unreadable.
        """
        if self._kb_data is not None:
            return self._kb_data

        self._kb_data = {}
        if not self.kb_path.exists():
            logger.warning("HR knowledge base not found at %s", self.kb_path)
            return self._kb_data

        try:
            if self.kb_path.suffix.lower() == ".json":
                raw = json.loads(self.kb_path.read_text(encoding="utf-8"))
                self._kb_data = raw.get("topics", {}) if isinstance(raw, dict) else {}
            else:
                # Backward compatibility: markdown knowledge base treated as a
                # single "process" topic so old callers keep working.
                text = self.kb_path.read_text(encoding="utf-8")
                self._kb_data = {
                    "process": {"keywords": [], "content": text},
                }
        except (OSError, ValueError, json.JSONDecodeError):
            logger.exception("Failed to load HR knowledge base at %s", self.kb_path)
            self._kb_data = {}

        return self._kb_data

    def _load_knowledge_base(self) -> str:
        """Return a plain-text rendering of the knowledge base (cached)."""
        if self._kb_content is not None:
            return self._kb_content

        topics = self._load_kb_data()
        if not topics:
            self._kb_content = "No knowledge base available."
            return self._kb_content

        lines = []
        for key, entry in topics.items():
            title = key.replace("_", " ").title()
            lines.append(f"## {title}\n{entry.get('content', '')}")
        self._kb_content = "\n\n".join(lines)
        return self._kb_content

    def _retrieve_relevant_sections(self, query: str, role: str) -> list[dict]:
        """Retrieve knowledge topics relevant to the query.

        Uses keyword + content-overlap scoring against the knowledge base.
        Recruiter-only topics are excluded for candidates.

        Args:
            query: The user's question.
            role: The user's role ('candidate' or 'recruiter').

        Returns:
            List of {topic, content, score} dicts, best match first.
        """
        topics = self._load_kb_data()
        if not topics:
            return []

        query_lower = query.lower()
        query_words = set(re.findall(r"[a-z]{4,}", query_lower))
        results: list[dict] = []

        for key, entry in topics.items():
            if role == "candidate" and key in _RECRUITER_ONLY_TOPICS:
                continue

            keywords = entry.get("keywords") or []
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 2

            # Candidates must match an explicit knowledge keyword so
            # recruiter-style phrasing ("screen resumes", "rank candidates")
            # can never leak into candidate answers. Content-overlap is used
            # only for recruiters as a soft fallback.
            if score == 0 and role != "candidate":
                content_words = set(re.findall(r"[a-z]{4,}", entry.get("content", "").lower()))
                overlap = len(query_words & content_words)
                if overlap >= 2:
                    score = 1

            if score > 0:
                results.append({"topic": key, "content": entry.get("content", ""), "score": score})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def _should_escalate(self, query: str) -> bool:
        """Determine if the question requires human HR intervention.

        Checks for legal, accommodation, discrimination, and other
        sensitive topics that should be handled by human HR.

        Args:
            query: The user's question.

        Returns:
            True if escalation is needed, False otherwise.
        """
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in ESCALATION_KEYWORDS)

    def lookup_candidate(self, candidate_id: str) -> dict | None:
        """Look up a candidate in the database by ID.

        Args:
            candidate_id: The candidate ID to look up.

        Returns:
            Candidate dict if found, None otherwise.
        """
        return self.candidate_db.get_candidate(candidate_id)

    def search_candidates(self, query: str) -> list[dict]:
        """Search candidates by name or skills.

        Args:
            query: Search query (name substring or skill).

        Returns:
            List of matching candidate dicts.
        """
        return self.candidate_db.search_candidates({"name": query})


if __name__ == "__main__":
    import sys

    from utils.llm_factory import get_llm

    logging.basicConfig(level=logging.INFO)
    llm = get_llm()
    agent = HRAssistantAgent(llm)

    query = sys.argv[1] if len(sys.argv) > 1 else "What are the stages of the hiring process?"
    result = agent.answer_query(HRAssistantInput(query=query, user_role="candidate"))
    print(f"\nAnswer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Escalation needed: {result.needs_escalation}")
    print(f"Sources: {result.sources}")
