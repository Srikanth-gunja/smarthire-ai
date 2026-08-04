"""HR Assistant Agent — answers recruitment FAQs and provides guidance.

Single responsibility: Answer candidate or recruiter questions about
recruitment processes, policies, and general HR knowledge. Grounded in
the approved knowledge base plus live workflow context. Escalates
to human HR when the question is outside its scope.
Does NOT make hiring decisions, process resumes, rank candidates,
or schedule interviews.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tools.candidate_database import CandidateDatabase
from utils.models import HRAssistantInput, HRAssistantOutput

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_PATH = Path("prompts/hr_knowledge_base.md")

ESCALATION_KEYWORDS = frozenset({
    "legal", "lawyer", "attorney", "sue", "lawsuit", "discrimination",
    "harassment", "accommodation", "disability", "religious",
    "non-compete", "nda", "contract", "termination", "fired",
    "salary negotiation", "offer negotiation",
})


class HRAnswer(BaseModel):
    """LLM output schema for an HR answer."""

    answer: str = Field(description="The answer to the HR question.")
    confidence: float = Field(ge=0, le=1, description="Confidence in the answer.")
    needs_escalation: bool = Field(description="Whether human HR intervention is needed.")
    relevant_sections: list[str] = Field(
        default_factory=list,
        description="Sections of the knowledge base used.",
    )


HR_PROMPT = ChatPromptTemplate.from_messages([
      (
          "system",
          (
              "You are an HR assistant for TalentBridge HR Solutions. Answer the "
              "following question using ONLY the approved knowledge base and "
              "the live workflow context provided below. "
              "If the question cannot be answered from those sources, say "
              "'I don't have enough information to answer that question. Please "
              "contact our HR team directly for assistance.'\n\n"
              "RULES:\n"
              "- Do NOT make up policies or procedures not in the knowledge base.\n"
              "- Do NOT give legal advice.\n"
              "- Do NOT make hiring decisions or promises.\n"
              "- If unsure, recommend escalation to human HR.\n\n"
              "KNOWLEDGE BASE:\n{knowledge_base}"
          ),
      ),
      (
          "human",
          (
              "Question: {query}\n\n"
              "LIVE WORKFLOW CONTEXT: {context}"
          ),
      ),
])


class HRAssistantAgent:
    """Answers recruitment FAQs and provides HR guidance."""

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
            knowledge_base_path: Path to the HR knowledge base markdown file.
        """
        self.llm = llm
        self.candidate_db = candidate_db or CandidateDatabase()
        self.kb_path = Path(knowledge_base_path) if knowledge_base_path else KNOWLEDGE_BASE_PATH
        self._kb_content: str | None = None

    def answer_query(
        self,
        input_data: HRAssistantInput,
        session_id: str | None = None,
    ) -> HRAssistantOutput:
        """Answer a candidate or recruiter HR question.

        Args:
            input_data: The query and optional conversation context.
            session_id: Optional session id used to tag the persisted answer.

        Returns:
            The answer with sources, confidence, and escalation flag.
        """
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

        kb_content = self._load_knowledge_base()
        context_str = str(input_data.context) if input_data.context else "None"
        if input_data.reflection_feedback:
            context_str = (
                f"{context_str}\n\nREFLECTION FEEDBACK (answer the part that was "
                f"previously left unaddressed):\n{input_data.reflection_feedback}"
            )

        structured_llm = self.llm.with_structured_output(HRAnswer)
        chain = HR_PROMPT | structured_llm
        result = chain.invoke({
            "knowledge_base": kb_content,
            "query": input_data.query,
            "context": context_str,
        })

        sources = []
        if result.relevant_sections:
            sources = result.relevant_sections
        elif self.kb_path.exists():
            sources = [self.kb_path.name]

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

    def _load_knowledge_base(self) -> str:
        """Load the HR knowledge base from disk (cached).

        Returns:
            The knowledge base content as a string.
        """
        if self._kb_content is not None:
            return self._kb_content

        if self.kb_path.exists():
            self._kb_content = self.kb_path.read_text(encoding="utf-8")
        else:
            logger.warning("HR knowledge base not found at %s", self.kb_path)
            self._kb_content = "No knowledge base available."
        return self._kb_content

    def _retrieve_policy_context(self, query: str) -> list[str]:
        """Retrieve relevant knowledge base sections for the query.

        Uses simple keyword matching against the knowledge base.
        For production, this could be replaced with vector search.

        Args:
            query: The user's question.

        Returns:
            List of relevant knowledge base sections.
        """
        kb = self._load_knowledge_base()
        sections = kb.split("\n## ")
        relevant = []
        query_lower = query.lower()
        for section in sections:
            section_lower = section.lower()
            query_words = query_lower.split()
            if any(w in section_lower for w in query_words if len(w) > 3):
                relevant.append(section.strip())
        return relevant

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
    result = agent.answer_query(HRAssistantInput(query=query))
    print(f"\nAnswer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Escalation needed: {result.needs_escalation}")
    print(f"Sources: {result.sources}")
