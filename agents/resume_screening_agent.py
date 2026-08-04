"""Resume Screening Agent — parses resumes and extracts structured data.

Single responsibility: Accept raw resume text, extract skills, experience,
education, and compute an initial match score against the job description.
Does NOT rank candidates across each other (that's Candidate Matching).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tools.resume_parser import ResumeParser
from tools.skill_normalizer import normalize_skill_list, skills_match
from utils.models import ResumeScreeningInput, ResumeScreeningOutput

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class ScreeningResult(BaseModel):
    """LLM output schema for screening a resume against a JD."""

    candidate_name: str = Field(description="Full name extracted from the resume.")
    skills: list[str] = Field(description="Technical and soft skills extracted.")
    experience_years: float = Field(description="Total years of professional experience.")
    education: list[dict] = Field(description="Education entries.")
    summary: str = Field(description="Brief screening summary (2-3 sentences).")
    match_score: float = Field(ge=0, le=100, description="Match score against the JD (0-100).")


SCREENING_PROMPT = ChatPromptTemplate.from_messages([
      (
          "system",
          (
              "You are a precise resume screening agent. Given a parsed resume and "
              "a job description, evaluate how well the candidate matches the role.\n\n"
              "CRITICAL RULES:\n"
              "- Return ONLY information explicitly stated in the resume.\n"
              "- Do NOT fabricate skills, experience, or qualifications.\n"
              "- Do NOT assume certifications or degrees not mentioned.\n"
              "- The match_score must reflect ONLY what is present in the resume.\n\n"
              "Return your response as a JSON object matching the ScreeningResult schema."
          ),
      ),
      (
          "human",
          (
              "PARSED RESUME:\n{resume_data}\n\n"
              "JOB DESCRIPTION:\n{jd_data}"
          ),
      ),
])


class ResumeScreeningAgent:
    """Parses a single resume and extracts structured screening data."""

    def __init__(self, llm: ChatOllama) -> None:
        """Initialize the Resume Screening Agent.

        Args:
            llm: The Ollama LLM instance used for extraction and scoring.
        """
        self.llm = llm
        self.parser = ResumeParser(llm)

    def screen_resume(
        self,
        input_data: ResumeScreeningInput,
        resume_filename: str | None = None,
    ) -> ResumeScreeningOutput:
        """Screen a single resume against the job description.

        Uses the ResumeParser to extract structured fields, then the LLM
        to compute a match score against the JD.  The extracted candidate
        and screening result are also persisted to SQLite (best-effort).

        Args:
            input_data: The resume text and parsed JD.
            resume_filename: Optional original resume file name.

        Returns:
            Structured screening output with extracted fields and match score.
        """
        parsed = self.parser.parse_text(input_data.resume_text)

        structured_llm = self.llm.with_structured_output(ScreeningResult)
        chain = SCREENING_PROMPT | structured_llm
        result = chain.invoke({
            "resume_data": str(parsed),
            "jd_data": str(input_data.job_description),
        })

        output = ResumeScreeningOutput(
            candidate_name=result.candidate_name,
            skills=result.skills,
            experience_years=result.experience_years,
            education=result.education,
            summary=result.summary,
            match_score=result.match_score,
            extracted_fields={
                "certifications": parsed.get("certifications", []),
                "past_roles": parsed.get("past_roles", []),
            },
        )

        # ── Persist to SQLite (never raises) ───────────────────────────
        try:
            from db.database import Database

            db = Database()
            jd_id = db.persist_job_description(input_data.job_description)
            candidate_id = db.persist_candidate(
                name=output.candidate_name,
                resume_raw_text=input_data.resume_text,
                resume_filename=resume_filename,
                skills=output.skills,
                experience_years=output.experience_years,
            )
            db.insert_screening(
                candidate_id=candidate_id,
                jd_id=jd_id,
                summary=output.summary,
                strengths=output.skills,
                gaps=[],
            )
        except Exception:
            logger.exception("Failed to persist resume screening result")

        return output

    def screen_resume_text(
        self,
        resume_text: str,
        jd_data: dict,
        resume_filename: str | None = None,
    ) -> ResumeScreeningOutput:
        """Convenience method: screen raw resume text against a parsed JD.

        Args:
            resume_text: Raw resume text content.
            jd_data: Parsed JD dict.
            resume_filename: Optional original resume file name.

        Returns:
            Structured screening output.
        """
        return self.screen_resume(ResumeScreeningInput(
            resume_text=resume_text,
            job_description=jd_data,
        ), resume_filename=resume_filename)

    def _calculate_match_score(
        self, skills: list[str], job_description: dict
    ) -> float:
        """Calculate initial match score against the JD (deterministic fallback).

        Args:
            skills: Extracted skills from the resume.
            job_description: Parsed JD with required skills.

        Returns:
            Match score between 0 and 100.
        """
        required = normalize_skill_list(job_description.get("required_skills", []))
        if not required:
            return 50.0

        skills = normalize_skill_list(skills)
        matched = sum(1 for required_skill in required if any(
            skills_match(required_skill, resume_skill) for resume_skill in skills
        ))
        return round((matched / len(required)) * 100, 1)


if __name__ == "__main__":
    import json
    import sys

    from utils.llm_factory import get_llm

    logging.basicConfig(level=logging.INFO)
    llm = get_llm()
    agent = ResumeScreeningAgent(llm)

    resume_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_resumes/sarah_chen.txt"
    jd_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_jd.txt"

    with open(resume_path, encoding="utf-8") as f:
        resume_text = f.read()
    with open(jd_path, encoding="utf-8") as f:
        jd_text = f.read()

    from tools.jd_analyzer import JDAnalyzer
    analyzer = JDAnalyzer(llm)
    jd_data = analyzer.analyze(jd_text)

    result = agent.screen_resume_text(resume_text, jd_data)
    print("\n=== Screening Result ===")
    print(json.dumps(result.model_dump(), indent=2))
