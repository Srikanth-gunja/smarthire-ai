"""Candidate Matching Agent — ranks candidates against a job description.

Single responsibility: Take screened resume data and a parsed JD, compute
composite match scores, rank candidates, and produce justified rankings.
Does NOT parse resumes (that's Resume Screening), schedule interviews,
or answer HR questions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tools.skill_normalizer import normalize_skill_list, skills_match
from utils.models import (
    CandidateMatchingInput,
    CandidateMatchingOutput,
    RankedCandidate,
)

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class LLMRankedCandidate(BaseModel):
    """LLM output schema for a single candidate ranking."""

    candidate_name: str = Field(description="Full name of the candidate.")
    match_score: float = Field(ge=0, le=100, description="Composite match score (0-100).")
    skills_match: list[str] = Field(description="Skills that match the JD.")
    skills_gap: list[str] = Field(description="Skills required by JD but missing.")
    experience_match: bool = Field(description="Whether experience requirements are met.")
    justification: str = Field(description="Explanation of the ranking.")


class LLMMatchResult(BaseModel):
    """LLM output schema for the full matching result."""

    ranked_candidates: list[LLMRankedCandidate] = Field(
        description="Candidates ranked by match score, highest first."
    )
    summary: str = Field(description="Brief summary of the matching results.")


MATCHING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a precise candidate matching agent. Given screened resume "
            "data and a job description, rank candidates by how well they match.\n\n"
            "RULES:\n"
            "- Base rankings ONLY on the resume data and JD requirements provided.\n"
            "- Do NOT fabricate skills, experience, or qualifications.\n"
            "- Each candidate must have a specific justification tied to JD requirements.\n"
            "- Rank 1 = best match, rank N = weakest match.\n"
            "- If REFLECTION FEEDBACK is present, treat it as a correction "
            "request: fix every flagged issue in your re-ranking (e.g. re-verify "
            "skills against the resume data, drop fabricated claims, only rank "
            "candidates whose extracted skills overlap the JD requirements).\n\n"
            "Return your response as a JSON object matching the LLMMatchResult schema."
        ),
    ),
    (
        "human",
        (
            "SCREENED CANDIDATES:\n{candidates_data}\n\n"
            "JOB DESCRIPTION:\n{jd_data}\n\n"
            "REFLECTION FEEDBACK:\n{reflection_feedback}"
        ),
    ),
])


class CandidateMatchingAgent:
    """Ranks screened candidates against a job description."""

    def __init__(self, llm: ChatOllama) -> None:
        """Initialize the Candidate Matching Agent.

        Args:
            llm: The Ollama LLM instance used for scoring and justification.
        """
        self.llm = llm

    def rank_candidates(
        self, input_data: CandidateMatchingInput
    ) -> CandidateMatchingOutput:
        """Rank all screened candidates against the job description.

        Args:
            input_data: List of screened resumes and the parsed JD.

        Returns:
            Ranked candidate list with scores and justifications.
        """
        if not input_data.resumes:
            return CandidateMatchingOutput(
                ranked_candidates=[],
                total_candidates_evaluated=0,
                summary="No candidates to evaluate.",
            )

        structured_llm = self.llm.with_structured_output(LLMMatchResult)
        chain = MATCHING_PROMPT | structured_llm
        result = chain.invoke({
            "candidates_data": str(input_data.resumes),
            "jd_data": str(input_data.job_description),
            "reflection_feedback": input_data.reflection_feedback or "None",
        })

        ranked = self._reconcile_rankings(
            result.ranked_candidates, input_data.resumes, input_data.job_description
        )

        output = CandidateMatchingOutput(
            ranked_candidates=ranked,
            total_candidates_evaluated=len(input_data.resumes),
            summary=result.summary,
        )

        # ── Persist to SQLite (never raises) ───────────────────────────
        try:
            from db.database import Database

            db = Database()
            jd_id = db.persist_job_description(input_data.job_description)
            for candidate in ranked:
                candidate_id = db.find_candidate_by_name(candidate.candidate_name)
                if candidate_id is None:
                    # Fall back to persisting a candidate row from the
                    # screened resume data so the ranking FK is satisfied.
                    resume_data = next(
                        (
                            r for r in input_data.resumes
                            if r.get("candidate_name") == candidate.candidate_name
                        ),
                        {},
                    )
                    candidate_id = db.persist_candidate(
                        name=candidate.candidate_name,
                        resume_raw_text=str(resume_data),
                        skills=resume_data.get("skills"),
                        experience_years=resume_data.get("experience_years"),
                    )
                db.insert_candidate_ranking(
                    candidate_id=candidate_id,
                    jd_id=jd_id,
                    match_score=candidate.match_score,
                    rank_position=candidate.rank,
                    reasoning=candidate.justification,
                )
        except Exception:
            logger.exception("Failed to persist candidate ranking result")

        return output

    def _compute_skills_match(
        self, resume_skills: list[str], jd_skills: list[str]
    ) -> tuple[list[str], list[str]]:
        """Compute matched and missing skills between resume and JD.

        Args:
            resume_skills: Skills from the candidate's resume.
            jd_skills: Skills required by the job description.

        Returns:
            Tuple of (matched_skills, missing_skills).
        """
        resume_skills = normalize_skill_list(resume_skills)
        jd_skills = normalize_skill_list(jd_skills)
        matched = [skill for skill in jd_skills if any(
            skills_match(skill, resume_skill) for resume_skill in resume_skills
        )]
        missing = [skill for skill in jd_skills if skill not in matched]
        return matched, missing

    def _compute_composite_score(
        self,
        skills_match: list[str],
        skills_gap: list[str],
        experience_match: bool,
        jd: dict,
    ) -> float:
        """Compute a weighted composite match score (deterministic fallback).

        Weights:
            - Skills match: 60%
            - Experience match: 30%
            - Skills gap penalty: 10%

        Args:
            skills_match: Skills that matched.
            skills_gap: Skills that are missing.
            experience_match: Whether experience requirements are met.
            jd: The parsed job description for weighting.

        Returns:
            Composite score between 0 and 100.
        """
        total_required = len(skills_match) + len(skills_gap)
        if total_required == 0:
            skills_score = 50.0
        else:
            skills_score = (len(skills_match) / total_required) * 100

        experience_score = 100.0 if experience_match else 30.0

        composite = (skills_score * 0.7) + (experience_score * 0.3)

        return round(min(composite, 100.0), 1)

    def _generate_justification(
        self,
        candidate_name: str,
        skills_match: list[str],
        skills_gap: list[str],
        experience_match: bool,
        score: float,
    ) -> str:
        """Generate a human-readable justification for the ranking.

        Args:
            candidate_name: Name of the candidate.
            skills_match: Matched skills.
            skills_gap: Missing skills.
            experience_match: Whether experience requirements are met.
            score: The computed composite score.

        Returns:
            A 1-2 sentence justification string.
        """
        parts = [f"{candidate_name} has a match score of {score}%."]
        if skills_match:
            parts.append(f"Matches {len(skills_match)} required skill(s): {', '.join(skills_match[:5])}.")
        if skills_gap:
            parts.append(f"Missing {len(skills_gap)} skill(s): {', '.join(skills_gap[:3])}.")
        if experience_match:
            parts.append("Meets experience requirements.")
        else:
            parts.append("Does not meet experience requirements.")
        return " ".join(parts)

    def _reconcile_rankings(
        self, llm_candidates: list[LLMRankedCandidate], resumes: list[dict], jd: dict
    ) -> list[RankedCandidate]:
        """Reconcile LLM rankings with input data to produce RankedCandidate objects.

        Args:
            llm_candidates: List of candidates ranked by the LLM.
            resumes: Original screened resume data.
            jd: Parsed job description.

        Returns:
            List of RankedCandidate objects with computed ranks.
        """
        jd_required_skills = jd.get("required_skills", [])
        # The LLM can explain results, but it must not decide which resumes
        # exist or omit candidates from a batch.  Build one result per parsed
        # resume and derive match evidence deterministically from the inputs.
        ranked = []
        for resume_data in resumes:
            candidate_name = resume_data.get("candidate_name", "Unknown candidate")
            resume_skills = resume_data.get("skills", [])
            skills_match, skills_gap = self._compute_skills_match(resume_skills, jd_required_skills)
            experience_match = resume_data.get("experience_years", 0) >= jd.get("min_experience_years", 0)

            score = self._compute_composite_score(
                skills_match, skills_gap, experience_match, jd
            )
            justification = self._generate_justification(
                candidate_name, skills_match, skills_gap, experience_match, score
            )

            ranked.append(RankedCandidate(
                candidate_name=candidate_name,
                match_score=score,
                rank=1,
                skills_match=skills_match,
                skills_gap=skills_gap,
                experience_match=experience_match,
                justification=justification,
            ))
        ranked.sort(key=lambda candidate: candidate.match_score, reverse=True)
        return [
            candidate.model_copy(update={"rank": rank})
            for rank, candidate in enumerate(ranked, start=1)
        ]


if __name__ == "__main__":
    import json
    import sys

    from utils.llm_factory import get_llm

    logging.basicConfig(level=logging.INFO)
    llm = get_llm()
    agent = CandidateMatchingAgent(llm)

    jd_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_jd.txt"
    with open(jd_path, encoding="utf-8") as f:
        jd_text = f.read()

    from tools.jd_analyzer import JDAnalyzer
    analyzer = JDAnalyzer(llm)
    jd_data = analyzer.analyze(jd_text)

    from agents.resume_screening_agent import ResumeScreeningAgent
    screening_agent = ResumeScreeningAgent(llm)

    import glob
    resume_files = sorted(glob.glob("data/sample_resumes/*.txt"))
    screened = []
    for rf in resume_files:
        with open(rf, encoding="utf-8") as f:
            text = f.read()
        result = screening_agent.screen_resume_text(text, jd_data)
        screened.append(result.model_dump())
        print(f"Screened: {result.candidate_name} (score: {result.match_score})")

    match_input = CandidateMatchingInput(resumes=screened, job_description=jd_data)
    ranking = agent.rank_candidates(match_input)
    print("\n=== Candidate Rankings ===")
    print(json.dumps(ranking.model_dump(), indent=2))
