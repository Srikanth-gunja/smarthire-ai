"""Pydantic input/output models for all SmartHire AI agents.

Every agent, the Supervisor, and the Reflection Node has dedicated Input and
Output Pydantic models. These serve as the contract between components:
- Agents validate their inputs before processing.
- Agents return structured outputs that downstream consumers can trust.
- The Supervisor and Reflection Node use these models to orchestrate and
  validate the graph's behavior.

All models use Pydantic v2 (BaseModel). Fields use explicit types and
docstrings for self-documentation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════
# Supervisor Models
# ═══════════════════════════════════════════════════════════════════════


class SupervisorInput(BaseModel):
    """Input to the Supervisor: the raw user query and prior conversation."""

    user_query: str = Field(description="The raw message from the recruiter or candidate.")
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Serialized prior messages for context.",
    )


class ExecutionPlan(BaseModel):
    """Output of the Supervisor: the routing decision for this turn."""

    intent: str = Field(
        description=(
            "Classified intent, e.g. 'resume_screening', 'candidate_matching', "
            "'interview_scheduling', 'hr_question', 'multi_intent', 'greeting'."
        )
    )
    agents_to_invoke: list[str] = Field(
        description="Ordered list of agent names to execute (e.g. ['resume_screening', 'candidate_matching'])."
    )
    reasoning: str = Field(
        description="Brief explanation of why these agents were chosen."
    )


# ═══════════════════════════════════════════════════════════════════════
# Resume Screening Agent Models
# ═══════════════════════════════════════════════════════════════════════


class ResumeScreeningInput(BaseModel):
    """Input to the Resume Screening Agent."""

    resume_text: str = Field(description="Raw resume text or extracted content.")
    job_description: dict = Field(description="Parsed JD for context matching.")


class ResumeScreeningOutput(BaseModel):
    """Structured output of the Resume Screening Agent."""

    candidate_name: str = Field(description="Full name extracted from the resume.")
    skills: list[str] = Field(description="Technical and soft skills extracted.")
    experience_years: float = Field(description="Total years of professional experience.")
    education: list[dict] = Field(
        description="List of education entries, each with degree, institution, year."
    )
    summary: str = Field(description="Brief screening summary (2-3 sentences).")
    match_score: float = Field(
        ge=0, le=100, description="Initial match score against the JD (0-100)."
    )
    extracted_fields: dict = Field(
        default_factory=dict,
        description="Any additional structured data extracted from the resume.",
    )


# ═══════════════════════════════════════════════════════════════════════
# Candidate Matching Agent Models
# ═══════════════════════════════════════════════════════════════════════


class CandidateMatchingInput(BaseModel):
    """Input to the Candidate Matching Agent."""

    resumes: list[dict] = Field(description="List of screened resume data dicts.")
    job_description: dict = Field(description="Parsed JD with requirements.")
    reflection_feedback: str | None = Field(
        default=None,
        description=(
            "Optional correction guidance from the Reflection Node when the "
            "previous ranking pass failed validation."
        ),
    )


class RankedCandidate(BaseModel):
    """A single candidate's ranking result."""

    candidate_name: str = Field(description="Full name of the candidate.")
    match_score: float = Field(
        ge=0, le=100, description="Composite match score (0-100)."
    )
    skills_match: list[str] = Field(description="Skills that match the JD.")
    skills_gap: list[str] = Field(description="Skills required by JD but missing.")
    experience_match: bool = Field(
        description="Whether the candidate meets experience requirements."
    )
    justification: str = Field(
        description="Explanation of why this candidate received this ranking."
    )
    rank: int = Field(ge=1, description="Position in the ranked list (1 = top).")


class CandidateMatchingOutput(BaseModel):
    """Output of the Candidate Matching Agent."""

    ranked_candidates: list[RankedCandidate] = Field(
        description="Candidates ranked by match score, highest first."
    )
    total_candidates_evaluated: int = Field(ge=0, description="Total resumes processed.")
    summary: str = Field(description="Brief summary of the matching results.")


# ═══════════════════════════════════════════════════════════════════════
# Interview Scheduling Agent Models
# ═══════════════════════════════════════════════════════════════════════


class InterviewSlot(BaseModel):
    """A single proposed or confirmed interview slot."""

    candidate_name: str = Field(description="Candidate being interviewed.")
    date: str = Field(description="ISO format date (YYYY-MM-DD).")
    time_start: str = Field(description="Start time in HH:MM (24h) format.")
    time_end: str = Field(description="End time in HH:MM (24h) format.")
    interviewer: str = Field(description="Assigned interviewer name or ID.")
    interview_type: str = Field(
        description="Type of interview: 'phone', 'technical', 'behavioral', 'panel'."
    )
    status: str = Field(
        description="Slot status: 'proposed', 'confirmed', or 'conflict'."
    )


class InterviewSchedulingInput(BaseModel):
    """Input to the Interview Scheduling Agent."""

    candidates: list[str] = Field(description="Names of candidates to schedule.")
    availability: list[dict] = Field(
        description="Availability data per candidate (date/time windows)."
    )
    interviewer_preferences: dict = Field(
        default_factory=dict,
        description="Optional constraints (preferred times, interviewer assignments).",
    )
    reflection_feedback: str | None = Field(
        default=None,
        description=(
            "Optional correction guidance from the Reflection Node when the "
            "previous scheduling pass contained conflicting slots."
        ),
    )


class InterviewSchedulingOutput(BaseModel):
    """Output of the Interview Scheduling Agent."""

    proposed_slots: list[InterviewSlot] = Field(
        description="Proposed interview slots."
    )
    conflicts: list[str] = Field(
        default_factory=list,
        description="Human-readable descriptions of any scheduling conflicts.",
    )
    summary: str = Field(description="Brief summary of scheduling results.")


# ═══════════════════════════════════════════════════════════════════════
# HR Assistant Agent Models
# ═══════════════════════════════════════════════════════════════════════


class HRAssistantInput(BaseModel):
    """Input to the HR Assistant Agent."""

    query: str = Field(description="The candidate or recruiter question.")
    context: dict = Field(
        default_factory=dict,
        description="Optional conversation context for grounding the answer.",
    )
    reflection_feedback: str | None = Field(
        default=None,
        description=(
            "Optional correction guidance from the Reflection Node when the "
            "previous answer was flagged as incomplete."
        ),
    )


class HRAssistantOutput(BaseModel):
    """Output of the HR Assistant Agent."""

    answer: str = Field(description="The guidance or answer to the query.")
    sources: list[str] = Field(
        default_factory=list,
        description="Policy references or knowledge sources used.",
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence level in the answer (0-1)."
    )
    needs_escalation: bool = Field(
        description="Whether the question requires human HR intervention."
    )


# ═══════════════════════════════════════════════════════════════════════
# Reflection Node Models
# ═══════════════════════════════════════════════════════════════════════


class ReflectionInput(BaseModel):
    """Input to the Reflection Node: all agent outputs for validation."""

    candidate_rankings: list[dict] = Field(
        default_factory=list, description="From Candidate Matching Agent."
    )
    interview_slots: list[dict] = Field(
        default_factory=list, description="From Interview Scheduling Agent."
    )
    hr_answers: list[dict] = Field(
        default_factory=list, description="From HR Assistant Agent."
    )
    job_description: dict = Field(description="Original job description.")
    user_query: str = Field(description="Original user question.")


class ReflectionOutput(BaseModel):
    """Output of the Reflection Node: validation result and revised data."""

    validation_passed: bool = Field(
        description="Whether all validation checks passed."
    )
    issues_found: list[str] = Field(
        default_factory=list, description="Specific issues detected."
    )
    revised_rankings: list[dict] | None = Field(
        default=None, description="Corrected rankings if issues were found."
    )
    revised_slots: list[dict] | None = Field(
        default=None, description="Corrected slots if conflicts were found."
    )
    clarity_notes: str = Field(
        default="", description="Improvements made to response clarity."
    )
    final_response: str = Field(
        description="The polished, validated response for the user."
    )
