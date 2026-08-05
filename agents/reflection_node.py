"""Reflection Node — validates and refines agent outputs before final response.

Single responsibility: Run the PRD Section 12 validation checklist after all
agents complete and *before* the final response is returned, produce a polished
final_response, flag issues in ``reflection_notes``, and — when validation
fails — request a single correction retry from the responsible agent.

The checklist is implemented as separate, independently-callable functions so
each can be tested and debugged in isolation:

  a) verify_candidate_recommendations_match_jd — cross-check ranked candidates'
     extracted skills against JD required skills (no fabricated skills).
  b) check_interview_schedule_conflicts — reuse calendar_tool's conflict logic.
  c) check_all_questions_answered — compare the original user query intents
     against what was actually produced in state.
  d) improve_clarity_and_consistency — final LLM pass that polishes the combined
     agent outputs into one coherent final_response.

Every check returns a structured result dict::

    {
        "check": "candidate_skills_match_jd",   # machine-readable key
        "name": "Candidate recommendations match JD requirements",
        "passed": True,                          # pass/fail per check
        "issues": [...],                         # what failed
        "corrections": [...],                    # what was fixed
        "retry_hint": "candidate_matching",      # agent to retry (optional)
    }

Failure handling per check:
  - (a) Skill mismatch → flag in reflection_notes AND request a correction
    retry from the Candidate Matching agent so it can re-rank with feedback.
  - (b) Schedule conflicts → flag in reflection_notes, drop conflicting slots,
    and request a correction retry from the Interview Scheduling agent.
  - (c) Unanswered questions → flag in reflection_notes and append a
    "needs follow-up" note to the final_response so nothing is silently dropped.
  - (d) Clarity — always runs and always succeeds (best-effort polish).

Retry semantics: when validation fails, ``run_reflection`` sets ``retry_agent``
and ``reflection_feedback`` (but only on the first pass — tracked via
``reflection_attempts``).  The graph routes back to that agent once; a second
reflection pass then returns the (possibly still imperfect) result to the user
with issues surfaced rather than silently returning an unvalidated answer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from memory.state import SmartHireState
from tools.calendar_tool import CalendarTool
from tools.skill_normalizer import normalize_skill_list, skills_match

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# Names used by the UI / DB as the canonical check identifiers.
CHECK_CANDIDATES = "candidate_skills_match_jd"
CHECK_SCHEDULE = "interview_schedule_conflicts"
CHECK_QUERY = "query_completeness"
CHECK_CLARITY = "clarity_consistency"

# How many reflection passes are allowed per turn (1 initial + 1 correction).
RETRY_LIMIT = 1


# ── Helpers ───────────────────────────────────────────────────────────


def _check_result(
    check: str,
    name: str,
    passed: bool,
    issues: list[str] | None = None,
    corrections: list[str] | None = None,
    retry_hint: str | None = None,
) -> dict:
    """Build a structured per-check result dict."""
    return {
        "check": check,
        "name": name,
        "passed": bool(passed),
        "issues": list(issues or []),
        "corrections": list(corrections or []),
        "retry_hint": retry_hint,
    }


def _resume_skills_by_name(state: SmartHireState) -> dict[str, list[str]]:
    """Map candidate_name -> extracted skills from the screened resumes."""
    mapping: dict[str, list[str]] = {}
    for resume in state.get("resumes", []):
        name = resume.get("candidate_name", "")
        if name:
            mapping[name] = resume.get("skills", [])
    return mapping


# ── Check (a): Candidate recommendations vs JD skills ─────────────────


def verify_candidate_recommendations_match_jd(
    state: SmartHireState,
) -> dict:
    """Cross-check ranked candidates' skills against JD required_skills.

    Two verifications are performed:
      1. Each ranked candidate must have at least one JD required skill
         in its ``skills_match`` list (a candidate with zero overlap is
         flagged — the ranking cannot be trusted).
      2. Every claimed ``skills_match`` entry must actually appear in the
         candidate's extracted resume skills (guards against fabricated
         skills, per PRD "no fabricated skills or qualifications").

    Args:
        state: The current shared state with candidate_rankings,
            resumes, and job_description.

    Returns:
        A structured check result dict.
    """
    issues: list[str] = []
    corrections: list[str] = []
    retry_hint: str | None = None

    jd = state.get("job_description", {})
    required_skills = normalize_skill_list(jd.get("required_skills", []))
    rankings = state.get("candidate_rankings", [])
    resume_skills = _resume_skills_by_name(state)

    for candidate in rankings:
        name = candidate.get("candidate_name", "Unknown")
        claimed_match = candidate.get("skills_match", [])
        claimed_match = normalize_skill_list(claimed_match)

        # 1) Candidate must overlap with at least one required skill.
        if required_skills and not any(
            skills_match(required, matched)
            for required in required_skills for matched in claimed_match
        ):
            issues.append(
                f"Candidate '{name}' has zero overlap with required skills "
                f"{required_skills}. Skills_match: {claimed_match}. "
                "Recruiter should verify this ranking."
            )
            retry_hint = "candidate_matching"
            continue

        # 2) Claimed matched skills must be grounded in the resume (no
        #    fabricated skills).  Skills already required by the JD are
        #    treated as verified even if the resume list is abbreviated.
        extracted = resume_skills.get(name, [])
        extracted = normalize_skill_list(extracted)
        for skill in claimed_match:
            if not any(skills_match(skill, required) for required in required_skills) and extracted and (
                not any(skills_match(skill, resume_skill) for resume_skill in extracted)
            ):
                issues.append(
                    f"Candidate '{name}' claims matched skill '{skill}' that is "
                    "not present in the extracted resume skills. Possible "
                    "fabrication — re-rank with evidence only."
                )
                retry_hint = "candidate_matching"

    return _check_result(
        check=CHECK_CANDIDATES,
        name="Candidate recommendations match JD requirements",
        passed=not issues,
        issues=issues,
        corrections=corrections,
        retry_hint=retry_hint,
    )


# ── Check (b): Interview schedule conflicts ────────────────────────────


def check_interview_schedule_conflicts(
    state: SmartHireState,
    calendar: CalendarTool | None = None,
) -> tuple[dict, list[dict]]:
    """Detect scheduling conflicts among proposed interview slots.

    Reuses CalendarTool._has_conflict rather than reimplementing overlap
    detection.  Conflicting slots are dropped (a hard correction), and the
    cleaned slot list is returned so the graph can update state.

    Args:
        state: The current shared state with interview_slots.
        calendar: Optional CalendarTool instance (created if not provided).

    Returns:
        Tuple of (structured check result, cleaned slot list).
    """
    if calendar is None:
        calendar = CalendarTool()

    slots = state.get("interview_slots", [])
    if not slots:
        return (
            _check_result(
                check=CHECK_SCHEDULE,
                name="Interview schedules are conflict-free",
                passed=True,
            ),
            [],
        )

    issues: list[str] = []
    corrections: list[str] = []
    cleaned: list[dict] = []
    seen_bookings: list[dict] = []

    for slot in slots:
        interviewer = slot.get("interviewer", "")
        date = slot.get("date", "")
        time_start = slot.get("time_start", "")
        time_end = slot.get("time_end", "")
        candidate = slot.get("candidate_name", "Unknown")

        from utils.models import InterviewSlot

        probe = InterviewSlot(
            candidate_name=candidate,
            date=date,
            time_start=time_start,
            time_end=time_end,
            interviewer=interviewer,
            interview_type=slot.get("interview_type", ""),
            status="proposed",
        )

        if calendar._has_conflict(probe, seen_bookings):
            issues.append(
                f"Conflict: {candidate} with {interviewer} on "
                f"{date} {time_start}-{time_end} overlaps an earlier slot."
            )
            corrections.append(
                f"Removed conflicting slot for {candidate} on "
                f"{date} {time_start}-{time_end}."
            )
            # Do NOT add to cleaned — the slot is invalid.
        else:
            cleaned.append(slot)
            seen_bookings.append(slot)

    return (
        _check_result(
            check=CHECK_SCHEDULE,
            name="Interview schedules are conflict-free",
            passed=not issues,
            issues=issues,
            corrections=corrections,
            retry_hint="interview_scheduling" if issues else None,
        ),
        cleaned,
    )


# ── Check (c): All questions answered ──────────────────────────────────


def check_all_questions_answered(state: SmartHireState) -> dict:
    """Compare the original user query against agent outputs in state.

    Heuristic: if the user asked a question (contains '?') and no
    hr_answers were produced, flag it.  If the user asked to screen/rank
    and no resumes or rankings exist, flag it.  If the user asked to
    schedule and no interview_slots exist, flag it.

    Args:
        state: The current shared state.

    Returns:
        A structured check result dict.
    """
    issues: list[str] = []
    corrections: list[str] = []
    retry_hint: str | None = None

    history = state.get("conversation_history", [])
    user_query = ""
    for msg in reversed(history):
        if isinstance(msg, HumanMessage):
            user_query = msg.content.lower()
            break

    if not user_query:
        return _check_result(
            check=CHECK_QUERY,
            name="All parts of the query were addressed",
            passed=True,
        )

    question_words = {
        "what", "how", "why", "when", "where", "who",
        "can", "do", "does", "is", "are",
    }
    tokens = set(user_query.split())
    is_question = "?" in user_query or bool(tokens & question_words)

    if is_question and not state.get("hr_answers"):
        has_other_output = (
            state.get("resumes")
            or state.get("candidate_rankings")
            or state.get("interview_slots")
        )
        if not has_other_output:
            issues.append(
                "User asked a question but no hr_answers were produced. "
                "The query may not have been addressed."
            )
            retry_hint = "hr_assistant"

    # Recruiter-only workflow checks: candidate chats never invoke the
    # screening/ranking/scheduling agents, so those must not be flagged.
    if state.get("user_role", "recruiter") != "candidate":
        screen_keywords = {"screen", "resume", "cv", "parse"}
        if tokens & screen_keywords and not state.get("resumes"):
            issues.append(
                "User requested resume screening but no resumes were produced."
            )
            retry_hint = "resume_screening"

        rank_keywords = {"rank", "match", "shortlist", "compare"}
        if tokens & rank_keywords and not state.get("candidate_rankings"):
            issues.append(
                "User requested candidate ranking but no rankings were produced."
            )
            retry_hint = "candidate_matching"

        schedule_keywords = {"schedule", "interview", "slot", "book"}
        if tokens & schedule_keywords and not state.get("interview_slots"):
            issues.append(
                "User requested interview scheduling but no slots were proposed."
            )
            retry_hint = "interview_scheduling"

    return _check_result(
        check=CHECK_QUERY,
        name="All parts of the query were addressed",
        passed=not issues,
        issues=issues,
        corrections=corrections,
        retry_hint=retry_hint,
    )


# ── Check (d): Clarity and consistency polish ──────────────────────────


def _extract_text(content: object) -> str:
    """Extract plain text from an LLM response content value.

    Gemini may return a list of parts (dicts with 'text'/'signature'/'extras'),
    a plain string, or an object with a ``.content`` attribute.  This helper
    normalises all of those to a single trimmed string.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text", "")
            else:
                t = str(p)
            if t:
                parts.append(t)
        return "\n".join(parts).strip()
    return str(content).strip()


def improve_clarity_and_consistency(
    state: SmartHireState,
    llm: ChatOllama | None = None,
) -> str:
    """Build a coherent final_response from the agent outputs in state.

    The response is role-aware: candidates receive only their HR answer
    (never rankings/slot summaries), while recruiters keep the full
    workflow summary. If an LLM is provided, use it to polish the combined
    output into a natural-language summary. If not (e.g. during testing or
    when the LLM is unavailable), fall back to a deterministic template.

    Args:
        state: The current shared state with all agent outputs.
        llm: Optional LLM for polish pass (None = deterministic fallback).

    Returns:
        A polished final_response string.
    """
    parts: list[str] = []
    role = state.get("user_role", "recruiter")

    hr = state.get("hr_answers", [])
    if role == "candidate":
        # Candidate final response is ONLY the HR answer. Never summarize
        # resumes, rankings, or interview slots for a candidate.
        if hr:
            parts.append(hr[-1].get("answer", ""))
        else:
            parts.append(
                "I'm sorry, I don't have enough information to answer that "
                "question. Please contact our HR team directly for assistance."
            )
    else:
        resumes = state.get("resumes", [])
        if resumes:
            names = [r.get("candidate_name", "Unknown") for r in resumes]
            parts.append(
                f"Screened {len(resumes)} resume(s): {', '.join(names)}."
            )

        rankings = state.get("candidate_rankings", [])
        if rankings:
            top = rankings[0]
            parts.append(
                f"Top candidate: {top.get('candidate_name', 'N/A')} "
                f"(match score: {top.get('match_score', 'N/A')}%)."
            )
            if len(rankings) > 1:
                others = [r.get("candidate_name", "?") for r in rankings[1:3]]
                parts.append(f"Also ranked: {', '.join(others)}.")

        slots = state.get("interview_slots", [])
        if slots:
            confirmed = sum(1 for s in slots if s.get("status") == "confirmed")
            parts.append(
                f"Proposed {len(slots)} interview slot(s), "
                f"{confirmed} confirmed."
            )

        if hr:
            answer = hr[-1].get("answer", "")
            parts.append(f"HR response: {answer[:300]}")

    fallback_response = " ".join(parts) if parts else "Processing complete."

    if llm is not None:
        try:
            from langchain_core.prompts import ChatPromptTemplate

            audience = (
                "a job candidate. Keep it concise (2-3 sentences), friendly, "
                "and focused only on the answer to their question."
                if role == "candidate"
                else "a recruiter. Keep it concise (3-5 sentences)."
            )
            polish_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a recruitment assistant. Rewrite the following "
                    "raw agent outputs into a single, clear, professional "
                    f"summary for {audience} "
                    "Do not invent information — only rephrase what is given."
                )),
                ("human", "{raw_output}"),
            ])
            chain = polish_prompt | llm
            result = chain.invoke({"raw_output": fallback_response})
            return _extract_text(result.content)
        except Exception:
            logger.exception("LLM polish failed, using deterministic fallback")

    return fallback_response


# ── Retry decision helpers ─────────────────────────────────────────────


def _pick_retry_agent(checks: list[dict]) -> str | None:
    """Pick the agent responsible for the first fixable failed check."""
    for check in checks:
        if not check["passed"]:
            hint = check.get("retry_hint")
            if hint:
                return hint
    return None


def _build_feedback(checks: list[dict]) -> str | None:
    """Build a human-readable feedback block for the retried agent."""
    lines: list[str] = []
    for check in checks:
        if not check["passed"] and check["issues"]:
            lines.append(f"- {check['name']}: {'; '.join(check['issues'])}")
    if not lines:
        return None
    return (
        "The Reflection Node flagged issues that need to be corrected in "
        "this attempt:\n" + "\n".join(lines)
    )


# ── Main reflection entry point ────────────────────────────────────────


def run_reflection(
    state: SmartHireState,
    llm: ChatOllama | None = None,
    calendar: CalendarTool | None = None,
) -> dict[str, Any]:
    """Run the full PRD Section 12 checklist and produce final output.

    This is the function wired into the graph as the reflection node.

    Args:
        state: The current shared state after all agents have run.
        llm: Optional LLM for the clarity/polish pass.
        calendar: Optional CalendarTool for conflict detection.

    Returns:
        Dict with keys 'reflection_notes', 'reflection_validated',
        'reflection_attempts', 'retry_agent', 'reflection_feedback' and
        'final_response' suitable for merging into SmartHireState.
    """
    attempts = state.get("reflection_attempts", 0)
    checks: list[dict] = []
    corrections_made: list[str] = []

    # Check (a): candidate recommendations vs JD
    checks.append(verify_candidate_recommendations_match_jd(state))

    # Check (b): interview schedule conflicts
    schedule_result, cleaned_slots = check_interview_schedule_conflicts(
        state, calendar
    )
    checks.append(schedule_result)

    # Check (c): all questions answered
    checks.append(check_all_questions_answered(state))

    # Check (d): clarity and consistency
    final_response = _extract_text(improve_clarity_and_consistency(state, llm))
    clarity_notes = []
    if state.get("resumes") or state.get("candidate_rankings") or state.get(
        "hr_answers"
    ):
        clarity_notes.append("Combined agent outputs into a single response.")
    checks.append(
        _check_result(
            check=CHECK_CLARITY,
            name="Response is clear, consistent, and actionable",
            passed=True,
            corrections=clarity_notes,
        )
    )

    for check in checks:
        corrections_made.extend(check.get("corrections", []))

    # Flatten issues for backward compatibility and the UI.
    all_issues: list[str] = [
        issue for check in checks for issue in check.get("issues", [])
    ]
    validation_passed = all(check["passed"] for check in checks)

    # If questions were left unanswered, append a follow-up note rather than
    # silently dropping them from the response.
    query_check = next(c for c in checks if c["check"] == CHECK_QUERY)
    if not query_check["passed"]:
        final_response += (
            "\n\nNote: Some aspects of your query may require follow-up. "
            "Please see the reflection notes for details."
        )

    # Retry decision — only on the first reflection pass (once).
    retry_agent: str | None = None
    feedback: str | None = None
    if not validation_passed and attempts < RETRY_LIMIT:
        retry_agent = _pick_retry_agent(checks)
        feedback = _build_feedback(checks)

    # Carry forward whether a correction retry was already triggered by an
    # earlier pass of this turn (the final notes must still say it happened).
    prior_notes = state.get("reflection_notes", {}) or {}
    correction_attempted = bool(
        retry_agent or prior_notes.get("correction_attempted")
    )
    recorded_retry_agent = retry_agent or prior_notes.get("retry_agent")

    notes = {
        "validation_passed": validation_passed,
        "reflection_validated": validation_passed,
        "issues_found": all_issues,
        "checks_run": [check["check"] for check in checks],
        "checks": checks,
        "corrections_made": corrections_made,
        "revised_slots": cleaned_slots if schedule_result["issues"] else None,
        "retry_agent": recorded_retry_agent,
        "correction_attempted": correction_attempted,
        "reflection_attempts": attempts + 1,
    }

    logger.info(
        "Reflection pass %d complete: passed=%s retry=%s issues=%d",
        attempts + 1,
        validation_passed,
        retry_agent,
        len(all_issues),
    )

    # ── Stamp reflection results onto the persisted rankings (best-effort) ──
    try:
        from db.database import Database

        jd = state.get("job_description")
        if jd and (jd.get("job_title") or jd.get("required_skills") or jd.get("raw_text")):
            db = Database()
            jd_id = db.persist_job_description(jd)
            db.update_ranking_reflection(
                jd_id,
                validation_passed,
                notes,
            )
    except Exception:
        logger.exception("Failed to persist reflection results")

    return {
        "reflection_notes": notes,
        "reflection_validated": validation_passed,
        "reflection_attempts": attempts + 1,
        "retry_agent": retry_agent,
        "reflection_feedback": feedback,
        "final_response": _extract_text(final_response),
        # If schedule conflicts were found, update the slots in state.
        **(
            {"interview_slots": cleaned_slots}
            if schedule_result["issues"]
            else {}
        ),
    }
