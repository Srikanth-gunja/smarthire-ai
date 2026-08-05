"""Interview Scheduling Agent — proposes and manages interview slots.

Single responsibility: Given candidates and availability data, propose
non-conflicting interview slots, detect conflicts, and assign interviewers.
Does NOT screen resumes, rank candidates, or answer HR questions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tools.calendar_tool import CalendarTool
from utils.models import (
    InterviewSchedulingInput,
    InterviewSchedulingOutput,
    InterviewSlot,
)

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

DEFAULT_INTERVIEWERS = ["Alice Manager", "Bob Tech Lead", "Carol Director"]
DEFAULT_INTERVIEW_TYPES = ["phone", "technical", "behavioral"]


class InterviewSchedulerAgent:
    """Proposes and manages interview scheduling slots."""

    def __init__(self, llm: ChatOllama, calendar: CalendarTool | None = None) -> None:
        """Initialize the Interview Scheduling Agent.

        Args:
            llm: The Ollama LLM instance (available for future optimization).
            calendar: CalendarTool instance for slot management.
        """
        self.llm = llm
        self.calendar = calendar or CalendarTool()

    def propose_schedule(
        self,
        input_data: InterviewSchedulingInput,
        session_id: str | None = None,
    ) -> InterviewSchedulingOutput:
        """Propose interview slots for the given candidates.

        For each candidate, finds available slots and books them.
        Detects and flags any conflicts.

        Args:
            input_data: Candidates, availability, and interviewer preferences.
            session_id: Optional session identifier for filtering interviews
                in the Session Interviews tab.

        Returns:
            Proposed slots with any conflicts identified.
        """
        proposed_slots: list[InterviewSlot] = []
        conflicts: list[str] = []

        interviewers = input_data.interviewer_preferences.get(
            "interviewers", DEFAULT_INTERVIEWERS
        )
        interview_types = input_data.interviewer_preferences.get(
            "interview_types", DEFAULT_INTERVIEW_TYPES
        )

        for i, candidate_name in enumerate(input_data.candidates):
            candidate_availability = self._get_availability_for_candidate(
                candidate_name, input_data.availability
            )

            if not candidate_availability:
                logger.warning("No availability data for %s", candidate_name)
                continue

            interview_type = interview_types[i % len(interview_types)]
            interviewer = interviewers[i % len(interviewers)]

            available_slots = self._find_available_slots(
                candidate_availability, interviewer
            )

            if not available_slots:
                conflicts.append(
                    f"No available slots found for {candidate_name} "
                    f"with interviewer {interviewer}"
                )
                continue

            slot_info = available_slots[0]
            slot = InterviewSlot(
                candidate_name=candidate_name,
                date=slot_info["date"],
                time_start=slot_info["time_start"],
                time_end=slot_info["time_end"],
                interviewer=interviewer,
                interview_type=interview_type,
                status="proposed",
            )

            booking_result = self.calendar.book_slot(slot)
            if booking_result["status"] == "conflict":
                slot.status = "conflict"
                conflicts.append(
                    f"Conflict: {candidate_name} with {interviewer} on "
                    f"{slot.date} {slot.time_start}-{slot.time_end}"
                )
            else:
                slot.status = "confirmed"

            proposed_slots.append(slot)

        new_conflicts = self._detect_conflicts(proposed_slots)
        conflicts.extend(new_conflicts)

        summary = self._build_summary(proposed_slots, conflicts)

        # A reflection-feedback retry re-proposes slots; the calendar-backed
        # booking naturally avoids the previously conflicting times.
        if input_data.reflection_feedback:
            logger.info(
                "Scheduling correction pass with reflection feedback: %s",
                input_data.reflection_feedback[:200],
            )
            summary += (
                " Re-proposed after the Reflection Node flagged conflicts in "
                "the previous attempt."
            )

        output = InterviewSchedulingOutput(
            proposed_slots=proposed_slots,
            conflicts=conflicts,
            summary=summary,
        )

        # ── Persist to SQLite (never raises) ───────────────────────────
        try:
            from db.database import Database

            db = Database()
            jd_id = db.get_recent_job_description_id()
            for slot in proposed_slots:
                candidate_id = db.find_candidate_by_name(slot.candidate_name)
                proposed_start = f"{slot.date} {slot.time_start}:00"
                proposed_end = f"{slot.date} {slot.time_end}:00"
                db.insert_interview(
                    candidate_id=candidate_id,
                    jd_id=jd_id,
                    proposed_start=proposed_start,
                    proposed_end=proposed_end,
                    status=slot.status,
                    session_id=session_id,
                    interview_type=slot.interview_type,
                    interviewer=slot.interviewer,
                )
        except Exception:
            logger.exception("Failed to persist interview scheduling result")

        return output

    def _get_availability_for_candidate(
        self, candidate_name: str, availability: list[dict]
    ) -> list[dict]:
        """Get availability data for a specific candidate.

        Args:
            candidate_name: Name of the candidate.
            availability: List of availability dicts.

        Returns:
            List of availability entries for the candidate.
        """
        return [
            a for a in availability
            if a.get("candidate_name") == candidate_name
        ]

    def _find_available_slots(
        self,
        candidate_availability: list[dict],
        interviewer: str,
        slot_duration_minutes: int = 60,
        max_slots: int = 3,
    ) -> list[dict]:
        """Find available slots from candidate availability data.

        Args:
            candidate_availability: List of availability entries.
            interviewer: Interviewer to check availability for.
            slot_duration_minutes: Duration of each slot.
            max_slots: Maximum slots to propose.

        Returns:
            List of available slot dicts.
        """
        all_available: list[dict] = []
        for avail in candidate_availability:
            date = avail.get("date", "")
            preferred_times = avail.get("preferred_times", [{"time_start": "09:00", "time_end": "17:00"}])

            slots = self.calendar.propose_available_slots(
                date=date,
                interviewer=interviewer,
                preferred_times=preferred_times,
                slot_duration_minutes=slot_duration_minutes,
                max_slots=max_slots - len(all_available),
            )
            all_available.extend(slots)
            if len(all_available) >= max_slots:
                break

        return all_available[:max_slots]

    def _detect_conflicts(self, slots: list[InterviewSlot]) -> list[str]:
        """Detect scheduling conflicts among proposed slots.

        Args:
            slots: Proposed interview slots to check.

        Returns:
            List of human-readable conflict descriptions.
        """
        conflicts: list[str] = []
        for i, slot_a in enumerate(slots):
            for slot_b in slots[i + 1:]:
                if (
                    slot_a.interviewer == slot_b.interviewer
                    and slot_a.date == slot_b.date
                    and slot_a.time_start < slot_b.time_end
                    and slot_a.time_end > slot_b.time_start
                ):
                    conflicts.append(
                        f"Overlap: {slot_a.candidate_name} and "
                        f"{slot_b.candidate_name} with {slot_a.interviewer} "
                        f"on {slot_a.date}"
                    )
                if (
                    slot_a.candidate_name == slot_b.candidate_name
                    and slot_a.date == slot_b.date
                    and slot_a.time_start < slot_b.time_end
                    and slot_a.time_end > slot_b.time_start
                ):
                    conflicts.append(
                        f"Candidate conflict: {slot_a.candidate_name} "
                        f"scheduled for overlapping slots on {slot_a.date}"
                    )
        return conflicts

    def _assign_interviewers(
        self, slots: list[InterviewSlot], preferences: dict
    ) -> list[InterviewSlot]:
        """Assign interviewers to slots based on preferences.

        Args:
            slots: Slots needing interviewer assignment.
            preferences: Interviewer preference constraints.

        Returns:
            Slots with interviewers assigned.
        """
        interviewers = preferences.get("interviewers", DEFAULT_INTERVIEWERS)
        for i, slot in enumerate(slots):
            if not slot.interviewer or slot.interviewer == "":
                slot.interviewer = interviewers[i % len(interviewers)]
        return slots

    def _build_summary(self, slots: list[InterviewSlot], conflicts: list[str]) -> str:
        """Build a human-readable summary of the scheduling results.

        Args:
            slots: All proposed slots.
            conflicts: All detected conflicts.

        Returns:
            Summary string.
        """
        confirmed = sum(1 for s in slots if s.status == "confirmed")
        total = len(slots)
        if not slots:
            return "No interview slots could be proposed."
        parts = [f"Proposed {total} interview slot(s), {confirmed} confirmed."]
        if conflicts:
            parts.append(f"Found {len(conflicts)} conflict(s) that need resolution.")
        return " ".join(parts)


if __name__ == "__main__":
    import json

    from utils.llm_factory import get_llm

    logging.basicConfig(level=logging.INFO)
    llm = get_llm()
    calendar = CalendarTool("data/test_interview_slots.json")
    agent = InterviewSchedulerAgent(llm, calendar)

    input_data = InterviewSchedulingInput(
        candidates=["Sarah Chen", "James Okafor"],
        availability=[
            {
                "candidate_name": "Sarah Chen",
                "date": "2025-02-10",
                "preferred_times": [{"time_start": "10:00", "time_end": "12:00"}],
            },
            {
                "candidate_name": "James Okafor",
                "date": "2025-02-10",
                "preferred_times": [{"time_start": "14:00", "time_end": "16:00"}],
            },
        ],
        interviewer_preferences={
            "interviewers": ["Alice Manager", "Bob Tech Lead"],
            "interview_types": ["technical", "behavioral"],
        },
    )

    result = agent.propose_schedule(input_data)
    print("\n=== Scheduling Result ===")
    print(json.dumps(result.model_dump(), indent=2))

    calendar.clear_all()
