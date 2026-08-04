"""Calendar Tool — manages interview scheduling and availability checks.

Single responsibility: Check availability, detect conflicts, book slots,
and retrieve existing bookings. Uses a JSON-backed store at
data/interview_slots.json for persistence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from utils.models import InterviewSlot

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = "data/interview_slots.json"


class CalendarTool:
    """Manages interview calendar operations: availability, conflicts, booking."""

    def __init__(self, store_path: str = DEFAULT_STORE_PATH) -> None:
        """Initialize the Calendar Tool.

        Args:
            store_path: Path to the JSON file for slot persistence.
        """
        self.store_path = Path(store_path)
        self._ensure_store()

    def _ensure_store(self) -> None:
        """Create the JSON store file if it does not exist."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self.store_path.write_text("[]", encoding="utf-8")

    def _load_slots(self) -> list[dict]:
        """Load all booked slots from the JSON store.

        Returns:
            List of slot dicts.
        """
        data = self.store_path.read_text(encoding="utf-8").strip()
        if not data:
            return []
        return json.loads(data)

    def _save_slots(self, slots: list[dict]) -> None:
        """Save slots to the JSON store.

        Args:
            slots: List of slot dicts to persist.
        """
        self.store_path.write_text(
            json.dumps(slots, indent=2), encoding="utf-8"
        )

    def check_availability(self, date: str, time_start: str, time_end: str, interviewer: str) -> bool:
        """Check if a given date/time slot is available for an interviewer.

        Args:
            date: ISO format date (YYYY-MM-DD).
            time_start: Start time in HH:MM (24h) format.
            time_end: End time in HH:MM (24h) format.
            interviewer: Interviewer name or ID.

        Returns:
            True if the slot is available, False if occupied.
        """
        bookings = self._load_slots()
        proposed_slot = InterviewSlot(
            candidate_name="",
            date=date,
            time_start=time_start,
            time_end=time_end,
            interviewer=interviewer,
            interview_type="",
            status="proposed",
        )
        return not self._has_conflict(proposed_slot, bookings)

    def book_slot(self, slot: InterviewSlot) -> dict:
        """Book an interview slot in the calendar.

        Args:
            slot: The InterviewSlot to book.

        Returns:
            Dict with booking confirmation details.

        Raises:
            ValueError: If the slot conflicts with an existing booking.
        """
        bookings = self._load_slots()
        if self._has_conflict(slot, bookings):
            logger.warning(
                "Conflict detected for interviewer %s on %s %s-%s",
                slot.interviewer, slot.date, slot.time_start, slot.time_end,
            )
            slot.status = "conflict"
            return {"status": "conflict", "slot": slot.model_dump()}

        slot.status = "confirmed"
        bookings.append(slot.model_dump())
        self._save_slots(bookings)
        logger.info(
            "Booked slot: %s with %s on %s %s-%s",
            slot.interview_type, slot.candidate_name,
            slot.date, slot.time_start, slot.time_end,
        )
        return {"status": "confirmed", "slot": slot.model_dump()}

    def get_existing_bookings(self, date: str | None = None) -> list[dict]:
        """Retrieve existing bookings, optionally filtered by date.

        Args:
            date: ISO format date to filter by (None = all bookings).

        Returns:
            List of booking dicts.
        """
        bookings = self._load_slots()
        if date is not None:
            bookings = [b for b in bookings if b.get("date") == date]
        return bookings

    def propose_available_slots(
        self,
        date: str,
        interviewer: str,
        preferred_times: list[dict] | None = None,
        slot_duration_minutes: int = 60,
        max_slots: int = 3,
    ) -> list[dict]:
        """Propose N available slots for a given date and interviewer.

        Args:
            date: ISO format date (YYYY-MM-DD).
            interviewer: Interviewer name or ID.
            preferred_times: List of preferred time windows
                [{"time_start": "09:00", "time_end": "12:00"}, ...].
                Defaults to business hours 09:00-17:00.
            slot_duration_minutes: Duration of each slot in minutes.
            max_slots: Maximum number of slots to propose.

        Returns:
            List of available slot dicts.
        """
        if preferred_times is None:
            preferred_times = [{"time_start": "09:00", "time_end": "17:00"}]

        bookings = self._get_existing_bookings_for_interviewer(date, interviewer)
        available: list[dict] = []

        for window in preferred_times:
            start_h, start_m = map(int, window["time_start"].split(":"))
            end_h, end_m = map(int, window["time_end"].split(":"))
            current_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m

            while current_minutes + slot_duration_minutes <= end_minutes and len(available) < max_slots:
                slot_start = f"{current_minutes // 60:02d}:{current_minutes % 60:02d}"
                slot_end_mins = current_minutes + slot_duration_minutes
                slot_end = f"{slot_end_mins // 60:02d}:{slot_end_mins % 60:02d}"

                candidate_slot = InterviewSlot(
                    candidate_name="",
                    date=date,
                    time_start=slot_start,
                    time_end=slot_end,
                    interviewer=interviewer,
                    interview_type="",
                    status="proposed",
                )

                if not self._has_conflict(candidate_slot, bookings):
                    available.append({
                        "date": date,
                        "time_start": slot_start,
                        "time_end": slot_end,
                        "interviewer": interviewer,
                    })

                current_minutes += slot_duration_minutes

        return available

    def _get_existing_bookings_for_interviewer(self, date: str, interviewer: str) -> list[dict]:
        """Get bookings for a specific interviewer on a specific date.

        Args:
            date: ISO format date.
            interviewer: Interviewer name or ID.

        Returns:
            List of matching booking dicts.
        """
        bookings = self._load_slots()
        return [
            b for b in bookings
            if b.get("date") == date and b.get("interviewer") == interviewer
        ]

    def _has_conflict(self, slot: InterviewSlot, bookings: list[dict]) -> bool:
        """Check if a proposed slot conflicts with existing bookings.

        Args:
            slot: The proposed InterviewSlot.
            bookings: Existing bookings to check against.

        Returns:
            True if there is a time overlap for the same interviewer.
        """
        for booking in bookings:
            if booking.get("interviewer") != slot.interviewer:
                continue
            if booking.get("date") != slot.date:
                continue
            existing_start = booking.get("time_start", "")
            existing_end = booking.get("time_end", "")
            if existing_start < slot.time_end and existing_end > slot.time_start:
                return True
        return False

    def clear_all(self) -> None:
        """Clear all bookings from the store."""
        self._save_slots([])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tool = CalendarTool()

    result = tool.book_slot(InterviewSlot(
        candidate_name="Sarah Chen",
        date="2025-02-10",
        time_start="10:00",
        time_end="11:00",
        interviewer="Alice Manager",
        interview_type="technical",
        status="proposed",
    ))
    print(f"Booking result: {result}")

    available = tool.propose_available_slots("2025-02-10", "Alice Manager")
    print(f"Available slots: {available}")

    tool.clear_all()
    print("Calendar cleared.")
