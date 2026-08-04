"""Tests for CalendarTool — tests slot management with JSON store."""

from __future__ import annotations

import pytest

from tools.calendar_tool import CalendarTool
from utils.models import InterviewSlot


@pytest.fixture
def calendar(tmp_path):
    """Create a CalendarTool with a temporary JSON store."""
    store_path = tmp_path / "test_slots.json"
    return CalendarTool(store_path=str(store_path))


@pytest.fixture
def sample_slot():
    """Return a sample InterviewSlot."""
    return InterviewSlot(
        candidate_name="Sarah Chen",
        date="2025-02-10",
        time_start="10:00",
        time_end="11:00",
        interviewer="Alice Manager",
        interview_type="technical",
        status="proposed",
    )


class TestCalendarToolInit:
    """Tests for CalendarTool initialization."""

    def test_init_creates_store_file(self, tmp_path):
        """Initialization creates the JSON store file."""
        store_path = tmp_path / "new_store.json"
        CalendarTool(store_path=str(store_path))
        assert store_path.exists()

    def test_init_empty_store(self, calendar):
        """New store starts with empty slot list."""
        slots = calendar._load_slots()
        assert slots == []


class TestBookSlot:
    """Tests for book_slot method."""

    def test_book_slot_success(self, calendar, sample_slot):
        """Booking a free slot returns confirmed status."""
        result = calendar.book_slot(sample_slot)
        assert result["status"] == "confirmed"
        assert result["slot"]["candidate_name"] == "Sarah Chen"

    def test_book_slot_persists(self, calendar, sample_slot):
        """Booked slot is persisted in the JSON store."""
        calendar.book_slot(sample_slot)
        bookings = calendar._load_slots()
        assert len(bookings) == 1
        assert bookings[0]["candidate_name"] == "Sarah Chen"

    def test_book_slot_conflict(self, calendar, sample_slot):
        """Booking a conflicting slot returns conflict status."""
        calendar.book_slot(sample_slot)

        conflict_slot = InterviewSlot(
            candidate_name="James Okafor",
            date="2025-02-10",
            time_start="10:30",
            time_end="11:30",
            interviewer="Alice Manager",
            interview_type="behavioral",
            status="proposed",
        )
        result = calendar.book_slot(conflict_slot)
        assert result["status"] == "conflict"

    def test_book_different_interviewer_no_conflict(self, calendar, sample_slot):
        """Slots for different interviewers don't conflict."""
        calendar.book_slot(sample_slot)

        different_slot = InterviewSlot(
            candidate_name="James Okafor",
            date="2025-02-10",
            time_start="10:00",
            time_end="11:00",
            interviewer="Bob Tech Lead",
            interview_type="behavioral",
            status="proposed",
        )
        result = calendar.book_slot(different_slot)
        assert result["status"] == "confirmed"

    def test_book_different_date_no_conflict(self, calendar, sample_slot):
        """Slots on different dates don't conflict."""
        calendar.book_slot(sample_slot)

        different_slot = InterviewSlot(
            candidate_name="James Okafor",
            date="2025-02-11",
            time_start="10:00",
            time_end="11:00",
            interviewer="Alice Manager",
            interview_type="behavioral",
            status="proposed",
        )
        result = calendar.book_slot(different_slot)
        assert result["status"] == "confirmed"


class TestCheckAvailability:
    """Tests for check_availability method."""

    def test_available_slot(self, calendar, sample_slot):
        """Returns True when slot is available."""
        available = calendar.check_availability(
            "2025-02-10", "10:00", "11:00", "Alice Manager"
        )
        assert available is True

    def test_unavailable_slot(self, calendar, sample_slot):
        """Returns False when slot is booked."""
        calendar.book_slot(sample_slot)
        available = calendar.check_availability(
            "2025-02-10", "10:00", "11:00", "Alice Manager"
        )
        assert available is False

    def test_available_different_interviewer(self, calendar, sample_slot):
        """Returns True for different interviewer on same time."""
        calendar.book_slot(sample_slot)
        available = calendar.check_availability(
            "2025-02-10", "10:00", "11:00", "Bob Tech Lead"
        )
        assert available is True


class TestGetExistingBookings:
    """Tests for get_existing_bookings method."""

    def test_get_all_bookings(self, calendar, sample_slot):
        """Returns all bookings when no date filter."""
        calendar.book_slot(sample_slot)
        bookings = calendar.get_existing_bookings()
        assert len(bookings) == 1

    def test_get_bookings_by_date(self, calendar, sample_slot):
        """Returns only bookings for the specified date."""
        calendar.book_slot(sample_slot)
        bookings = calendar.get_existing_bookings("2025-02-10")
        assert len(bookings) == 1
        bookings = calendar.get_existing_bookings("2025-02-11")
        assert len(bookings) == 0


class TestProposeAvailableSlots:
    """Tests for propose_available_slots method."""

    def test_propose_slots(self, calendar):
        """Proposes available slots within the time window."""
        slots = calendar.propose_available_slots(
            "2025-02-10", "Alice Manager",
            preferred_times=[{"time_start": "09:00", "time_end": "12:00"}],
            slot_duration_minutes=60,
            max_slots=3,
        )
        assert len(slots) > 0
        assert slots[0]["date"] == "2025-02-10"

    def test_propose_slots_respects_existing(self, calendar, sample_slot):
        """Proposed slots avoid existing bookings."""
        calendar.book_slot(sample_slot)
        slots = calendar.propose_available_slots(
            "2025-02-10", "Alice Manager",
            preferred_times=[{"time_start": "10:00", "time_end": "12:00"}],
            slot_duration_minutes=60,
            max_slots=3,
        )
        for slot in slots:
            assert not (slot["time_start"] < "11:00" and slot["time_end"] > "10:00")


class TestHasConflict:
    """Tests for _has_conflict method."""

    def test_no_conflict_empty_bookings(self, calendar, sample_slot):
        """No conflict when bookings list is empty."""
        assert calendar._has_conflict(sample_slot, []) is False

    def test_no_conflict_different_time(self, calendar):
        """No conflict when times don't overlap."""
        slot = InterviewSlot(
            candidate_name="A", date="2025-02-10",
            time_start="10:00", time_end="11:00",
            interviewer="Alice", interview_type="technical", status="proposed",
        )
        bookings = [{"interviewer": "Alice", "date": "2025-02-10",
                      "time_start": "14:00", "time_end": "15:00"}]
        assert calendar._has_conflict(slot, bookings) is False

    def test_conflict_overlapping(self, calendar):
        """Conflict detected when times overlap for same interviewer."""
        slot = InterviewSlot(
            candidate_name="A", date="2025-02-10",
            time_start="10:30", time_end="11:30",
            interviewer="Alice", interview_type="technical", status="proposed",
        )
        bookings = [{"interviewer": "Alice", "date": "2025-02-10",
                      "time_start": "10:00", "time_end": "11:00"}]
        assert calendar._has_conflict(slot, bookings) is True


class TestClearAll:
    """Tests for clear_all method."""

    def test_clear_all(self, calendar, sample_slot):
        """clear_all empties the store."""
        calendar.book_slot(sample_slot)
        assert len(calendar._load_slots()) == 1
        calendar.clear_all()
        assert len(calendar._load_slots()) == 0
