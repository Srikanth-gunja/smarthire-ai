"""Tests for EmailNotification — tests log-based stub behavior."""

from __future__ import annotations

import logging

import pytest

from tools.email_notification import EmailNotification
from utils.models import InterviewSlot


@pytest.fixture
def email_service():
    """Create an EmailNotification instance."""
    return EmailNotification()


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


class TestEmailNotificationInit:
    """Tests for EmailNotification initialization."""

    def test_init_default_host(self):
        """Default SMTP host is localhost."""
        service = EmailNotification()
        assert service.smtp_host == "localhost"

    def test_init_custom_host(self):
        """Custom SMTP host is stored."""
        service = EmailNotification(smtp_host="smtp.example.com", smtp_port=465)
        assert service.smtp_host == "smtp.example.com"
        assert service.smtp_port == 465


class TestSendInterviewInvite:
    """Tests for send_interview_invite method."""

    def test_returns_true(self, email_service, sample_slot):
        """send_interview_invite always returns True (stub)."""
        result = email_service.send_interview_invite(sample_slot, "test@email.com")
        assert result is True

    def test_logs_email(self, email_service, sample_slot, caplog):
        """send_interview_invite logs the email action."""
        with caplog.at_level(logging.INFO):
            email_service.send_interview_invite(sample_slot, "test@email.com")
        assert "EMAIL (stub)" in caplog.text
        assert "test@email.com" in caplog.text
        assert "Sarah Chen" in caplog.text


class TestSendStatusUpdate:
    """Tests for send_status_update method."""

    def test_returns_true(self, email_service):
        """send_status_update always returns True (stub)."""
        result = email_service.send_status_update(
            "Sarah Chen", "shortlisted", "test@email.com"
        )
        assert result is True

    def test_logs_email(self, email_service, caplog):
        """send_status_update logs the email action."""
        with caplog.at_level(logging.INFO):
            email_service.send_status_update(
                "Sarah Chen", "shortlisted", "test@email.com"
            )
        assert "EMAIL (stub)" in caplog.text
        assert "shortlisted" in caplog.text


class TestBuildEmailBody:
    """Tests for _build_email_body method."""

    def test_builds_body(self, email_service):
        """_build_email_body formats template with data."""
        template = "Dear {name}, your status is {status}."
        data = {"name": "Sarah", "status": "shortlisted"}
        result = email_service._build_email_body(template, data)
        assert result == "Dear Sarah, your status is shortlisted."

    def test_builds_empty_data(self, email_service):
        """_build_email_body handles template with no placeholders."""
        template = "No placeholders here."
        result = email_service._build_email_body(template, {})
        assert result == "No placeholders here."
