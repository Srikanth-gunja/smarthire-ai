"""Email Notification — log-based stub for interview invitations and status updates.

Single responsibility: Log email actions that would be sent in production.
This supports the bonus "email notifications" scope without requiring real
SMTP infrastructure. Real email sending can be added in a future phase.
"""

from __future__ import annotations

import logging

from utils.models import InterviewSlot

logger = logging.getLogger(__name__)


class EmailNotification:
    """Log-based stub for sending interview invitations and status updates."""

    def __init__(self, smtp_host: str = "localhost", smtp_port: int = 587) -> None:
        """Initialize the Email Notification service.

        Args:
            smtp_host: SMTP server hostname (unused in stub mode).
            smtp_port: SMTP server port (unused in stub mode).
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    def send_interview_invite(
        self, slot: InterviewSlot, recipient_email: str
    ) -> bool:
        """Log an interview invitation that would be sent.

        Args:
            slot: The interview slot details.
            recipient_email: Email address of the recipient.

        Returns:
            Always True (simulates successful send).
        """
        logger.info(
            "EMAIL (stub): Interview invite to %s | Candidate: %s | "
            "Date: %s %s-%s | Type: %s | Interviewer: %s",
            recipient_email,
            slot.candidate_name,
            slot.date,
            slot.time_start,
            slot.time_end,
            slot.interview_type,
            slot.interviewer,
        )
        return True

    def send_status_update(
        self, candidate_name: str, status: str, recipient_email: str
    ) -> bool:
        """Log a status update email that would be sent.

        Args:
            candidate_name: Name of the candidate.
            status: New status to communicate.
            recipient_email: Email address of the recipient.

        Returns:
            Always True (simulates successful send).
        """
        logger.info(
            "EMAIL (stub): Status update to %s | Candidate: %s | Status: %s",
            recipient_email,
            candidate_name,
            status,
        )
        return True

    def _build_email_body(self, template: str, data: dict) -> str:
        """Build the email body from a template and data dict.

        Args:
            template: Email template string with {placeholder} syntax.
            data: Dict of values to substitute into the template.

        Returns:
            The formatted email body string.
        """
        return template.format(**data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    email = EmailNotification()

    slot = InterviewSlot(
        candidate_name="Sarah Chen",
        date="2025-02-10",
        time_start="10:00",
        time_end="11:00",
        interviewer="Alice Manager",
        interview_type="technical",
        status="proposed",
    )
    email.send_interview_invite(slot, "sarah.chen@email.com")
    email.send_status_update("Sarah Chen", "shortlisted", "sarah.chen@email.com")
