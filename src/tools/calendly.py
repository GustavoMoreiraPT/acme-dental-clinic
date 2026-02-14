"""LangChain tools for Calendly appointment management.

Each tool wraps a CalendlyClient method and returns a human-readable string
that the LLM can use to formulate its response to the patient.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

from langchain_core.tools import tool

from src.services.calendly_client import CalendlyAPIError, get_calendly_client

logger = logging.getLogger(__name__)

# RFC 5322-ish pattern — covers the vast majority of real-world emails
# without requiring an external dependency.
_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$"
)


def _validate_email(email: str) -> str | None:
    """Return an error message if *email* looks invalid, else ``None``."""
    if not email or not email.strip():
        return "No email address was provided. Please ask the patient for their email."
    email = email.strip()
    if not _EMAIL_RE.match(email):
        return (
            f'"{email}" does not look like a valid email address. '
            "Please ask the patient to double-check and provide a corrected email."
        )
    return None


def _format_dt(iso_str: str) -> str:
    """Convert an ISO 8601 string to a friendly 'Mon 17 Feb 2026 at 10:30' format."""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.strftime("%a %d %b %Y at %H:%M")


def _get_event_type_uri() -> str:
    """Retrieve the first active event type URI for the clinic (there is only one)."""
    client = get_calendly_client()
    event_types = client.get_event_types()
    if not event_types:
        raise CalendlyAPIError("No active event types found. Please check the Calendly configuration.")
    return event_types[0]["uri"]


# ── Tool 1: Check available appointment slots ───────────────────────


@tool
def get_available_slots(start_date: str, end_date: str) -> str:
    """Check available dental check-up appointment slots between two dates.

    Args:
        start_date: Start date in YYYY-MM-DD format (e.g. "2026-02-17").
        end_date: End date in YYYY-MM-DD format (e.g. "2026-02-21").
                  Must be within 7 days of start_date.
    """
    try:
        event_type_uri = _get_event_type_uri()
        start_iso = f"{start_date}T00:00:00Z"
        end_iso = f"{end_date}T23:59:59Z"

        slots = get_calendly_client().get_available_times(event_type_uri, start_iso, end_iso)

        if not slots:
            return f"No available slots found between {start_date} and {end_date}. Please try different dates."

        available = [s for s in slots if s.get("status") == "available"]
        if not available:
            return f"All slots between {start_date} and {end_date} are fully booked. Please try different dates."

        lines = [f"Available 30-minute check-up slots ({start_date} to {end_date}):\n"]
        for slot in available:
            lines.append(f"  • {_format_dt(slot['start_time'])}")

        return "\n".join(lines)

    except CalendlyAPIError as e:
        logger.error("Failed to get available slots: %s", e)
        return f"Sorry, I couldn't check availability right now. Error: {e}. Please try again in a moment."


# ── Tool 2: Create a new booking ────────────────────────────────────


@tool
def create_booking(start_time: str, full_name: str, email: str) -> str:
    """Book a dental check-up appointment for a patient.

    This directly creates the appointment via the Calendly Scheduling
    API (POST /invitees). The booking is confirmed immediately and
    Calendly sends a confirmation email automatically.

    Args:
        start_time: The exact appointment start time in ISO 8601 format
                    (e.g. "2026-02-17T10:30:00Z"). Must be one of the
                    available slots returned by get_available_slots.
        full_name: The patient's full name.
        email: The patient's email address.
    """
    email_error = _validate_email(email)
    if email_error:
        return email_error

    try:
        event_type_uri = _get_event_type_uri()
        client = get_calendly_client()

        # Create the booking directly via POST /invitees
        invitee = client.create_invitee(
            event_type_uri,
            start_time,
            name=full_name,
            email=email,
        )

        cancel_url = invitee.get("cancel_url", "N/A")
        reschedule_url = invitee.get("reschedule_url", "N/A")

        return (
            f"Appointment booked successfully!\n"
            f"  Patient: {full_name}\n"
            f"  Email: {email}\n"
            f"  Time: {_format_dt(start_time)}\n"
            f"  Duration: 30 minutes\n"
            f"  Cancel link: {cancel_url}\n"
            f"  Reschedule link: {reschedule_url}\n\n"
            f"Calendly will send a confirmation email to {email} "
            f"with all the details and a calendar invite."
        )

    except CalendlyAPIError as e:
        logger.error("Failed to create booking: %s", e)
        return f"Sorry, I couldn't complete the booking. Error: {e}. Please try again."


# ── Tool 3: Find an existing booking by email ───────────────────────


@tool
def find_booking(email: str) -> str:
    """Look up existing appointments for a patient by their email address.

    Args:
        email: The patient's email address used when booking.
    """
    email_error = _validate_email(email)
    if email_error:
        return email_error

    try:
        client = get_calendly_client()
        events = client.find_event_by_invitee_email(email)

        if not events:
            return (
                f"No active appointments found for {email}. "
                "The patient may not have a booking, or it may have been cancelled."
            )

        lines = [f"Found {len(events)} active appointment(s) for {email}:\n"]
        for event in events:
            event_uuid = event["uri"].split("/")[-1]
            start = _format_dt(event["start_time"])
            end = _format_dt(event["end_time"])
            status = event.get("status", "unknown")
            invitee_name = event.get("invitee", {}).get("name", "N/A")
            lines.append(
                f"  • Appointment ID: {event_uuid}\n"
                f"    Patient: {invitee_name}\n"
                f"    Time: {start} – {end}\n"
                f"    Status: {status}"
            )

        return "\n".join(lines)

    except CalendlyAPIError as e:
        logger.error("Failed to find booking: %s", e)
        return f"Sorry, I couldn't look up that booking. Error: {e}. Please try again."


# ── Tool 4: Cancel an existing booking ──────────────────────────────


@tool
def cancel_booking(event_uuid: str, reason: str = "Cancelled by patient via chat agent") -> str:
    """Cancel an existing dental appointment.

    Args:
        event_uuid: The appointment ID (UUID) to cancel.
                    Obtain this from find_booking first.
        reason: The reason for cancellation.
    """
    try:
        client = get_calendly_client()
        client.cancel_event(event_uuid, reason)

        return (
            f"Appointment {event_uuid} has been successfully cancelled.\n"
            f"Reason: {reason}\n\n"
            f"Calendly will send a cancellation confirmation email to the patient.\n"
            f"Note: Cancellations made less than 24 hours before the appointment "
            f"may incur a €20 late cancellation fee."
        )

    except CalendlyAPIError as e:
        logger.error("Failed to cancel booking: %s", e)
        return f"Sorry, I couldn't cancel the appointment. Error: {e}. Please try again."


# ── Tool 5: Reschedule an existing booking ──────────────────────────


@tool
def reschedule_booking(event_uuid: str, new_start_time: str) -> str:
    """Reschedule an existing dental appointment to a new time.

    This cancels the old appointment and shows the patient available slots
    near their requested new time. The patient should then create a new booking.

    Args:
        event_uuid: The appointment ID (UUID) to reschedule.
                    Obtain this from find_booking first.
        new_start_time: The desired new start time in ISO 8601 format
                        (e.g. "2026-02-20T14:00:00Z").
    """
    try:
        client = get_calendly_client()

        # Cancel the existing appointment
        client.cancel_event(event_uuid, reason="Rescheduled by patient via chat agent")

        # Check availability around the new requested time
        event_type_uri = _get_event_type_uri()
        new_dt = datetime.fromisoformat(new_start_time.replace("Z", "+00:00"))
        range_start = new_dt.replace(hour=0, minute=0, second=0).isoformat()
        range_end = (new_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0).isoformat()

        slots = client.get_available_times(event_type_uri, range_start, range_end)
        available = [s for s in slots if s.get("status") == "available"]

        result = (
            f"The original appointment ({event_uuid}) has been cancelled for rescheduling.\n"
            f"Calendly will send a cancellation notice for the old appointment.\n\n"
        )

        if available:
            result += f"Available slots on {new_dt.strftime('%a %d %b %Y')}:\n"
            for slot in available:
                result += f"  • {_format_dt(slot['start_time'])}\n"
            result += "\nPlease ask the patient which slot they'd like, then use create_booking to confirm."
        else:
            result += (
                f"Unfortunately, no slots are available on {new_dt.strftime('%a %d %b %Y')}. "
                f"Please ask the patient for alternative dates."
            )

        return result

    except CalendlyAPIError as e:
        logger.error("Failed to reschedule booking: %s", e)
        return f"Sorry, I couldn't reschedule the appointment. Error: {e}. Please try again."
