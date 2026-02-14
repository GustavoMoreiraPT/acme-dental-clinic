"""HTTP client for the Calendly API v2 with retry logic, timeout handling,
and an in-memory LRU cache.

Calendly API docs: https://developer.calendly.com/api-docs/
All requests require a Personal Access Token passed as a Bearer token.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from src.config import CALENDLY_API_TOKEN, CALENDLY_BASE_URL
from src.services.cache import LRUCache

logger = logging.getLogger(__name__)

# ── Retry configuration ─────────────────────────────────────────────
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 15.0

# ── Cache key prefixes ──────────────────────────────────────────────
_CK_EVENT_TYPES = "event_types"
_CK_EVENT_TYPE_LOC = "event_type_loc:"
_CK_EVENTS_BY_EMAIL = "events_by_email:"
_CK_INVITEES = "invitees:"


class CalendlyAPIError(Exception):
    """Raised when a Calendly API call fails after all retries."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class CalendlyClient:
    """Thin wrapper around the Calendly REST API v2 with automatic retries
    and an in-memory LRU cache (20 MB ceiling, evicts least-recently-used
    entries first).

    **Cache invalidation contract**

    Every *write* method (``create_invitee``, ``cancel_event``) invalidates
    the cached entries that the operation affects, so subsequent reads
    always see fresh data.  The cache is purely ephemeral — it is lost on
    process restart.
    """

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        *,
        cache: LRUCache | None = None,
    ):
        self._token = token or CALENDLY_API_TOKEN
        self._base_url = base_url or CALENDLY_BASE_URL
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        # Cache for the current user URI – fetched lazily once (never changes)
        self._user_uri: str | None = None
        # Shared LRU cache (injectable for tests)
        self._cache = cache or LRUCache()

    # ── Internal helpers ─────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an HTTP request with exponential-backoff retries."""
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.request(
                    method,
                    path,
                    params=params,
                    json=json_body,
                )
                if response.status_code >= 500:
                    raise CalendlyAPIError(
                        f"Server error {response.status_code}: {response.text}",
                        status_code=response.status_code,
                    )
                if response.status_code >= 400:
                    raise CalendlyAPIError(
                        f"Client error {response.status_code}: {response.text}",
                        status_code=response.status_code,
                    )
                return response.json()

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = exc
                logger.warning(
                    "Calendly API attempt %d/%d failed (%s). Retrying in %.1fs…",
                    attempt,
                    MAX_RETRIES,
                    type(exc).__name__,
                    INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                )
            except CalendlyAPIError as exc:
                if exc.status_code and exc.status_code >= 500:
                    last_error = exc
                    logger.warning(
                        "Calendly API server error on attempt %d/%d. Retrying…",
                        attempt,
                        MAX_RETRIES,
                    )
                else:
                    raise  # 4xx errors are not retried

            backoff = INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
            time.sleep(backoff)

        raise CalendlyAPIError(
            f"Calendly API request failed after {MAX_RETRIES} retries: {last_error}"
        )

    # ── Cache helpers ────────────────────────────────────────────────

    def _invalidate_events_for_email(self, email: str) -> None:
        """Remove cached event lists for *email* so the next read is fresh."""
        key = f"{_CK_EVENTS_BY_EMAIL}{email.lower()}"
        if self._cache.invalidate(key):
            logger.debug("Cache: invalidated %s", key)

    def _invalidate_events_for_event_uuid(self, event_uuid: str) -> None:
        """Reverse-lookup: find which email cache contains *event_uuid* and
        invalidate it.  This is used by ``cancel_event`` which only receives
        the UUID, not the patient email.
        """
        key = self._cache.find_key_containing_value(
            _CK_EVENTS_BY_EMAIL, event_uuid,
        )
        if key:
            self._cache.invalidate(key)
            logger.debug("Cache: invalidated %s (contains event %s)", key, event_uuid)
        # Also drop the invitees sub-cache for this event
        self._cache.invalidate(f"{_CK_INVITEES}{event_uuid}")

    # ── Public API methods ───────────────────────────────────────────

    def get_current_user_uri(self) -> str:
        """Return the URI of the authenticated Calendly user (cached)."""
        if self._user_uri is None:
            data = self._request("GET", "/users/me")
            self._user_uri = data["resource"]["uri"]
        return self._user_uri

    def get_event_types(self) -> list[dict[str, Any]]:
        """List all event types for the current user (cached)."""
        cached = self._cache.get(_CK_EVENT_TYPES)
        if cached is not None:
            return cached

        user_uri = self.get_current_user_uri()
        data = self._request(
            "GET", "/event_types", params={"user": user_uri, "active": "true"},
        )
        result = data.get("collection", [])
        self._cache.put(_CK_EVENT_TYPES, result)
        return result

    def get_available_times(
        self,
        event_type_uri: str,
        start_time: str,
        end_time: str,
    ) -> list[dict[str, Any]]:
        """Get available time slots for a given event type and date range.

        **Not cached** — availability changes in real-time and must always
        be fetched fresh.

        Args:
            event_type_uri: The URI of the event type.
            start_time: ISO 8601 start datetime (e.g. "2026-02-15T00:00:00Z").
            end_time: ISO 8601 end datetime (e.g. "2026-02-16T00:00:00Z").

        Returns:
            A list of available time slot dicts with 'start_time' and 'status'.
        """
        data = self._request(
            "GET",
            "/event_type_available_times",
            params={
                "event_type": event_type_uri,
                "start_time": start_time,
                "end_time": end_time,
            },
        )
        return data.get("collection", [])

    def list_scheduled_events(
        self,
        *,
        min_start_time: str | None = None,
        max_start_time: str | None = None,
        status: str = "active",
    ) -> list[dict[str, Any]]:
        """List scheduled events for the current user.

        Args:
            min_start_time: Optional ISO 8601 lower bound.
            max_start_time: Optional ISO 8601 upper bound.
            status: 'active' or 'canceled'.
        """
        user_uri = self.get_current_user_uri()
        params: dict[str, str] = {"user": user_uri, "status": status}
        if min_start_time:
            params["min_start_time"] = min_start_time
        if max_start_time:
            params["max_start_time"] = max_start_time
        data = self._request("GET", "/scheduled_events", params=params)
        return data.get("collection", [])

    def get_event_invitees(self, event_uuid: str) -> list[dict[str, Any]]:
        """List invitees for a specific scheduled event (cached).

        Args:
            event_uuid: The UUID of the scheduled event.
        """
        cache_key = f"{_CK_INVITEES}{event_uuid}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        data = self._request("GET", f"/scheduled_events/{event_uuid}/invitees")
        result = data.get("collection", [])
        self._cache.put(cache_key, result)
        return result

    def create_scheduling_link(self, event_type_uri: str, max_event_count: int = 1) -> str:
        """Create a single-use scheduling link for an event type.

        Returns:
            The booking URL that can be shared with the patient.
        """
        data = self._request(
            "POST",
            "/scheduling_links",
            json_body={
                "max_event_count": max_event_count,
                "owner": event_type_uri,
                "owner_type": "EventType",
            },
        )
        return data["resource"]["booking_url"]

    def create_invitee(
        self,
        event_type_uri: str,
        start_time: str,
        *,
        name: str,
        email: str,
        timezone: str = "Europe/London",
    ) -> dict[str, Any]:
        """Create a booking by adding an invitee to an event type slot.

        Uses the POST /invitees endpoint (Calendly Scheduling API) to
        programmatically book a meeting without redirecting the user.

        See: https://developer.calendly.com/schedule-events-with-ai-agents

        Args:
            event_type_uri: Full URI of the event type.
            start_time: ISO 8601 start time in UTC (e.g. "2026-02-17T09:00:00Z").
            name: Invitee full name.
            email: Invitee email address.
            timezone: IANA timezone string (default "Europe/London").

        Returns:
            The invitee resource dict from the Calendly API containing
            cancel_url, reschedule_url, event URI, etc.
        """
        # Build the location object from the event type configuration (cached)
        loc_key = f"{_CK_EVENT_TYPE_LOC}{event_type_uri}"
        locations = self._cache.get(loc_key)
        if locations is None:
            path = event_type_uri.replace(self._base_url, "")
            et_data = self._request("GET", path)
            locations = et_data.get("resource", {}).get("locations", [])
            self._cache.put(loc_key, locations)

        payload: dict[str, Any] = {
            "event_type": event_type_uri,
            "start_time": start_time,
            "invitee": {
                "name": name,
                "email": email,
                "timezone": timezone,
            },
        }

        # Include location if the event type has one configured
        if locations:
            loc = locations[0]
            payload["location"] = {
                "kind": loc["kind"],
                "location": loc.get("location", ""),
            }

        data = self._request("POST", "/invitees", json_body=payload)

        # ── Invalidate cache so subsequent reads are fresh ───────────
        self._invalidate_events_for_email(email)
        logger.info("Cache: invalidated events for %s after booking", email)

        return data["resource"]

    def cancel_event(
        self,
        event_uuid: str,
        reason: str = "Cancelled by patient via chat agent",
    ) -> dict[str, Any]:
        """Cancel a scheduled event.

        Args:
            event_uuid: The UUID of the scheduled event.
            reason: Cancellation reason.
        """
        result = self._request(
            "POST",
            f"/scheduled_events/{event_uuid}/cancellation",
            json_body={"reason": reason},
        )

        # ── Invalidate cache: reverse-lookup email from cached events ─
        self._invalidate_events_for_event_uuid(event_uuid)
        logger.info("Cache: invalidated events for event %s after cancel", event_uuid)

        return result

    def find_event_by_invitee_email(self, email: str) -> list[dict[str, Any]]:
        """Find active scheduled events where the given email is an invitee.

        Uses the Calendly ``invitee_email`` filter on GET /scheduled_events
        to retrieve matching events in a single API call.  Results are cached
        until invalidated by a write operation.

        Returns events with their invitee details attached.
        """
        cache_key = f"{_CK_EVENTS_BY_EMAIL}{email.lower()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        user_uri = self.get_current_user_uri()
        data = self._request(
            "GET",
            "/scheduled_events",
            params={
                "user": user_uri,
                "status": "active",
                "invitee_email": email,
            },
        )
        events = data.get("collection", [])

        # Fetch invitee details for each matching event (usually just 1-2)
        results: list[dict[str, Any]] = []
        for event in events:
            event_uuid = event["uri"].split("/")[-1]
            try:
                invitees = self.get_event_invitees(event_uuid)
                for invitee in invitees:
                    if invitee.get("email", "").lower() == email.lower():
                        results.append({**event, "invitee": invitee})
                        break
            except CalendlyAPIError:
                logger.warning(
                    "Could not fetch invitees for event %s", event_uuid,
                )
                # Still include the event without invitee details
                results.append(event)

        self._cache.put(cache_key, results)
        return results


# ── Module-level singleton (thread-safe) ────────────────────────────
_client: CalendlyClient | None = None
_client_lock = threading.Lock()


def get_calendly_client() -> CalendlyClient:
    """Return a module-level CalendlyClient singleton.

    Uses double-checked locking so that the lock is only acquired during
    the first initialisation, not on every subsequent call.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = CalendlyClient()
    return _client
