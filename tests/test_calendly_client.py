"""Tests for the CalendlyClient service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.services.cache import LRUCache
from src.services.calendly_client import (
    INITIAL_BACKOFF_SECONDS,
    MAX_RETRIES,
    CalendlyAPIError,
    CalendlyClient,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = data
    mock.text = str(data)
    return mock


# ── Tests: get_current_user_uri ──────────────────────────────────────


class TestGetCurrentUserUri:
    def test_returns_user_uri(self):
        client = CalendlyClient(token="test-token")
        user_data = {"resource": {"uri": "https://api.calendly.com/users/TESTUSER123"}}

        with patch.object(client._client, "request", return_value=_mock_response(user_data)):
            uri = client.get_current_user_uri()
            assert uri == "https://api.calendly.com/users/TESTUSER123"

    def test_caches_user_uri(self):
        client = CalendlyClient(token="test-token")
        user_data = {"resource": {"uri": "https://api.calendly.com/users/TESTUSER123"}}

        with patch.object(client._client, "request", return_value=_mock_response(user_data)) as mock_req:
            client.get_current_user_uri()
            client.get_current_user_uri()
            # Should only call the API once
            assert mock_req.call_count == 1


# ── Tests: get_available_times ───────────────────────────────────────


class TestGetAvailableTimes:
    def test_returns_available_slots(self):
        client = CalendlyClient(token="test-token")
        client._user_uri = "https://api.calendly.com/users/TEST"

        slots_data = {
            "collection": [
                {"start_time": "2026-02-17T10:00:00Z", "status": "available"},
                {"start_time": "2026-02-17T10:30:00Z", "status": "available"},
                {"start_time": "2026-02-17T11:00:00Z", "status": "unavailable"},
            ]
        }

        with patch.object(client._client, "request", return_value=_mock_response(slots_data)):
            slots = client.get_available_times(
                "https://api.calendly.com/event_types/TEST",
                "2026-02-17T00:00:00Z",
                "2026-02-17T23:59:59Z",
            )
            assert len(slots) == 3
            assert slots[0]["status"] == "available"

    def test_returns_empty_list_when_no_slots(self):
        client = CalendlyClient(token="test-token")
        client._user_uri = "https://api.calendly.com/users/TEST"

        with patch.object(
            client._client,
            "request",
            return_value=_mock_response({"collection": []}),
        ):
            slots = client.get_available_times(
                "https://api.calendly.com/event_types/TEST",
                "2026-02-17T00:00:00Z",
                "2026-02-17T23:59:59Z",
            )
            assert slots == []


# ── Tests: retry logic ───────────────────────────────────────────────


class TestRetryLogic:
    @patch("src.services.calendly_client.time.sleep")
    def test_retries_on_timeout(self, mock_sleep):
        client = CalendlyClient(token="test-token")

        success_response = _mock_response(
            {"resource": {"uri": "https://api.calendly.com/users/TEST"}}
        )

        with patch.object(
            client._client,
            "request",
            side_effect=[
                httpx.TimeoutException("timeout"),
                success_response,
            ],
        ):
            uri = client.get_current_user_uri()
            assert uri == "https://api.calendly.com/users/TEST"
            mock_sleep.assert_called_once_with(INITIAL_BACKOFF_SECONDS)

    @patch("src.services.calendly_client.time.sleep")
    def test_retries_on_500_error(self, mock_sleep):
        client = CalendlyClient(token="test-token")

        error_response = _mock_response({"error": "Internal Server Error"}, 500)
        success_response = _mock_response(
            {"resource": {"uri": "https://api.calendly.com/users/TEST"}}
        )

        with patch.object(
            client._client,
            "request",
            side_effect=[error_response, success_response],
        ):
            uri = client.get_current_user_uri()
            assert uri == "https://api.calendly.com/users/TEST"

    @patch("src.services.calendly_client.time.sleep")
    def test_does_not_retry_on_400_error(self, mock_sleep):
        client = CalendlyClient(token="test-token")

        error_response = _mock_response({"error": "Bad Request"}, 400)

        with patch.object(client._client, "request", return_value=error_response):
            with pytest.raises(CalendlyAPIError) as exc_info:
                client.get_current_user_uri()
            assert "400" in str(exc_info.value)
            mock_sleep.assert_not_called()

    @patch("src.services.calendly_client.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        client = CalendlyClient(token="test-token")

        with patch.object(
            client._client,
            "request",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            with pytest.raises(CalendlyAPIError) as exc_info:
                client.get_current_user_uri()
            assert "after" in str(exc_info.value).lower()
            assert mock_sleep.call_count == MAX_RETRIES


# ── Tests: cancel_event ──────────────────────────────────────────────


class TestCancelEvent:
    def test_cancel_sends_reason(self):
        client = CalendlyClient(token="test-token")
        cancel_response = {"resource": {"canceled": True}}

        with patch.object(
            client._client,
            "request",
            return_value=_mock_response(cancel_response),
        ) as mock_req:
            result = client.cancel_event("event-uuid-123", "Patient request")
            assert result == cancel_response
            call_args = mock_req.call_args
            assert call_args[1]["json"] == {"reason": "Patient request"}


# ── Tests: find_event_by_invitee_email ───────────────────────────────


class TestFindEventByInviteeEmail:
    def test_finds_matching_event(self):
        client = CalendlyClient(token="test-token")
        client._user_uri = "https://api.calendly.com/users/TEST"

        events_data = {
            "collection": [
                {
                    "uri": "https://api.calendly.com/scheduled_events/EVT1",
                    "start_time": "2026-02-20T10:00:00Z",
                    "end_time": "2026-02-20T10:30:00Z",
                    "status": "active",
                },
            ]
        }
        invitees_data = {
            "collection": [
                {"email": "patient@example.com", "name": "Jane Doe"},
            ]
        }

        with patch.object(
            client._client,
            "request",
            side_effect=[
                _mock_response(events_data),
                _mock_response(invitees_data),
            ],
        ):
            results = client.find_event_by_invitee_email("patient@example.com")
            assert len(results) == 1
            assert results[0]["invitee"]["name"] == "Jane Doe"

    def test_returns_empty_when_no_match(self):
        client = CalendlyClient(token="test-token")
        client._user_uri = "https://api.calendly.com/users/TEST"

        events_data = {
            "collection": [
                {
                    "uri": "https://api.calendly.com/scheduled_events/EVT1",
                    "start_time": "2026-02-20T10:00:00Z",
                    "end_time": "2026-02-20T10:30:00Z",
                    "status": "active",
                },
            ]
        }
        invitees_data = {
            "collection": [
                {"email": "other@example.com", "name": "Other Person"},
            ]
        }

        with patch.object(
            client._client,
            "request",
            side_effect=[
                _mock_response(events_data),
                _mock_response(invitees_data),
            ],
        ):
            results = client.find_event_by_invitee_email("patient@example.com")
            assert results == []


# ── Tests: create_invitee (direct booking) ──────────────────────────


class TestCreateInvitee:
    def test_creates_booking_with_location(self):
        """POST /invitees should send the correct payload and return invitee data."""
        client = CalendlyClient(token="test-token")
        event_type_uri = "https://api.calendly.com/event_types/ET123"

        event_type_response = _mock_response({
            "resource": {
                "locations": [{"kind": "physical", "location": "Acme Dental Lane"}],
            }
        })
        invitee_response = _mock_response({
            "resource": {
                "name": "Jane Doe",
                "email": "jane@example.com",
                "cancel_url": "https://calendly.com/cancellations/INV123",
                "reschedule_url": "https://calendly.com/reschedulings/INV123",
                "event": "https://api.calendly.com/scheduled_events/EVT456",
                "status": "active",
            }
        })

        with patch.object(
            client._client,
            "request",
            side_effect=[event_type_response, invitee_response],
        ) as mock_req:
            result = client.create_invitee(
                event_type_uri,
                "2026-02-17T09:00:00Z",
                name="Jane Doe",
                email="jane@example.com",
            )

            assert result["name"] == "Jane Doe"
            assert result["cancel_url"] == "https://calendly.com/cancellations/INV123"
            assert result["status"] == "active"

            # Check the POST payload
            post_call = mock_req.call_args_list[1]
            payload = post_call[1]["json"]
            assert payload["event_type"] == event_type_uri
            assert payload["start_time"] == "2026-02-17T09:00:00Z"
            assert payload["invitee"]["name"] == "Jane Doe"
            assert payload["invitee"]["email"] == "jane@example.com"
            assert payload["location"]["kind"] == "physical"
            assert payload["location"]["location"] == "Acme Dental Lane"

    def test_omits_location_when_not_configured(self):
        """When event type has no locations, do not send a location object."""
        client = CalendlyClient(token="test-token")
        event_type_uri = "https://api.calendly.com/event_types/ET123"

        event_type_response = _mock_response({
            "resource": {"locations": []}
        })
        invitee_response = _mock_response({
            "resource": {
                "name": "Jane Doe",
                "email": "jane@example.com",
                "cancel_url": "https://calendly.com/cancellations/INV123",
                "reschedule_url": "https://calendly.com/reschedulings/INV123",
                "event": "https://api.calendly.com/scheduled_events/EVT456",
                "status": "active",
            }
        })

        with patch.object(
            client._client,
            "request",
            side_effect=[event_type_response, invitee_response],
        ) as mock_req:
            client.create_invitee(
                event_type_uri,
                "2026-02-17T09:00:00Z",
                name="Jane Doe",
                email="jane@example.com",
            )

            # POST payload should NOT include a location key
            post_call = mock_req.call_args_list[1]
            payload = post_call[1]["json"]
            assert "location" not in payload


# ── Tests: LRU cache integration ────────────────────────────────────


class TestCacheIntegration:
    """Verify that CalendlyClient reads from cache and invalidates on writes."""

    def _make_client(self) -> tuple[CalendlyClient, LRUCache]:
        cache = LRUCache()
        client = CalendlyClient(token="test-token", cache=cache)
        client._user_uri = "https://api.calendly.com/users/TEST"
        return client, cache

    # ── get_event_types caching ──────────────────────────────────────

    def test_get_event_types_caches_result(self):
        client, cache = self._make_client()
        et_data = {"collection": [{"uri": "https://api.calendly.com/event_types/ET1"}]}

        with patch.object(
            client._client, "request", return_value=_mock_response(et_data),
        ) as mock_req:
            first = client.get_event_types()
            second = client.get_event_types()
            assert first == second
            # Only one API call — second read served from cache
            assert mock_req.call_count == 1

    # ── find_event_by_invitee_email caching ──────────────────────────

    def test_find_events_caches_result(self):
        client, cache = self._make_client()
        events_resp = _mock_response({
            "collection": [{
                "uri": "https://api.calendly.com/scheduled_events/EVT1",
                "start_time": "2026-02-20T10:00:00Z",
                "end_time": "2026-02-20T10:30:00Z",
                "status": "active",
            }]
        })
        invitees_resp = _mock_response({
            "collection": [{"email": "alice@test.com", "name": "Alice"}]
        })

        with patch.object(
            client._client, "request",
            side_effect=[events_resp, invitees_resp],
        ) as mock_req:
            first = client.find_event_by_invitee_email("alice@test.com")
            second = client.find_event_by_invitee_email("alice@test.com")
            assert first == second
            assert len(first) == 1
            # 2 API calls for the first read (events + invitees), 0 for second
            assert mock_req.call_count == 2

    # ── get_event_invitees caching ───────────────────────────────────

    def test_get_event_invitees_caches_result(self):
        client, cache = self._make_client()
        inv_data = {"collection": [{"email": "a@b.com", "name": "A"}]}

        with patch.object(
            client._client, "request", return_value=_mock_response(inv_data),
        ) as mock_req:
            first = client.get_event_invitees("EVT1")
            second = client.get_event_invitees("EVT1")
            assert first == second
            assert mock_req.call_count == 1

    # ── create_invitee invalidates email cache ───────────────────────

    def test_create_invitee_invalidates_email_cache(self):
        client, cache = self._make_client()

        # Pre-populate the cache with a find result
        cache.put("events_by_email:alice@test.com", [{"old": "data"}])
        assert cache.has("events_by_email:alice@test.com")

        et_response = _mock_response({
            "resource": {"locations": [{"kind": "physical", "location": "Lane"}]}
        })
        inv_response = _mock_response({
            "resource": {
                "name": "Alice",
                "email": "alice@test.com",
                "cancel_url": "http://c",
                "reschedule_url": "http://r",
                "event": "https://api.calendly.com/scheduled_events/NEW",
                "status": "active",
            }
        })

        with patch.object(
            client._client, "request",
            side_effect=[et_response, inv_response],
        ):
            client.create_invitee(
                "https://api.calendly.com/event_types/ET1",
                "2026-02-17T09:00:00Z",
                name="Alice",
                email="alice@test.com",
            )

        # Cache entry for this email must be gone
        assert not cache.has("events_by_email:alice@test.com")

    # ── cancel_event invalidates email cache via reverse lookup ──────

    def test_cancel_event_invalidates_email_cache(self):
        client, cache = self._make_client()

        # Simulate a cached find result that includes EVT1
        cache.put(
            "events_by_email:bob@test.com",
            [{"uri": "https://api.calendly.com/scheduled_events/EVT1"}],
        )
        assert cache.has("events_by_email:bob@test.com")

        cancel_resp = _mock_response({"resource": {"canceled": True}})

        with patch.object(
            client._client, "request", return_value=cancel_resp,
        ):
            client.cancel_event("EVT1", "No longer needed")

        # The reverse-lookup should have found and invalidated bob's cache
        assert not cache.has("events_by_email:bob@test.com")

    # ── cancel_event is safe when nothing cached ─────────────────────

    def test_cancel_event_no_crash_when_cache_empty(self):
        client, cache = self._make_client()
        cancel_resp = _mock_response({"resource": {"canceled": True}})

        with patch.object(
            client._client, "request", return_value=cancel_resp,
        ):
            # Should not raise even though no email is cached
            result = client.cancel_event("EVT999")
            assert result["resource"]["canceled"] is True

    # ── location config is cached across bookings ────────────────────

    def test_event_type_location_cached_across_bookings(self):
        client, cache = self._make_client()
        et_uri = "https://api.calendly.com/event_types/ET1"

        et_response = _mock_response({
            "resource": {"locations": [{"kind": "physical", "location": "Lane"}]}
        })
        inv_response = _mock_response({
            "resource": {
                "name": "A", "email": "a@b.com",
                "cancel_url": "c", "reschedule_url": "r",
                "event": "e", "status": "active",
            }
        })

        with patch.object(
            client._client, "request",
            side_effect=[et_response, inv_response, inv_response],
        ) as mock_req:
            # First booking: fetches event type + POST
            client.create_invitee(et_uri, "2026-02-17T09:00:00Z", name="A", email="a@b.com")
            # Second booking (different email): should skip event type GET
            client.create_invitee(et_uri, "2026-02-17T10:00:00Z", name="B", email="b@c.com")
            # 3 calls total: GET event_type, POST invitee, POST invitee
            assert mock_req.call_count == 3
