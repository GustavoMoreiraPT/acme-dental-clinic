"""Tests for the FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.server import app


@pytest.fixture
def mock_agent():
    """Create a mock agent and attach it to app state (mirrors the lifespan)."""
    agent = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Hello! I'm Linda, your dental receptionist. How can I help you?"
    agent.invoke.return_value = {"messages": [mock_message]}

    # Attach to app state the same way the lifespan does
    app.state.agent = agent
    yield agent
    # Clean up
    app.state.agent = None


@pytest.fixture
def client(mock_agent):
    """FastAPI test client with the mock agent wired up."""
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "acme-dental-agent"


class TestChatEndpoint:
    def test_chat_returns_response(self, client, mock_agent):
        response = client.post(
            "/api/chat",
            json={"message": "Hello!", "session_id": "test-session-1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert data["session_id"] == "test-session-1"
        assert "Linda" in data["reply"]

    def test_chat_passes_session_id(self, client, mock_agent):
        client.post(
            "/api/chat",
            json={"message": "Hi!", "session_id": "my-unique-session"},
        )
        call_args = mock_agent.invoke.call_args
        assert call_args[1]["config"]["configurable"]["thread_id"] == "my-unique-session"

    def test_chat_validates_empty_message(self, client):
        response = client.post(
            "/api/chat",
            json={"message": "", "session_id": "test-session"},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_chat_validates_missing_session(self, client):
        response = client.post(
            "/api/chat",
            json={"message": "Hello!"},
        )
        assert response.status_code == 422

    def test_chat_handles_agent_error(self, client, mock_agent):
        mock_agent.invoke.side_effect = RuntimeError("LLM exploded")
        response = client.post(
            "/api/chat",
            json={"message": "Hello!", "session_id": "test-session"},
        )
        assert response.status_code == 500
        # Verify we do NOT leak the internal error message to the client
        detail = response.json()["detail"]
        assert "LLM exploded" not in detail
        assert "internal error" in detail.lower()

    def test_response_includes_request_id_header(self, client):
        response = client.post(
            "/api/chat",
            json={"message": "Hello!", "session_id": "test-session"},
        )
        assert "X-Request-ID" in response.headers

    def test_client_supplied_request_id_is_echoed(self, client):
        response = client.post(
            "/api/chat",
            json={"message": "Hello!", "session_id": "test-session"},
            headers={"X-Request-ID": "my-trace-id-123"},
        )
        assert response.headers["X-Request-ID"] == "my-trace-id-123"


class TestAgentNotReady:
    def test_returns_503_when_agent_not_initialised(self):
        """If the agent hasn't been set via lifespan, return 503."""
        # Enter the test client (triggers lifespan), then wipe the agent
        # to simulate the state before lifespan completes.
        with TestClient(app) as tc:
            app.state.agent = None
            response = tc.post(
                "/api/chat",
                json={"message": "Hello!", "session_id": "s1"},
            )
            assert response.status_code == 503
            assert "starting up" in response.json()["detail"].lower()


class TestRootEndpoint:
    def test_root_returns_service_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Acme Dental AI Agent"
        assert "docs" in data
