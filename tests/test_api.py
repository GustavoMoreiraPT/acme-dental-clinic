"""Tests for the FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked agent."""
    # We need to mock the agent before importing the app
    with patch("src.api.routes._get_agent") as mock_get_agent:
        mock_agent = MagicMock()
        mock_get_agent.return_value = mock_agent

        # Configure the mock to return a realistic response
        mock_message = MagicMock()
        mock_message.content = "Hello! I'm Linda, your dental receptionist. How can I help you?"
        mock_agent.invoke.return_value = {"messages": [mock_message]}

        from src.server import app

        yield TestClient(app), mock_agent


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        test_client, _ = client
        response = test_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "acme-dental-agent"


class TestChatEndpoint:
    def test_chat_returns_response(self, client):
        test_client, mock_agent = client
        response = test_client.post(
            "/api/chat",
            json={"message": "Hello!", "session_id": "test-session-1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert data["session_id"] == "test-session-1"
        assert "Linda" in data["reply"]

    def test_chat_passes_session_id(self, client):
        test_client, mock_agent = client
        test_client.post(
            "/api/chat",
            json={"message": "Hi!", "session_id": "my-unique-session"},
        )
        # Verify the agent was called with the correct thread_id
        call_args = mock_agent.invoke.call_args
        assert call_args[1]["config"]["configurable"]["thread_id"] == "my-unique-session"

    def test_chat_validates_empty_message(self, client):
        test_client, _ = client
        response = test_client.post(
            "/api/chat",
            json={"message": "", "session_id": "test-session"},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_chat_validates_missing_session(self, client):
        test_client, _ = client
        response = test_client.post(
            "/api/chat",
            json={"message": "Hello!"},
        )
        assert response.status_code == 422

    def test_chat_handles_agent_error(self, client):
        test_client, mock_agent = client
        mock_agent.invoke.side_effect = RuntimeError("LLM exploded")
        response = test_client.post(
            "/api/chat",
            json={"message": "Hello!", "session_id": "test-session"},
        )
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()


class TestRootEndpoint:
    def test_root_returns_service_info(self, client):
        test_client, _ = client
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Acme Dental AI Agent"
        assert "docs" in data
