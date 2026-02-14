"""Shared test fixtures for the Acme Dental test suite."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest


def pytest_configure(config):
    """Set test environment variables BEFORE collection starts.

    This runs before any imports, so config.py won't fail on module load.
    """
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key-123")
    os.environ.setdefault("CALENDLY_API_TOKEN", "test-calendly-token-456")


@pytest.fixture
def mock_calendly_response():
    """Factory fixture for creating mock Calendly API responses."""

    def _make(data: dict, status_code: int = 200):
        mock = MagicMock()
        mock.status_code = status_code
        mock.json.return_value = data
        mock.text = str(data)
        return mock

    return _make
