"""Tests for email validation in the Calendly tools."""

from __future__ import annotations

import pytest

from src.tools.calendly import _validate_email


class TestValidateEmail:
    """Unit tests for the _validate_email helper."""

    @pytest.mark.parametrize(
        "email",
        [
            "alice@example.com",
            "bob.jones@clinic.co.uk",
            "jane+tag@gmail.com",
            "user@sub.domain.org",
            "UPPER@CASE.COM",
            "digits123@test456.io",
        ],
    )
    def test_accepts_valid_emails(self, email: str):
        assert _validate_email(email) is None

    @pytest.mark.parametrize(
        "email",
        [
            "",
            "   ",
            "not-an-email",
            "missing@",
            "@no-local.com",
            "spaces in@email.com",
            "double@@at.com",
            "no-tld@localhost",
            "user@.leading-dot.com",
        ],
    )
    def test_rejects_invalid_emails(self, email: str):
        result = _validate_email(email)
        assert result is not None
        assert "does not look like a valid email" in result or "No email" in result
