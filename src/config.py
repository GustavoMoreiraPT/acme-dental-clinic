"""Centralized configuration for the Acme Dental AI Agent."""

import os

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    """Return an environment variable or raise a clear error if missing."""
    value = os.getenv(name)
    if not value or value.startswith("your_"):
        raise OSError(
            f"Missing required environment variable: {name}. "
            f"Please set it in your .env file."
        )
    return value


# ── LLM ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = _require_env("ANTHROPIC_API_KEY")
MODEL_NAME: str = os.getenv("MODEL_NAME", "claude-opus-4-6")

# ── Calendly ────────────────────────────────────────────────────────
CALENDLY_API_TOKEN: str = _require_env("CALENDLY_API_TOKEN")
CALENDLY_BASE_URL: str = "https://api.calendly.com"

# ── Server ──────────────────────────────────────────────────────────
SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
