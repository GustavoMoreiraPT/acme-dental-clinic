"""Centralized configuration for the Acme Dental AI Agent.

Secret resolution order (per variable):
  1. Environment variable / ``.env`` file  (local dev)
  2. AWS SSM Parameter Store SecureString  (when ``AWS_EXECUTION_ENV`` is set)

The SSM paths follow the convention ``/acme-dental/<VARIABLE_NAME>``.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Feature flag: running on AWS? ────────────────────────────────────
_ON_AWS = bool(os.getenv("AWS_EXECUTION_ENV"))


# ── Secret resolution ────────────────────────────────────────────────

def _get_ssm_parameter(name: str) -> str | None:
    """Fetch a SecureString from SSM Parameter Store.

    Returns ``None`` if the parameter does not exist or boto3 is
    unavailable.  Errors are logged but never raised so that local-dev
    fallback still works.
    """
    try:
        import boto3  # noqa: PLC0415 — lazy import to avoid boto3 dep in tests

        ssm = boto3.client("ssm")
        resp = ssm.get_parameter(Name=f"/acme-dental/{name}", WithDecryption=True)
        return resp["Parameter"]["Value"]
    except Exception:
        logger.debug("SSM lookup for %s failed (expected locally)", name)
        return None


def _require_env(name: str) -> str:
    """Return a config value from env-var or SSM, or raise a clear error."""
    # 1. Env var / .env (always checked first — allows local override)
    value = os.getenv(name)
    if value and not value.startswith("your_"):
        return value

    # 2. SSM Parameter Store (only on AWS)
    if _ON_AWS:
        ssm_value = _get_ssm_parameter(name)
        if ssm_value:
            return ssm_value

    raise OSError(
        f"Missing required configuration: {name}. "
        f"Set it in .env (local) or SSM Parameter Store /acme-dental/{name} (AWS)."
    )


# ── LLM ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = _require_env("ANTHROPIC_API_KEY")
MODEL_NAME: str = os.getenv("MODEL_NAME", "claude-opus-4-6")

# Multi-model routing: cheap models for classification and simple tasks
ROUTER_MODEL_NAME: str = os.getenv("ROUTER_MODEL_NAME", "claude-haiku-4-5")
FAST_MODEL_NAME: str = os.getenv("FAST_MODEL_NAME", "claude-haiku-4-5")

# ── Calendly ────────────────────────────────────────────────────────
CALENDLY_API_TOKEN: str = _require_env("CALENDLY_API_TOKEN")
CALENDLY_BASE_URL: str = "https://api.calendly.com"

# ── Server ──────────────────────────────────────────────────────────
SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
).split(",")
