"""Pydantic schemas for the FastAPI endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message from the frontend."""

    message: str = Field(..., min_length=1, max_length=2000, description="The user's message")
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique session identifier for conversation continuity",
    )


class ChatResponse(BaseModel):
    """Response from the agent."""

    reply: str = Field(..., description="The agent's response message")
    session_id: str = Field(..., description="The session ID for this conversation")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    service: str = "acme-dental-agent"
