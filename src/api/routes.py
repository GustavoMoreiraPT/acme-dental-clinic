"""FastAPI route definitions for the Acme Dental agent API."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage

from src.api.schemas import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_agent(request: Request):
    """Retrieve the compiled LangGraph agent from app state.

    The agent is initialised once during the FastAPI lifespan (see
    ``server.py``), which avoids the thread-safety issue of a bare
    module-level global with lazy init.
    """
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="The agent is still starting up. Please try again in a moment.",
        )
    return agent


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """Send a message to the dental agent and get a response.

    The session_id is used to maintain conversation context across
    multiple requests from the same user.

    **Implementation note**: ``agent.invoke()`` is a synchronous blocking
    call (it talks to the Anthropic API).  We offload it to a thread via
    ``asyncio.to_thread`` so that the main event loop is never blocked
    and other requests (health checks, concurrent users) are served
    normally.
    """
    agent = _get_agent(http_request)
    request_id = getattr(http_request.state, "request_id", "?")

    try:
        # Offload the blocking LLM call to the default thread-pool so
        # the async event loop stays responsive.
        result = await asyncio.to_thread(
            agent.invoke,
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.session_id}},
        )

        # Extract the last AI message
        messages = result.get("messages", [])
        if not messages:
            logger.error("[%s] Agent returned no messages", request_id)
            raise HTTPException(status_code=500, detail="Agent produced no response.")

        last_message = messages[-1]
        reply = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        return ChatResponse(reply=reply, session_id=request.session_id)

    except HTTPException:
        raise
    except Exception as e:
        # Log the full traceback server-side, but do NOT leak it to the
        # client — a senior-engineer habit that prevents information
        # disclosure in production.
        logger.exception("[%s] Error processing chat request", request_id)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        ) from e
