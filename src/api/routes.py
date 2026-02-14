"""FastAPI route definitions for the Acme Dental agent API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from src.agent import create_acme_dental_agent
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Create the agent once at module load ─────────────────────────────
_agent = None


def _get_agent():
    """Lazy-init the compiled LangGraph agent."""
    global _agent
    if _agent is None:
        _agent = create_acme_dental_agent()
    return _agent


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the dental agent and get a response.

    The session_id is used to maintain conversation context across
    multiple requests from the same user.
    """
    try:
        agent = _get_agent()

        # Invoke the agent with the user's message and session context
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": request.session_id}},
        )

        # Extract the last AI message
        messages = result.get("messages", [])
        if not messages:
            raise HTTPException(status_code=500, detail="Agent produced no response")

        last_message = messages[-1]
        reply = last_message.content if hasattr(last_message, "content") else str(last_message)

        return ChatResponse(reply=reply, session_id=request.session_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing chat request")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}",
        ) from e
