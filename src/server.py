"""FastAPI server for the Acme Dental AI Agent.

Run with:
    uv run uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.agent import create_acme_dental_agent
from src.api.routes import router
from src.config import CORS_ORIGINS, SERVER_HOST, SERVER_PORT

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: initialise / tear-down shared resources ────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Start-up: compile the LangGraph agent once and store it in app state.

    Using the lifespan context manager (instead of a module-level global)
    is the FastAPI-recommended way to manage expensive resources.  It is
    thread-safe and makes the init point explicit.
    """
    logger.info("Compiling LangGraph agent…")
    application.state.agent = create_acme_dental_agent()
    logger.info("Agent ready.")
    yield
    # Shutdown: nothing to clean up for now (MemorySaver is in-memory)


# ── FastAPI application ──────────────────────────────────────────────
app = FastAPI(
    title="Acme Dental AI Agent",
    description=(
        "AI-powered dental receptionist — book, reschedule, "
        "cancel appointments and answer FAQs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (needed for React frontend) ────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request-ID middleware ────────────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next) -> Response:
    """Attach a unique request ID to every request for log correlation.

    The ID is added to the response headers (``X-Request-ID``) so the
    client can reference it in support tickets, and injected into the
    logging context so every log line for this request is traceable.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    logger.info(
        "[%s] %s %s", request_id, request.method, request.url.path,
    )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Register routes ──────────────────────────────────────────────────
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Acme Dental AI Agent",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Acme Dental API server on %s:%d", SERVER_HOST, SERVER_PORT)
    uvicorn.run(
        "src.server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,
    )
