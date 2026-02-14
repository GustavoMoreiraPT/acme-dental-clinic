"""FastAPI server for the Acme Dental AI Agent.

Run with:
    uv run uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import CORS_ORIGINS, SERVER_HOST, SERVER_PORT

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI application ──────────────────────────────────────────────
app = FastAPI(
    title="Acme Dental AI Agent",
    description="AI-powered dental receptionist — book, reschedule, cancel appointments and answer FAQs.",
    version="1.0.0",
)

# ── CORS (needed for React frontend) ────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
