"""Acme Dental AI Agent — an AI-powered dental clinic receptionist.

Architecture Overview
=====================

This agent is built using **LangGraph** as a state machine with two core nodes:

1. **chatbot** — Invokes Claude Opus 4.6 with the conversation history and a
   detailed system prompt that includes the clinic's FAQ. The LLM decides whether
   to respond directly or call a tool.

2. **tools** — Executes the tool calls requested by the LLM. Results are passed
   back to the chatbot node for interpretation.

Routing: chatbot → (tool calls?) → tools → chatbot (loop until no tool calls → END)

Key Design Decisions
--------------------
- **LLM**: Claude Opus 4.6 via Anthropic API — chosen for superior agentic
  reasoning and reliable tool usage over Sonnet 4.5.
- **Calendly Integration**: All scheduling goes through Calendly's REST API v2.
  Calendly handles email confirmations automatically.
- **FAQ Strategy**: The full KNOWLEDGE_BASE.md is injected into the system prompt
  (small enough at ~4KB) AND available via a `search_faq` tool for explicit lookups.
- **Resilience**: The CalendlyClient uses exponential backoff retries (3 attempts)
  for timeouts and 5xx errors, with clear error messages for the LLM.
- **Memory**: LangGraph's MemorySaver provides per-session conversation history,
  enabling multi-turn interactions.
- **Dual Interface**: FastAPI server (production) + CLI chat loop (development/testing).

Package Structure
-----------------
- ``src/agent.py`` — LangGraph StateGraph definition
- ``src/config.py`` — Centralized configuration from environment variables
- ``src/prompts.py`` — System prompt with FAQ injection
- ``src/server.py`` — FastAPI application
- ``src/main.py`` — CLI chat interface
- ``src/services/`` — External API clients (Calendly)
- ``src/tools/`` — LangChain tools (scheduling + FAQ)
- ``src/api/`` — FastAPI routes and Pydantic schemas
"""
