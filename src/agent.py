"""LangGraph-based AI Agent for the Acme Dental Clinic.

Architecture:
  The agent uses a **multi-model routing** pattern built as a LangGraph
  StateGraph with four nodes:

    1. **router**       — cheap Haiku call that classifies each message
                          as ``faq`` or ``booking``
    2. **faq_chatbot**  — Haiku LLM (no tools) handles FAQ / simple queries
    3. **chatbot**      — Opus LLM with tool bindings for booking workflows
    4. **tools**        — executes any tool calls the LLM requests

  Routing:
    router → (faq?)     → faq_chatbot → END
    router → (booking?) → chatbot → (has tool calls?) → tools → chatbot (loop)
                                   → (no tool calls?)  → END

  This tiered approach sends ~70-80% of messages (FAQs, greetings) through
  Haiku (~$0.80/M input) instead of Opus (~$15/M input), cutting cost by
  ~75% at scale with no quality loss for the complex booking flows.

  Memory:
    Conversation state is managed per session via LangGraph's MemorySaver
    checkpoint, enabling multi-turn conversations across API calls.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.config import ANTHROPIC_API_KEY, FAST_MODEL_NAME, MODEL_NAME, ROUTER_MODEL_NAME
from src.prompts import get_system_prompt
from src.services.metrics import metrics
from src.tools.calendly import (
    cancel_booking,
    create_booking,
    find_booking,
    get_available_slots,
    reschedule_booking,
)
from src.tools.faq import search_faq

logger = logging.getLogger(__name__)


# ── Router classification prompt ─────────────────────────────────────

ROUTER_PROMPT = (
    "Classify this dental clinic conversation. "
    "Reply with exactly one word — either FAQ or BOOKING.\n\n"
    "- FAQ: greetings, questions about services, pricing, location, "
    "hours, policies, insurance, general info, thank-you messages, "
    "or anything that does NOT require looking up or modifying "
    "appointments.\n"
    "- BOOKING: appointment booking, rescheduling, cancelling, "
    "checking availability, providing personal details (name, email, "
    "preferences) as part of an ongoing booking flow, or any request "
    "that requires looking up or modifying appointments.\n\n"
    "IMPORTANT: If the recent conversation shows an active "
    "booking/rescheduling/cancelling flow (e.g. the assistant asked "
    "for the patient's name, email, or time preference), then the "
    "patient's reply is ALWAYS part of that booking flow — classify "
    "it as BOOKING.\n\n"
    "{context}Latest message: {message}\n\n"
    "Classification:"
)


# ── State schema ─────────────────────────────────────────────────────


class AgentState(TypedDict):
    """The state that flows through the graph.

    ``messages`` uses the LangGraph ``add_messages`` reducer so that each
    node can append messages without overwriting the full history.

    ``intent`` is set by the router node and read by the conditional edge
    to decide which downstream path to take.  It is internal plumbing and
    never shown to the user.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    intent: str


# ── All tools the agent can use ──────────────────────────────────────

ALL_TOOLS = [
    get_available_slots,
    create_booking,
    find_booking,
    cancel_booking,
    reschedule_booking,
    search_faq,
]


# ── LLM builders ────────────────────────────────────────────────────


def _build_router_llm() -> ChatAnthropic:
    """Build a lightweight Haiku LLM for intent classification (no tools)."""
    return ChatAnthropic(
        model=ROUTER_MODEL_NAME,
        api_key=ANTHROPIC_API_KEY,
        temperature=0.0,  # Deterministic classification
        max_tokens=10,    # Only need one word back
    )


def _build_fast_llm() -> ChatAnthropic:
    """Build a Haiku LLM for FAQ / simple responses (no tools)."""
    return ChatAnthropic(
        model=FAST_MODEL_NAME,
        api_key=ANTHROPIC_API_KEY,
        temperature=0.1,
        max_tokens=1024,
    )


def _build_llm() -> ChatAnthropic:
    """Build the primary Opus LLM with tool bindings for booking workflows."""
    llm = ChatAnthropic(
        model=MODEL_NAME,
        api_key=ANTHROPIC_API_KEY,
        temperature=0.1,  # Low temperature for consistent, factual responses
        max_tokens=1024,
    )
    return llm.bind_tools(ALL_TOOLS)


# ── Node: router (Haiku — classifies intent) ────────────────────────


def _build_router_context(messages: list[AnyMessage], max_turns: int = 3) -> str:
    """Build a short context summary from recent messages for the router.

    Includes the last ``max_turns`` exchanges so the router can detect
    ongoing booking flows (e.g. the assistant just asked for name/email).
    """
    # Grab the last few messages (excluding the very last human message,
    # which is passed separately as "Latest message")
    recent = messages[-(max_turns * 2 + 1):-1] if len(messages) > 1 else []
    if not recent:
        return ""

    lines = ["Recent conversation:\n"]
    for msg in recent:
        if isinstance(msg, HumanMessage):
            lines.append(f"  Patient: {msg.content[:200]}")
        elif hasattr(msg, "content") and msg.content:
            lines.append(f"  Assistant: {msg.content[:200]}")
    lines.append("")
    return "\n".join(lines)


def _make_router_node():
    """Create the router node that classifies each user message.

    Uses Haiku to determine whether the message is a FAQ/simple query
    or a booking-related request.  The router receives recent conversation
    context so it can detect ongoing booking flows (e.g. the assistant
    just asked for name/email and the patient is replying with details).

    Writes the result to ``state["intent"]`` without polluting the
    conversation ``messages``.
    """
    router_llm = _build_router_llm()

    def router_node(state: AgentState) -> dict:
        """Classify the latest user message as 'faq' or 'booking'."""
        # Find the last human message to classify
        last_human_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg.content
                break

        context = _build_router_context(state["messages"])
        prompt = ROUTER_PROMPT.format(context=context, message=last_human_msg)
        t0 = time.perf_counter()
        try:
            response = router_llm.invoke([HumanMessage(content=prompt)])
            elapsed = (time.perf_counter() - t0) * 1000
            metrics.record_success("anthropic", "router_classify", latency_ms=elapsed)

            classification = response.content.strip().upper()
            intent = "faq" if "FAQ" in classification else "booking"
            logger.debug(
                "Router (%s) classified as: %s (raw: %r, %.0fms)",
                ROUTER_MODEL_NAME, intent, classification, elapsed,
            )
            return {"intent": intent}

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            metrics.record_failure(
                "anthropic", "router_classify",
                error_type=type(exc).__name__, latency_ms=elapsed,
            )
            # Fallback: route to the full Opus path (safe default)
            logger.warning("Router failed, defaulting to booking path: %s", exc)
            return {"intent": "booking"}

    return router_node


# ── Node: faq_chatbot (Haiku — no tools) ────────────────────────────


def _make_faq_node():
    """Create the FAQ chatbot node using Haiku (no tools).

    Handles simple questions, greetings, and FAQ lookups using the
    knowledge base embedded in the system prompt.  Much cheaper than
    routing these through Opus.
    """
    faq_llm = _build_fast_llm()

    def faq_node(state: AgentState) -> dict:
        """Answer a FAQ / simple query using Haiku."""
        logger.debug("faq_chatbot node invoked — model: %s", FAST_MODEL_NAME)
        system = SystemMessage(content=get_system_prompt())
        t0 = time.perf_counter()
        try:
            response = faq_llm.invoke([system] + state["messages"])
            elapsed = (time.perf_counter() - t0) * 1000
            metrics.record_success("anthropic", "faq_invoke", latency_ms=elapsed)
            logger.debug("faq_chatbot responded in %.0fms", elapsed)
            return {"messages": [response]}
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            metrics.record_failure(
                "anthropic", "faq_invoke",
                error_type=type(exc).__name__, latency_ms=elapsed,
            )
            raise

    return faq_node


# ── Node: chatbot (Opus — with tools) ───────────────────────────────


def _make_chatbot_node():
    """Create the chatbot node with the Opus LLM for booking workflows.

    The LLM + tool bindings are captured in the closure so that repeated
    node invocations (chatbot -> tool -> chatbot -> ...) share one client
    instead of re-constructing it on every loop iteration.
    """
    llm_with_tools = _build_llm()

    def chatbot_node(state: AgentState) -> dict:
        """Invoke the Opus LLM with the current conversation history."""
        logger.debug("chatbot node invoked — model: %s (with tools)", MODEL_NAME)
        system = SystemMessage(content=get_system_prompt())
        t0 = time.perf_counter()
        try:
            response = llm_with_tools.invoke([system] + state["messages"])
            elapsed = (time.perf_counter() - t0) * 1000
            metrics.record_success("anthropic", "llm_invoke", latency_ms=elapsed)
            logger.debug("chatbot responded in %.0fms", elapsed)
            return {"messages": [response]}
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            metrics.record_failure(
                "anthropic", "llm_invoke",
                error_type=type(exc).__name__, latency_ms=elapsed,
            )
            raise

    return chatbot_node


# ── Conditional edges ────────────────────────────────────────────────


def route_by_intent(state: AgentState) -> str:
    """Route to faq_chatbot or chatbot based on the router's classification."""
    intent = state.get("intent", "booking")
    if intent == "faq":
        return "faq_chatbot"
    return "chatbot"


def should_use_tools(state: AgentState) -> str:
    """Check if the last message has tool calls; if so, route to tools node."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ── Graph assembly ───────────────────────────────────────────────────


def create_acme_dental_agent() -> StateGraph:
    """Build and compile the Acme Dental LangGraph agent.

    The graph implements multi-model routing:
      - **router** (Haiku) classifies each message as FAQ or booking
      - **faq_chatbot** (Haiku) answers simple queries cheaply
      - **chatbot** (Opus) handles booking workflows with full tool access

    Returns a compiled graph that can be invoked with:
        graph.invoke(
            {"messages": [HumanMessage(content="...")]},
            config={"configurable": {"thread_id": "session-123"}},
        )
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", _make_router_node())
    graph.add_node("faq_chatbot", _make_faq_node())
    graph.add_node("chatbot", _make_chatbot_node())
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    # Entry point: always classify first
    graph.set_entry_point("router")

    # Router decides which path to take
    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {"faq_chatbot": "faq_chatbot", "chatbot": "chatbot"},
    )

    # FAQ path goes straight to END (no tools)
    graph.add_edge("faq_chatbot", END)

    # Booking path: chatbot may call tools, then loops back
    graph.add_conditional_edges(
        "chatbot", should_use_tools, {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "chatbot")

    # Compile with memory checkpointer for multi-turn conversations
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    logger.debug(
        "Acme Dental agent compiled — router: %s, faq: %s, booking: %s, tools: %d",
        ROUTER_MODEL_NAME, FAST_MODEL_NAME, MODEL_NAME, len(ALL_TOOLS),
    )
    return compiled
