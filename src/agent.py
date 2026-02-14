"""LangGraph-based AI Agent for the Acme Dental Clinic.

Architecture:
  The agent is built as a LangGraph StateGraph with two nodes:
    1. **chatbot** — calls the LLM with the conversation history + system prompt
    2. **tools**   — executes any tool calls the LLM requests

  Routing:
    chatbot → (has tool calls?) → tools → chatbot   (loop)
    chatbot → (no tool calls?)  → END               (respond to user)

  Memory:
    Conversation state is managed per session via LangGraph's MemorySaver
    checkpoint, enabling multi-turn conversations across API calls.
"""

from __future__ import annotations

import logging
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.config import ANTHROPIC_API_KEY, MODEL_NAME
from src.prompts import get_system_prompt
from src.tools.calendly import (
    cancel_booking,
    create_booking,
    find_booking,
    get_available_slots,
    reschedule_booking,
)
from src.tools.faq import search_faq

logger = logging.getLogger(__name__)


# ── State schema ─────────────────────────────────────────────────────


class AgentState(TypedDict):
    """The state that flows through the graph.

    `messages` uses the LangGraph `add_messages` reducer so that each
    node can append messages without overwriting the full history.
    """

    messages: Annotated[list[AnyMessage], add_messages]


# ── All tools the agent can use ──────────────────────────────────────

ALL_TOOLS = [
    get_available_slots,
    create_booking,
    find_booking,
    cancel_booking,
    reschedule_booking,
    search_faq,
]


# ── Node: chatbot ────────────────────────────────────────────────────


def _build_llm() -> ChatAnthropic:
    """Construct the Claude LLM with tool bindings.

    Called once at graph-compilation time and captured in the closure of
    ``chatbot_node`` so we don't reconstruct the client on every invocation.
    """
    llm = ChatAnthropic(
        model=MODEL_NAME,
        api_key=ANTHROPIC_API_KEY,
        temperature=0.1,  # Low temperature for consistent, factual responses
        max_tokens=1024,
    )
    return llm.bind_tools(ALL_TOOLS)


def _make_chatbot_node():
    """Create the chatbot node with the LLM built once and reused.

    The LLM + tool bindings are captured in the closure so that repeated
    node invocations (chatbot → tool → chatbot → …) share one client
    instead of re-constructing it on every loop iteration.
    """
    llm_with_tools = _build_llm()

    def chatbot_node(state: AgentState) -> dict:
        """Invoke the LLM with the current conversation history."""
        system = SystemMessage(content=get_system_prompt())
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    return chatbot_node


# ── Conditional edge: route to tools or end ──────────────────────────


def should_use_tools(state: AgentState) -> str:
    """Check if the last message has tool calls; if so, route to tools node."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ── Graph assembly ───────────────────────────────────────────────────


def create_acme_dental_agent() -> StateGraph:
    """Build and compile the Acme Dental LangGraph agent.

    Returns a compiled graph that can be invoked with:
        graph.invoke(
            {"messages": [HumanMessage(content="...")]},
            config={"configurable": {"thread_id": "session-123"}},
        )
    """
    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes — the chatbot node captures a single LLM instance via closure
    graph.add_node("chatbot", _make_chatbot_node())
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    # Set entry point
    graph.set_entry_point("chatbot")

    # Add conditional routing from chatbot
    graph.add_conditional_edges("chatbot", should_use_tools, {"tools": "tools", END: END})

    # After tools execute, always go back to chatbot to interpret results
    graph.add_edge("tools", "chatbot")

    # Compile with memory checkpointer for multi-turn conversations
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    logger.info("Acme Dental agent compiled with %d tools", len(ALL_TOOLS))
    return compiled
