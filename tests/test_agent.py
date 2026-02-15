"""Tests for the multi-model agent routing logic.

Covers:
  - Router classification (FAQ vs booking intent)
  - FAQ node behaviour (Haiku, no tools)
  - End-to-end graph routing with mocked LLMs
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import (
    AgentState,
    _build_router_context,
    _make_chatbot_node,
    _make_faq_node,
    _make_router_node,
    route_by_intent,
    should_use_tools,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_mock_llm(response_content: str, tool_calls: list | None = None):
    """Create a mock LLM that returns a fixed AIMessage."""
    mock_llm = MagicMock()
    ai_msg = AIMessage(content=response_content)
    if tool_calls:
        ai_msg.tool_calls = tool_calls
    mock_llm.invoke.return_value = ai_msg
    return mock_llm


# ── TestRouterClassification ─────────────────────────────────────────


class TestRouterClassification:
    """Verify the router node classifies messages correctly."""

    @patch("src.agent._build_router_llm")
    def test_faq_message_classified_as_faq(self, mock_build):
        mock_build.return_value = _make_mock_llm("FAQ")
        router = _make_router_node()

        state: AgentState = {
            "messages": [HumanMessage(content="What are your opening hours?")],
            "intent": "",
        }
        result = router(state)
        assert result["intent"] == "faq"

    @patch("src.agent._build_router_llm")
    def test_booking_message_classified_as_booking(self, mock_build):
        mock_build.return_value = _make_mock_llm("BOOKING")
        router = _make_router_node()

        state: AgentState = {
            "messages": [HumanMessage(content="I want to book an appointment")],
            "intent": "",
        }
        result = router(state)
        assert result["intent"] == "booking"

    @patch("src.agent._build_router_llm")
    def test_greeting_classified_as_faq(self, mock_build):
        mock_build.return_value = _make_mock_llm("FAQ")
        router = _make_router_node()

        state: AgentState = {
            "messages": [HumanMessage(content="Hello!")],
            "intent": "",
        }
        result = router(state)
        assert result["intent"] == "faq"

    @patch("src.agent._build_router_llm")
    def test_reschedule_classified_as_booking(self, mock_build):
        mock_build.return_value = _make_mock_llm("BOOKING")
        router = _make_router_node()

        state: AgentState = {
            "messages": [HumanMessage(content="I need to reschedule my appointment")],
            "intent": "",
        }
        result = router(state)
        assert result["intent"] == "booking"

    @patch("src.agent._build_router_llm")
    def test_unexpected_response_defaults_to_booking(self, mock_build):
        """If the router returns gibberish, default to the safer booking path."""
        mock_build.return_value = _make_mock_llm("SOMETHING_RANDOM")
        router = _make_router_node()

        state: AgentState = {
            "messages": [HumanMessage(content="What time is it?")],
            "intent": "",
        }
        result = router(state)
        assert result["intent"] == "booking"

    @patch("src.agent._build_router_llm")
    def test_router_handles_llm_error_gracefully(self, mock_build):
        """If the router LLM fails, default to booking (safe fallback)."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        mock_build.return_value = mock_llm
        router = _make_router_node()

        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "intent": "",
        }
        result = router(state)
        assert result["intent"] == "booking"

    @patch("src.agent._build_router_llm")
    def test_router_does_not_add_messages(self, mock_build):
        """The router should only set intent, not add messages."""
        mock_build.return_value = _make_mock_llm("FAQ")
        router = _make_router_node()

        state: AgentState = {
            "messages": [HumanMessage(content="What's the price?")],
            "intent": "",
        }
        result = router(state)
        assert "messages" not in result
        assert "intent" in result

    @patch("src.agent._build_router_llm")
    def test_providing_name_email_during_booking_classified_as_booking(self, mock_build):
        """When the AI asked for name/email during a booking, the reply must be BOOKING."""
        mock_build.return_value = _make_mock_llm("BOOKING")
        router = _make_router_node()

        # Simulate an active booking flow: AI asked for name and email
        state: AgentState = {
            "messages": [
                HumanMessage(content="I want to book for next Monday at 10am"),
                AIMessage(content="Great! To complete the booking, I'll need your full name and email address."),
                HumanMessage(content="gustavo is my name, and my email is gustavo@example.com"),
            ],
            "intent": "",
        }
        result = router(state)
        assert result["intent"] == "booking"


# ── TestBuildRouterContext ───────────────────────────────────────────


class TestBuildRouterContext:
    """Verify the context builder for the router prompt."""

    def test_empty_messages_returns_empty_string(self):
        assert _build_router_context([]) == ""

    def test_single_message_returns_empty_string(self):
        """A single human message has no prior context to show."""
        msgs = [HumanMessage(content="Hello")]
        assert _build_router_context(msgs) == ""

    def test_includes_prior_ai_message(self):
        msgs = [
            HumanMessage(content="I want to book"),
            AIMessage(content="I'd be happy to help! What date works?"),
            HumanMessage(content="Next Monday"),
        ]
        context = _build_router_context(msgs)
        assert "I want to book" in context
        assert "happy to help" in context
        # The latest human message should NOT be in context (it's passed separately)
        assert "Next Monday" not in context

    def test_includes_booking_flow_context(self):
        """The context must show when the AI asked for name/email."""
        msgs = [
            HumanMessage(content="Book me in for Monday 10am"),
            AIMessage(content="To complete the booking, I need your full name and email."),
            HumanMessage(content="John Smith, john@example.com"),
        ]
        context = _build_router_context(msgs)
        assert "full name" in context
        assert "email" in context

    def test_truncates_long_messages(self):
        long_text = "a" * 500
        msgs = [
            HumanMessage(content=long_text),
            AIMessage(content="OK"),
            HumanMessage(content="Thanks"),
        ]
        context = _build_router_context(msgs)
        # Messages are truncated to 200 chars
        assert len(context) < 500


# ── TestFaqNode ──────────────────────────────────────────────────────


class TestFaqNode:
    """Verify the FAQ chatbot node returns a response without tool calls."""

    @patch("src.agent._build_fast_llm")
    def test_faq_node_returns_ai_message(self, mock_build):
        mock_build.return_value = _make_mock_llm(
            "Our clinic is open Monday to Friday, 9am to 5pm."
        )
        faq_node = _make_faq_node()

        state: AgentState = {
            "messages": [HumanMessage(content="What are your opening hours?")],
            "intent": "faq",
        }
        result = faq_node(state)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "open" in result["messages"][0].content.lower()

    @patch("src.agent._build_fast_llm")
    def test_faq_node_never_produces_tool_calls(self, mock_build):
        """The FAQ node uses a model with no tool bindings."""
        mock_build.return_value = _make_mock_llm("Check-ups cost €60.")
        faq_node = _make_faq_node()

        state: AgentState = {
            "messages": [HumanMessage(content="How much is a check-up?")],
            "intent": "faq",
        }
        result = faq_node(state)
        ai_msg = result["messages"][0]
        assert not hasattr(ai_msg, "tool_calls") or not ai_msg.tool_calls

    @patch("src.agent._build_fast_llm")
    def test_faq_node_raises_on_llm_error(self, mock_build):
        """Unlike the router, the FAQ node should propagate errors."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        mock_build.return_value = mock_llm
        faq_node = _make_faq_node()

        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "intent": "faq",
        }
        with pytest.raises(RuntimeError, match="LLM down"):
            faq_node(state)


# ── TestChatbotNode ──────────────────────────────────────────────────


class TestChatbotNode:
    """Verify the Opus chatbot node works for booking workflows."""

    @patch("src.agent._build_llm")
    def test_chatbot_node_returns_ai_message(self, mock_build):
        mock_build.return_value = _make_mock_llm(
            "I'd be happy to help you book an appointment!"
        )
        chatbot_node = _make_chatbot_node()

        state: AgentState = {
            "messages": [HumanMessage(content="I want to book")],
            "intent": "booking",
        }
        result = chatbot_node(state)
        assert "messages" in result
        assert len(result["messages"]) == 1


# ── TestRouteByIntent ────────────────────────────────────────────────


class TestRouteByIntent:
    """Verify the conditional edge function routes correctly."""

    def test_faq_intent_routes_to_faq_chatbot(self):
        state: AgentState = {
            "messages": [HumanMessage(content="Hi")],
            "intent": "faq",
        }
        assert route_by_intent(state) == "faq_chatbot"

    def test_booking_intent_routes_to_chatbot(self):
        state: AgentState = {
            "messages": [HumanMessage(content="Book me in")],
            "intent": "booking",
        }
        assert route_by_intent(state) == "chatbot"

    def test_missing_intent_defaults_to_chatbot(self):
        """If intent is missing entirely, default to the full Opus path."""
        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "intent": "",
        }
        # Empty string is falsy but .get() returns it; "booking" is not "faq"
        assert route_by_intent(state) == "chatbot"

    def test_unknown_intent_routes_to_chatbot(self):
        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "intent": "something_else",
        }
        assert route_by_intent(state) == "chatbot"


# ── TestShouldUseTools ───────────────────────────────────────────────


class TestShouldUseTools:
    """Verify the tool-routing edge function."""

    def test_message_with_tool_calls_routes_to_tools(self):
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"name": "get_available_slots", "args": {}, "id": "1"}]
        state: AgentState = {
            "messages": [ai_msg],
            "intent": "booking",
        }
        assert should_use_tools(state) == "tools"

    def test_message_without_tool_calls_routes_to_end(self):
        ai_msg = AIMessage(content="Here are the available slots.")
        state: AgentState = {
            "messages": [ai_msg],
            "intent": "booking",
        }
        result = should_use_tools(state)
        assert result == "__end__"  # LangGraph's END sentinel

    def test_message_with_empty_tool_calls_routes_to_end(self):
        ai_msg = AIMessage(content="All done!")
        ai_msg.tool_calls = []
        state: AgentState = {
            "messages": [ai_msg],
            "intent": "booking",
        }
        result = should_use_tools(state)
        assert result == "__end__"
