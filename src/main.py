"""CLI entry point for the Acme Dental AI Agent.

This provides a simple terminal-based chat interface for testing and
development. For production, use the FastAPI server (src/server.py).

Usage:
    uv run python -m src.main            # normal mode (quiet)
    uv run python -m src.main --debug    # debug mode (shows API calls)
"""

from __future__ import annotations

import argparse
import logging
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.agent import create_acme_dental_agent

logger = logging.getLogger(__name__)


def _configure_logging(debug: bool = False) -> None:
    """Set up logging: WARNING by default, DEBUG when --debug is passed."""
    root_level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    if not debug:
        # Silence chatty HTTP loggers even if root is WARNING
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Always keep our own logger at INFO minimum so session starts show
    logging.getLogger("src").setLevel(logging.DEBUG if debug else logging.INFO)


def main():
    """Run the interactive CLI chat loop."""
    parser = argparse.ArgumentParser(description="Acme Dental AI Agent CLI")
    parser.add_argument(
        "--debug", action="store_true",
        help="Show all log messages including HTTP requests",
    )
    args = parser.parse_args()

    load_dotenv()
    _configure_logging(debug=args.debug)

    print("\n" + "=" * 60)
    print("  Acme Dental AI Agent - CLI Chat")
    print("=" * 60)
    print("  Type your message and press Enter.")
    print("  Commands: 'quit' to exit, 'new' for a new session.")
    print("=" * 60 + "\n")

    agent = create_acme_dental_agent()
    session_id = str(uuid.uuid4())
    logger.info("Started new session: %s", session_id)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("\nGoodbye! Have a great day!")
            break

        if user_input.lower() == "new":
            session_id = str(uuid.uuid4())
            print(f"\n>> New session started: {session_id[:8]}...\n")
            continue

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": session_id}},
            )

            messages = result.get("messages", [])
            if not messages:
                print("Linda: I'm sorry, I wasn't able to generate a response. Please try again.\n")
                continue

            last_message = messages[-1]
            reply = last_message.content if hasattr(last_message, "content") else str(last_message)
            print(f"\nLinda: {reply}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.exception("Error processing message")
            print(f"\nLinda: I'm sorry, something went wrong: {e}")
            print("     Please try again or type 'new' to start a fresh session.\n")


if __name__ == "__main__":
    main()
