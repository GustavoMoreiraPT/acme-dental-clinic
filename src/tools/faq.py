"""FAQ knowledge base tool.

Loads the KNOWLEDGE_BASE.md file and provides a tool for the LLM to search
it when answering patient questions about the clinic.

Strategy: We embed the full FAQ in the system prompt (it's small enough) and
also provide a search_faq tool for explicit lookups. The tool splits the FAQ
into section chunks and does keyword matching to find the most relevant sections.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── Load the knowledge base at module import time ────────────────────

_KB_PATH = Path(__file__).resolve().parent.parent.parent / "KNOWLEDGE_BASE.md"


def _load_knowledge_base() -> str:
    """Read the full KNOWLEDGE_BASE.md file."""
    try:
        return _KB_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("KNOWLEDGE_BASE.md not found at %s", _KB_PATH)
        return ""


def _split_into_sections(content: str) -> list[dict[str, str]]:
    """Split the markdown FAQ into individual Q&A sections.

    Returns a list of dicts like:
      {"heading": "What services do you offer?", "body": "Acme Dental currently..."}
    """
    sections: list[dict[str, str]] = []
    # Split on ### headings (the Q&A questions)
    pattern = r"###\s+(.+?)(?=\n)"
    parts = re.split(pattern, content)

    # parts[0] is the preamble, then alternating heading/body pairs
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # Clean up body: remove trailing ---
        body = re.sub(r"\n---\s*$", "", body).strip()
        sections.append({"heading": heading, "body": body})

    return sections


# Pre-load at import time
_FULL_FAQ: str = _load_knowledge_base()
_FAQ_SECTIONS: list[dict[str, str]] = _split_into_sections(_FULL_FAQ)


def get_full_faq() -> str:
    """Return the complete FAQ content (used for system prompt injection)."""
    return _FULL_FAQ


@tool
def search_faq(query: str) -> str:
    """Search the Acme Dental FAQ knowledge base for answers to patient questions.

    Use this tool when a patient asks a question about the clinic's services,
    pricing, policies, booking process, what to bring, or any general inquiry.

    Args:
        query: The patient's question or keywords to search for.
    """
    if not _FAQ_SECTIONS:
        return "The FAQ knowledge base is currently unavailable. Please provide general guidance."

    query_lower = query.lower()
    query_words = set(query_lower.split())

    # Score each section by keyword overlap
    scored: list[tuple[float, dict[str, str]]] = []
    for section in _FAQ_SECTIONS:
        text = f"{section['heading']} {section['body']}".lower()

        # Count matching words (excluding very short ones)
        matches = sum(1 for w in query_words if len(w) > 2 and w in text)
        # Bonus for exact phrase match in heading
        heading_lower = section["heading"].lower()
        if any(w in heading_lower for w in query_words if len(w) > 3):
            matches += 2

        if matches > 0:
            scored.append((matches, section))

    if not scored:
        return (
            "I couldn't find a specific FAQ entry for that question. "
            "Here's a summary: Acme Dental offers 30-minute dental check-ups "
            "for €60 (€50 for students and seniors 65+). We have one dentist, "
            "accept no walk-ins, and appointments must be booked in advance. "
            "You can ask me about services, pricing, cancellation policies, "
            "what to bring, or how to book."
        )

    # Return the top 3 most relevant sections
    scored.sort(key=lambda x: x[0], reverse=True)
    top_sections = scored[:3]

    lines = ["Here's what I found in our FAQ:\n"]
    for _, section in top_sections:
        lines.append(f"**{section['heading']}**")
        lines.append(section["body"])
        lines.append("")

    return "\n".join(lines)
