# Acme Dental AI Agent

An AI-powered dental clinic receptionist built with **LangGraph** and **Claude Opus 4.6**. The agent can book, reschedule, and cancel dental appointments via the **Calendly API**, and answer patient FAQs using a knowledge base.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    LangGraph Agent                      │
│                                                         │
│  ┌──────────┐    tool calls?    ┌────────────────────┐ │
│  │ chatbot  │──── YES ─────────►│   tools node       │ │
│  │ (Claude  │                   │ • get_available_    │ │
│  │  Opus    │◄── always ────────│   slots             │ │
│  │  4.6)    │                   │ • create_booking    │ │
│  │          │──── NO ──► END    │ • find_booking      │ │
│  └──────────┘                   │ • cancel_booking    │ │
│       │                         │ • reschedule_booking│ │
│  System Prompt                  │ • search_faq        │ │
│  + FAQ Knowledge Base           └────────────────────┘ │
└────────────────────────────────────────────────────────┘
         │                              │
    ┌────┴────┐                  ┌──────┴──────┐
    │ FastAPI │                  │ Calendly    │
    │ Server  │                  │ API v2      │
    │ :8000   │                  │ (with retry)│
    └─────────┘                  └─────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Claude Opus 4.6** | Superior agentic reasoning and tool-use reliability over Sonnet 4.5 |
| **LangGraph StateGraph** | Required by spec; provides clean state machine with conditional routing |
| **FAQ in system prompt** | Knowledge base is small (~4KB) — embedding it directly avoids unnecessary RAG complexity |
| **Exponential backoff retries** | Calendly API can be unreliable; 3 retries with backoff ensures resilience |
| **Per-session MemorySaver** | Enables multi-turn conversations across API calls without external DB |
| **Dual interface** | FastAPI for production/frontend, CLI for quick testing and development |
| **Calendly email handling** | Calendly sends confirmation/cancellation emails automatically — no custom email needed |
| **Direct booking via POST /invitees** | Uses Calendly's [Scheduling API](https://developer.calendly.com/schedule-events-with-ai-agents) to create bookings programmatically — no redirect to Calendly needed |

## Project Structure

```
src/
├── agent.py              # LangGraph StateGraph — chatbot + tools nodes
├── config.py             # Centralized environment configuration
├── prompts.py            # System prompt with persona + FAQ injection
├── server.py             # FastAPI application (production entry point)
├── main.py               # CLI chat interface (dev/testing)
├── services/
│   └── calendly_client.py  # Calendly API v2 client with retry logic
├── tools/
│   ├── calendly.py       # 5 LangChain tools for appointment management
│   └── faq.py            # FAQ knowledge base loader + search tool
└── api/
    ├── routes.py          # FastAPI endpoints (/chat, /health)
    └── schemas.py         # Pydantic request/response models
tests/
├── test_calendly_client.py  # CalendlyClient unit tests (retries, caching, etc.)
├── test_faq.py              # FAQ loading and search tests
└── test_api.py              # FastAPI endpoint tests
```

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Anthropic API key (for Claude Opus 4.6)
- Calendly API token (for booking functionality)

### Installation

1. **Install uv** (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Install dependencies**:

```bash
make install
```

3. **Set up environment variables**:

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
CALENDLY_API_TOKEN=your_calendly_api_token_here
```

### Running the Agent

**Option 1 — FastAPI Server** (recommended, supports frontend):

```bash
make serve
# Server starts at http://localhost:8000
# API docs at http://localhost:8000/docs
```

**Option 2 — CLI Chat** (quick testing):

```bash
make run
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/chat` | Send a message and get a response |

**POST `/api/chat`** example:

```json
{
  "message": "I'd like to book a dental check-up",
  "session_id": "user-session-123"
}
```

### Development Commands

```bash
make install    # Install dependencies (including dev)
make format     # Format code with ruff
make lint       # Lint code with ruff
make check      # Format and lint code
make run        # Run the agent (CLI)
make serve      # Start FastAPI server
make test       # Run tests
make help       # Show all available commands
```

### Running Tests

```bash
make test
```

All 30 tests run without external API calls — the Calendly client and LLM are fully mocked.

### Example Interaction

```
You: Hello, I'd like to book an appointment

Linda: Hi there! 😊 I'd be happy to help you book a dental check-up at Acme Dental.
     What dates work best for you? I can check our availability.

You: Next Monday would be great

Linda: Let me check what's available on Monday...
     Here are the available 30-minute slots:
       • Mon 17 Feb 2026 at 09:00
       • Mon 17 Feb 2026 at 10:30
       • Mon 17 Feb 2026 at 14:00
     Which time works best for you?

You: 10:30 please. My name is Jane Doe, email jane@example.com

Linda: Perfect! I've booked your appointment:
     📅 Monday 17 Feb 2026 at 10:30 (30 minutes)
     👤 Jane Doe
     📧 Confirmation sent to jane@example.com
     See you then!
```

## The Challenge

### Functional Requirements

1. **Create New Bookings** — Greet users, check availability, collect info, confirm
2. **Reschedule Bookings** — Find existing booking, cancel old, book new slot
3. **Cancel Bookings** — Find booking, confirm cancellation with policy reminder
4. **Answer FAQs** — Services, pricing, policies, what to bring, etc.

### Non-Functional Requirements

- ✅ Implemented with LangGraph
- ✅ LLM model choice documented (Claude Opus 4.6)
- ✅ Resilient API integration (retries, timeouts, error handling)
- ✅ Architectural decisions documented
