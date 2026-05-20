# Multi-Agent Business Research Assistant

A production-grade multi-agent research assistant built with **LangGraph** and **FastAPI**, featuring a custom high-fidelity frontend UI. It orchestrates four specialized AI agents to collect, validate, and synthesize real-time business intelligence — supporting multi-turn conversations, follow-up questions, and Human-in-the-Loop clarification interrupts.

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```env
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

- **GEMINI_API_KEY** — Obtain from [Google AI Studio](https://aistudio.google.com/)
- **TAVILY_API_KEY** — Obtain from [Tavily](https://tavily.com/)

### 3. Run the Server

```bash
uvicorn main:app --reload
```

Then open your browser at **http://localhost:8000**

---

## Agent Architecture

All agents operate on a shared `ResearchState` TypedDict that carries conversation history, query fields, scores, and flags across every node in the graph.

```
User Query
    │
    ▼
┌─────────────┐
│ Clarity     │ ── needs_clarification ──► INTERRUPT (ask user) ──► loop back
│ Agent       │
└──────┬──────┘
       │ clear
       ▼
┌─────────────┐
│ Research    │ ── confidence ≥ 6 ───────────────────────────────► Synthesis
│ Agent       │
└──────┬──────┘
       │ confidence < 6
       ▼
┌─────────────┐
│ Validator   │ ── sufficient OR attempts ≥ 3 ──────────────────► Synthesis
│ Agent       │
└──────┬──────┘
       │ insufficient + attempts < 3
       └──────────────────────────────────────────────────────► Research (retry)
                                                                      │
                                                                      ▼
                                                              ┌───────────────┐
                                                              │   Synthesis   │
                                                              │   Agent       │
                                                              └──────┬────────┘
                                                                     │
                                                                    END
```

### Agent Roles

| Agent | Role | Key Output |
|---|---|---|
| **Clarity Agent** | Determines if the query names a specific company and is researchable | `clarity_status`: `clear` / `needs_clarification` |
| **Research Agent** | Rewrites the query for search, runs Tavily, and extracts business findings | `research_findings`, `confidence_score` (0–10) |
| **Validator Agent** | Assesses whether findings are complete enough to answer the query | `validation_result`: `sufficient` / `insufficient` |
| **Synthesis Agent** | Writes a structured markdown response with suggested follow-up questions | `final_response` |

### Multi-Turn Conversation

Session state is stored server-side in `sessions` (keyed by `session_id`). Every request appends to the `messages` list as LangChain `HumanMessage` / `AIMessage` objects. All agents receive this full history, enabling coherent follow-up queries like *"What about their competitors?"* or *"Tell me more about the CEO"* without restating the company name.

### Human-in-the-Loop (HITL) Interrupt

When the Clarity Agent returns `needs_clarification`, the graph routes to `END` early. FastAPI catches this, stores the original query as `pending_query`, sets `awaiting_clarification = True`, and returns the clarification question to the user. On the next request, the backend merges the original query with the user's clarification before re-invoking the graph — completing the interrupt-resume cycle without any LangGraph native interrupt primitives.