from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import ResearchState
from agents import clarity_agent, research_agent, validator_agent, synthesis_agent

# ── Routing functions ─────────────────────────────────────────────────────────
def route_after_clarity(state: ResearchState) -> str:
    if state.get("clarity_status") == "needs_clarification":
        return "needs_clarification"
    return "research"

def route_after_research(state: ResearchState) -> str:
    score = state.get("confidence_score", 0)
    if score >= 6:
        return "synthesis"
    return "validator"

def route_after_validator(state: ResearchState) -> str:
    if state.get("validation_result") == "sufficient":
        return "synthesis"
    if state.get("attempt_count", 0) >= 3:
        return "synthesis"   # max retries reached, synthesize anyway
    return "research"

# ── Build graph ───────────────────────────────────────────────────────────────
def build_graph():
    memory = MemorySaver()   # persists state across turns (multi-turn support)
    graph = StateGraph(ResearchState)

    graph.add_node("clarity", clarity_agent)
    graph.add_node("research", research_agent)
    graph.add_node("validator", validator_agent)
    graph.add_node("synthesis", synthesis_agent)

    graph.set_entry_point("clarity")

    graph.add_conditional_edges(
        "clarity",
        route_after_clarity,
        {
            "needs_clarification": END,   # FastAPI handles the interrupt
            "research": "research"
        }
    )
    graph.add_conditional_edges(
        "research",
        route_after_research,
        {
            "validator": "validator",
            "synthesis": "synthesis"
        }
    )
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "research": "research",
            "synthesis": "synthesis"
        }
    )
    graph.add_edge("synthesis", END)

    return graph.compile(checkpointer=memory)