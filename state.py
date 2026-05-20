from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage

class ResearchState(TypedDict):
    messages: List[BaseMessage]        # full conversation history
    query: str                         # current user query
    pending_query: Optional[str]       # waiting for clarification
    resolved_query: Optional[str]      # combined query after clarification
    search_query: Optional[str]        # rewritten query used for search
    awaiting_clarification: bool       # gate for follow-up clarification
    clarity_status: Optional[str]      # "clear" | "needs_clarification"
    clarification_question: Optional[str]  # what to ask the user
    research_findings: Optional[str]   # raw research output
    confidence_score: Optional[int]    # 0-10
    validation_result: Optional[str]   # "sufficient" | "insufficient"
    attempt_count: int                 # retry counter for validator loop
    final_response: Optional[str]      # synthesis output