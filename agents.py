import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from state import ResearchState
from tools import get_search_tool

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.3
        )
    return _llm

def _strip_json_fence(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
    return raw.strip()

def _parse_json_or_none(text: str):
    try:
        raw = _strip_json_fence(text)
        return json.loads(raw)
    except Exception:
        return None

def _safe_int(value, default=0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+", value)
        if match:
            return int(match.group(0))
    return default

def _clamp_score(score: int) -> int:
    return max(0, min(10, score))

def _build_history(messages, exclude_last: bool = True) -> str:
    if not messages:
        return ""
    items = messages[:-1] if exclude_last else messages
    history = ""
    for msg in items:
        if isinstance(msg, HumanMessage):
            history += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history += f"Assistant: {msg.content}\n"
    return history.strip()

def _looks_like_company(query: str) -> bool:
    # Heuristic: treat acronyms or possessive proper nouns as company-like
    if re.search(r"\b[A-Z]{2,}(?:'s)?\b", query):
        return True
    if re.search(r"\b[A-Z][A-Za-z0-9&.-]+'s\b", query):
        return True
    if re.search(r"\b(?:Inc|Incorporated|Corp|Corporation|LLC|Ltd|PLC|Company|Co)\b", query, re.IGNORECASE):
        return True
    return False

def _rewrite_query_for_search(llm, query: str, history: str) -> dict:
    prompt = f"""You rewrite user queries for web search.

Conversation history (may include company names):
{history or "None"}

Current query: "{query}"

Rules:
- If the current query is a follow-up that references a prior company, include that company name explicitly
- Preserve the user's intent (financials, news, CEO, competitors, etc.)
- Output a concise search_query suitable for web search

Respond with ONLY valid JSON:
{{
  "resolved_query": "Full, explicit query",
  "search_query": "Search-friendly query"
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    result = _parse_json_or_none(response.content)
    if not result:
        return {"resolved_query": query, "search_query": query}

    resolved = (result.get("resolved_query") or "").strip() or query
    search_query = (result.get("search_query") or "").strip() or resolved
    return {"resolved_query": resolved, "search_query": search_query}

# ── 1. CLARITY AGENT ─────────────────────────────────────────────────────────
def clarity_agent(state: ResearchState) -> ResearchState:
    llm = get_llm()
    query = state["query"]
    history = _build_history(state.get("messages", []), exclude_last=True)

    prompt = f"""You are a query clarity evaluator. Analyze this query and decide if it is specific enough to research a company.

Conversation history:
{history or "None"}

Query: "{query}"

A query is CLEAR if:
- It mentions a specific company name
- The intent is understandable (financials, news, CEO, competitors, etc.)
- It refers to a company mentioned earlier in the conversation

A query NEEDS_CLARIFICATION if:
- No company name is mentioned
- It is too vague to research (e.g. "tell me about that company", "what about them")

Important:
- Do NOT validate whether the company exists in the real world
- Treat acronyms and placeholders (e.g., "XYZ") as valid company names

Respond with ONLY valid JSON, nothing else:
{{
  "clarity_status": "clear" or "needs_clarification",
  "clarification_question": "Your question to the user if unclear, else null"
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    result = _parse_json_or_none(response.content)
    if not result:
        result = {
            "clarity_status": "needs_clarification",
            "clarification_question": "Which company are you asking about?"
        }

    clarity_status = (result.get("clarity_status") or "needs_clarification").strip().lower()
    if clarity_status not in {"clear", "needs_clarification"}:
        clarity_status = "needs_clarification"

    if clarity_status == "needs_clarification" and _looks_like_company(query):
        clarity_status = "clear"
        result["clarification_question"] = None

    return {
        **state,
        "clarity_status": clarity_status,
        "clarification_question": result.get("clarification_question")
    }


# ── 2. RESEARCH AGENT ────────────────────────────────────────────────────────
def research_agent(state: ResearchState) -> ResearchState:
    llm = get_llm()
    search = get_search_tool()
    query = state["query"]

    # Build context from conversation history (exclude the current message — last in list)
    history = _build_history(state.get("messages", []), exclude_last=True)

    rewrite = _rewrite_query_for_search(llm, query, history)
    resolved_query = rewrite["resolved_query"]
    search_query = rewrite["search_query"]

    # Run Tavily search
    search_results = search.invoke(search_query)
    search_text = "\n\n".join(
        [f"Source: {r.get('url') or r.get('link', '')}\n{r.get('content', '')}" for r in search_results]
    ) if isinstance(search_results, list) else str(search_results)

    prompt = f"""You are a business research specialist. Use the search results below to answer the user's query.

Conversation history:
{history}

Current query: "{resolved_query}"

Search results:
{search_text}

Instructions:
- Extract relevant facts about the company (news, financials, leadership, recent events)
- Be factual and cite sources where possible
- Assign a confidence score (0-10) based on how complete and relevant the search results are
  - 8-10: Rich, directly relevant results
  - 5-7: Partial information, some gaps
  - 0-4: Little useful data found

Respond with ONLY valid JSON:
{{
  "findings": "Detailed research findings here...",
  "confidence_score": 7
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    result = _parse_json_or_none(response.content)
    if not result:
        result = {"findings": response.content, "confidence_score": 4}

    confidence = _clamp_score(_safe_int(result.get("confidence_score", 4), default=4))

    return {
        **state,
        "research_findings": result.get("findings", ""),
        "confidence_score": confidence,
        "attempt_count": state.get("attempt_count", 0) + 1,
        "resolved_query": resolved_query,
        "search_query": search_query
    }


# ── 3. VALIDATOR AGENT ───────────────────────────────────────────────────────
def validator_agent(state: ResearchState) -> ResearchState:
    llm = get_llm()

    history = _build_history(state.get("messages", []), exclude_last=True)
    current_query = state.get("resolved_query") or state["query"]

    prompt = f"""You are a research quality validator. Evaluate whether the research findings adequately answer the user's query.

Conversation history:
{history or "None"}

User query: "{current_query}"

Research findings:
{state.get('research_findings', 'No findings yet')}

Confidence score assigned by researcher: {state.get('confidence_score', 0)}/10

Is this sufficient to give the user a helpful, accurate answer?
- SUFFICIENT: Key facts are present, query is meaningfully answered
- INSUFFICIENT: Major gaps, off-topic results, or very thin data

Respond with ONLY valid JSON:
{{
  "validation_result": "sufficient" or "insufficient",
  "reason": "Brief reason"
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    result = _parse_json_or_none(response.content)
    if not result:
        result = {"validation_result": "insufficient", "reason": "parse error fallback"}

    validation_result = (result.get("validation_result") or "insufficient").strip().lower()
    if validation_result not in {"sufficient", "insufficient"}:
        validation_result = "insufficient"

    return {
        **state,
        "validation_result": validation_result
    }


# ── 4. SYNTHESIS AGENT ───────────────────────────────────────────────────────
def synthesis_agent(state: ResearchState) -> ResearchState:
    llm = get_llm()

    # Build conversation history for context (exclude current message — last in list)
    history = _build_history(state.get("messages", []), exclude_last=True)
    current_query = state.get("resolved_query") or state["query"]

    prompt = f"""You are a business intelligence analyst. Synthesize the research into a clear, well-structured response.

Conversation history (for context on follow-up questions):
{history}

Current query: "{current_query}"

Research findings:
{state.get('research_findings', 'No findings available')}

Instructions:
- Write a clear, structured summary for the user
- Use sections with headers where appropriate (## Header)
- If this is a follow-up question, acknowledge the prior context
- Keep it factual, concise, and useful
- End with 1-2 suggested follow-up questions the user might want to ask"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        **state,
        "final_response": response.content
    }