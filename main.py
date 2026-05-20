from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import uuid

load_dotenv()

import sys
import os
if not os.getenv("GEMINI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    print("ERROR: Missing GEMINI_API_KEY or TAVILY_API_KEY in .env — please set them before starting.")
    sys.exit(1)

from graph import build_graph

app = FastAPI()
compiled_graph = build_graph()

# In-memory session store
sessions: dict[str, dict] = {}

app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ── Fix 1: Optional[str] instead of str = None ────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")

@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    # ── Fix 2: Always hold real BaseMessage objects in sessions ───────────────
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "query": "",
            "pending_query": None,
            "resolved_query": None,
            "search_query": None,
            "awaiting_clarification": False,
            "clarity_status": None,
            "clarification_question": None,
            "research_findings": None,
            "confidence_score": None,
            "validation_result": None,
            "attempt_count": 0,
            "final_response": None
        }

    state = sessions[session_id]

    # Append as proper HumanMessage object
    state["messages"].append(HumanMessage(content=req.message))
    state["attempt_count"] = 0
    state["clarity_status"] = None
    state["clarification_question"] = None
    state["final_response"] = None
    state["research_findings"] = None
    state["confidence_score"] = None
    state["validation_result"] = None
    state["search_query"] = None
    state["resolved_query"] = None

    if state.get("awaiting_clarification") and state.get("pending_query"):
        combined = f"{state['pending_query']}\nUser clarification: {req.message}".strip()
        state["query"] = combined
        state["resolved_query"] = combined
        state["pending_query"] = None
        state["awaiting_clarification"] = False
    else:
        state["query"] = req.message

    # Run the graph
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = compiled_graph.invoke(state, config=config)
    except Exception as e:
        return {
            "session_id": session_id,
            "response": f"Pipeline error: {str(e)}",
            "status": "error"
        }

    # Save updated state back — merge result so no field is accidentally wiped
    sessions[session_id].update(result)

    # Clarity agent flagged the query as ambiguous
    if result.get("clarity_status") == "needs_clarification":
        question = (
            result.get("clarification_question")
            or "Could you clarify? Which company are you asking about?"
        )
        sessions[session_id]["pending_query"] = state.get("query")
        sessions[session_id]["awaiting_clarification"] = True
        sessions[session_id]["messages"].append(AIMessage(content=question))
        return {
            "session_id": session_id,
            "response": question,
            "status": "needs_clarification"
        }

    # Normal path — synthesis produced a response
    final = result.get("final_response") or "Sorry, I couldn't find relevant information."
    sessions[session_id]["messages"].append(AIMessage(content=final))

    return {
        "session_id": session_id,
        "response": final,
        "status": "complete",
        "confidence_score": result.get("confidence_score"),
        "attempts": result.get("attempt_count")
    }