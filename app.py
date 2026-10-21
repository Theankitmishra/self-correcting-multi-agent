"""
================================================================================
FastAPI Server & Web Interface for Self-Correcting Multi-Agent System
Author: Ankit Kumar Mishra (M.Tech CS, DTU)
================================================================================
"""

import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request
import uvicorn

from backend import LLM_PROVIDER, MAX_REVISIONS, multi_agent_graph, run_workflow

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Self-Correcting Multi-Agent AI System",
    description="Cyclic Agentic Workflow built with LangGraph, Agent-as-a-Judge Evaluation, and FastAPI.",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and templates
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
templates_dir = BASE_DIR / "templates"
templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


class QueryRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=500, description="Topic or question for multi-agent synthesis")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "provider": LLM_PROVIDER,
            "max_revisions": MAX_REVISIONS,
        },
    )


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "provider": LLM_PROVIDER,
        "max_revisions": MAX_REVISIONS,
        "framework": "LangGraph StateGraph",
    }


@app.post("/api/run")
def run_agent_job(payload: QueryRequest):
    """
    Synchronous execution endpoint returning full trace and final answer.
    """
    try:
        result = run_workflow(payload.topic)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stream")
async def stream_agent_execution(topic: str):
    """
    Server-Sent Events (SSE) streaming endpoint that yields real-time agent updates.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        initial_state = {
            "topic": topic,
            "draft": "",
            "feedback": "",
            "decision": "",
            "score": 0.0,
            "revision_count": 0,
            "review_history": [],
        }

        yield f"data: {json.dumps({'event': 'started', 'topic': topic})}\n\n"

        for update in multi_agent_graph.stream(initial_state, stream_mode="updates"):
            for node_name, values in update.items():
                payload = {
                    "event": "node_update",
                    "agent": node_name,
                    "draft": values.get("draft", ""),
                    "decision": values.get("decision", ""),
                    "score": values.get("score", 0.0),
                    "feedback": values.get("feedback", ""),
                    "revision_count": values.get("revision_count", 0),
                }
                yield f"data: {json.dumps(payload)}\n\n"

        yield f"data: {json.dumps({'event': 'completed'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
