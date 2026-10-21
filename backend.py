"""
================================================================================
Self-Correcting Multi-Agent System (LangGraph + FastAPI)
Author: Ankit Kumar Mishra (M.Tech CS, DTU)
Repository: https://github.com/Theankitmishra/self-correcting-multi-agent
================================================================================
A cyclic multi-agent architecture featuring:
  1. Writer Agent: Generates initial structured content drafts.
  2. Reviewer Agent (Agent-as-a-Judge): Evaluates clarity, analogies, accuracy, and rules.
  3. Conditional Router: Decides whether to PASS or trigger autonomous self-correction.
  4. Reviser Agent: Dynamically refines drafts based on precise critique.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, TypedDict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HumanMessage = None
SystemMessage = None
ChatOpenAI = None
StateGraph = None
START = "__start__"
END = "__end__"

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        @classmethod
        def model_validate(cls, data):
            return cls(**data)
    def Field(*args, **kwargs):
        return kwargs.get("default", None)
    class ValidationError(Exception):
        pass

# -----------------------------------------------------------------------------
# Configuration & Provider Setup
# -----------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# DigitalOcean Serverless Configuration
DO_INFERENCE_BASE_URL = os.getenv(
    "DO_INFERENCE_BASE_URL", "https://inference.do-ai.run/v1"
).rstrip("/")
MODEL_ACCESS_KEY = os.getenv("MODEL_ACCESS_KEY", "")
DO_MODEL = os.getenv("DO_MODEL", "kimi-k3")
DO_ROUTER = os.getenv("DO_INFERENCE_ROUTER", "").strip()

# OpenAI & Groq Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class MockLLM:
    """Deterministic offline fallback LLM for local testing and CI/CD."""
    def __init__(self, role: str):
        self.role = role

    def invoke(self, messages):
        user_prompt = ""
        for m in messages:
            content = getattr(m, "content", "")
            if isinstance(content, str):
                user_prompt += " " + content

        class MockResponse:
            def __init__(self, content):
                self.content = content

        if self.role == "writer":
            return MockResponse(
                "An AI agent is an autonomous software entity capable of perceiving its environment, "
                "reasoning, and taking actions to achieve specific goals. Think of it like an airplane's "
                "autopilot: it reads sensor telemetry (perception), computes pitch/yaw adjustments (reasoning), "
                "and trims control surfaces (action) without continuous human input. In production engineering, "
                "agents automate multi-step workflows with minimal friction."
            )
        elif self.role == "reviewer":
            if "Example Code Snippet" in user_prompt or "Perception-Action Loop" in user_prompt or "state =" in user_prompt:
                return MockResponse(
                    json.dumps({
                        "decision": "PASS",
                        "score": 0.96,
                        "feedback": "",
                        "passed_rules": ["Beginner Accessibility", "Real-World Analogy", "Concrete Example", "Topic Precision"],
                        "failed_rules": []
                    })
                )
            else:
                return MockResponse(
                    json.dumps({
                        "decision": "REVISE",
                        "score": 0.64,
                        "feedback": "Include a concrete Python LangGraph code snippet or structured perception-action loop to anchor the explanation.",
                        "passed_rules": ["Beginner Accessibility", "Real-World Analogy", "Topic Precision"],
                        "failed_rules": ["Concrete Example"]
                    })
                )
        elif self.role == "reviser":
            return MockResponse(
                "An AI agent is an autonomous software entity that perceives its environment and takes actions toward a defined objective.\n\n"
                "Analogy: Like an airplane's autopilot that constantly reads sensor telemetry and trims control surfaces without manual stick input.\n\n"
                "Example Code Snippet:\n"
                "```python\n"
                "# Minimal Perception-Action Loop in LangGraph\n"
                "state = {'goal': 'resolve_incident', 'status': 'investigating'}\n"
                "while state['status'] != 'resolved':\n"
                "    action = agent.decide(state)\n"
                "    state = tool_runner.execute(action)\n"
                "```\n\n"
                "By chaining perception with deterministic tool-calling, agents solve complex distributed problems reliably."
            )
        return MockResponse("Execution completed successfully.")


def get_llm(role: str = "default"):
    """
    Factory to instantiate LLM client based on configured provider.
    Supports OpenAI, DigitalOcean Serverless Inference, Groq, and Mock Demo Mode.
    """
    if LLM_PROVIDER == "digitalocean":
        if ChatOpenAI is None or not MODEL_ACCESS_KEY or MODEL_ACCESS_KEY.startswith("your_"):
            return MockLLM(role)
        model_name = f"router:{DO_ROUTER}" if DO_ROUTER else DO_MODEL
        return ChatOpenAI(
            model=model_name,
            base_url=DO_INFERENCE_BASE_URL,
            api_key=MODEL_ACCESS_KEY,
            temperature=TEMPERATURE,
            max_retries=2,
            timeout=90,
        )

    elif LLM_PROVIDER == "groq":
        try:
            from langchain_groq import ChatGroq
            if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
                return MockLLM(role)
            return ChatGroq(
                model_name=GROQ_MODEL,
                groq_api_key=GROQ_API_KEY,
                temperature=TEMPERATURE,
            )
        except ImportError:
            return MockLLM(role)

    else:
        # Default to OpenAI
        if ChatOpenAI is None or not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_") or OPENAI_API_KEY == "dummy-key":
            return MockLLM(role)
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_retries=2,
        )


# -----------------------------------------------------------------------------
# Structured Pydantic Evaluation Schema
# -----------------------------------------------------------------------------
class Review(BaseModel):
    decision: Literal["PASS", "REVISE"] = Field(
        description="PASS only if the draft satisfies all quality criteria; otherwise REVISE."
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quality assessment score between 0.0 and 1.0."
    )
    feedback: str = Field(
        default="",
        description="Actionable, pinpoint feedback explaining exact flaws when decision is REVISE."
    )
    passed_rules: List[str] = Field(
        default_factory=list,
        description="List of verification rules that passed."
    )
    failed_rules: List[str] = Field(
        default_factory=list,
        description="List of verification rules that failed."
    )


# -----------------------------------------------------------------------------
# LangGraph Stateful Schema
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    topic: str
    draft: str
    feedback: str
    decision: str
    score: float
    revision_count: int
    review_history: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# Helper Utilities
# -----------------------------------------------------------------------------
def _content_to_text(content: Any) -> str:
    """Normalize message response content to string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content)


def _parse_reviewer_json(raw_text: str) -> Review:
    """Robustly extract and validate JSON from reviewer LLM output."""
    clean = raw_text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)

    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if not match:
            # Fallback heuristic if formatting broke
            if "pass" in clean.lower() and "revise" not in clean.lower():
                return Review(decision="PASS", score=0.9, feedback="")
            return Review(
                decision="REVISE",
                score=0.5,
                feedback="Clarify the explanation, include an everyday analogy, and provide a concrete example."
            )
        payload = json.loads(match.group(0))

    try:
        review = Review.model_validate(payload)
    except ValidationError:
        decision = "PASS" if str(payload.get("decision", "")).upper() == "PASS" else "REVISE"
        score = float(payload.get("score", 0.7 if decision == "PASS" else 0.4))
        feedback = str(payload.get("feedback", ""))
        review = Review(decision=decision, score=score, feedback=feedback)

    if review.decision == "PASS":
        review.feedback = ""
    return review


# -----------------------------------------------------------------------------
# Node Implementations
# -----------------------------------------------------------------------------
def _msg(role: str, content: str):
    if SystemMessage is not None and HumanMessage is not None:
        return SystemMessage(content=content) if role == "system" else HumanMessage(content=content)
    class SimpleMsg:
        def __init__(self, c):
            self.content = c
    return SimpleMsg(content)


def writer_node(state: AgentState) -> Dict[str, Any]:
    """
    Writer Agent: Generates initial technical draft.
    Instructed to be beginner-accessible, include an everyday analogy, and a concrete example.
    """
    llm = get_llm("writer")
    system_prompt = (
        "You are the WRITER agent in a self-correcting multi-agent system.\n"
        "Your role: Explain the topic clearly for beginners in 120-170 words.\n"
        "Requirements:\n"
        "1. Start with an intuitive one-sentence definition.\n"
        "2. Include one vivid everyday real-world analogy.\n"
        "3. Include one tiny, practical code snippet or concrete real-world example.\n"
        "4. Conclude with why this concept matters in modern production engineering."
    )

    response = llm.invoke([
        _msg("system", system_prompt),
        _msg("user", f"Explain the following concept thoroughly: {state['topic']}")
    ])

    draft_text = _content_to_text(response.content)
    return {
        "draft": draft_text,
        "feedback": "",
        "decision": "PENDING_REVIEW",
        "score": 0.0,
        "revision_count": 0,
        "review_history": [],
    }


def reviewer_node(state: AgentState) -> Dict[str, Any]:
    """
    Reviewer Agent (Agent-as-a-Judge):
    Rigorously tests the draft against 4 objective verification criteria.
    """
    llm = get_llm("reviewer")
    system_prompt = (
        "You are the REVIEWER agent (Agent-as-a-Judge) in a self-correcting multi-agent system.\n"
        "Carefully evaluate the draft against these 4 strict criteria:\n"
        "1. Beginner Accessibility: Is it clear, intuitive, and jargon-free?\n"
        "2. Real-World Analogy: Does it contain an explicit everyday analogy?\n"
        "3. Concrete Example: Does it contain a concrete example or code snippet?\n"
        "4. Topic Precision: Does it stay strictly focused on the requested topic without hallucination?\n\n"
        "If ALL 4 pass and score >= 0.85, set decision='PASS' and feedback=''.\n"
        "Otherwise set decision='REVISE' and provide 1-2 specific, actionable revision steps.\n\n"
        "Return ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "decision": "PASS" | "REVISE",\n'
        '  "score": 0.0 - 1.0,\n'
        '  "feedback": "...",\n'
        '  "passed_rules": ["rule 1", ...],\n'
        '  "failed_rules": ["rule 2", ...]\n'
        "}"
    )

    user_prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Candidate Draft:\n{state['draft']}"
    )

    response = llm.invoke([
        _msg("system", system_prompt),
        _msg("user", user_prompt)
    ])

    review = _parse_reviewer_json(_content_to_text(response.content))

    history_item = {
        "revision": state.get("revision_count", 0),
        "decision": review.decision,
        "score": review.score,
        "feedback": review.feedback,
        "passed": review.passed_rules,
        "failed": review.failed_rules,
    }

    current_history = list(state.get("review_history", []))
    current_history.append(history_item)

    return {
        "decision": review.decision,
        "score": review.score,
        "feedback": review.feedback,
        "review_history": current_history,
    }


def reviser_node(state: AgentState) -> Dict[str, Any]:
    """
    Reviser Agent:
    Self-corrects the draft by actively applying the reviewer's critique.
    """
    llm = get_llm("reviser")
    system_prompt = (
        "You are the REVISER agent in a self-correcting multi-agent system.\n"
        "Your task: Rewrite and upgrade the draft to strictly address the reviewer's feedback.\n"
        "Ensure the everyday analogy is explicit and the concrete example is crystal clear.\n"
        "Keep the explanation concise, high-impact, and beginner friendly.\n"
        "Output ONLY the revised draft."
    )

    user_prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Original Draft:\n{state['draft']}\n\n"
        f"Reviewer Feedback:\n{state['feedback']}\n\n"
        f"Please provide the corrected revision now."
    )

    response = llm.invoke([
        _msg("system", system_prompt),
        _msg("user", user_prompt)
    ])

    new_draft = _content_to_text(response.content)
    new_revision_count = state.get("revision_count", 0) + 1

    return {
        "draft": new_draft,
        "revision_count": new_revision_count,
    }


# -----------------------------------------------------------------------------
# Conditional Routing Edge
# -----------------------------------------------------------------------------
def route_after_review(state: AgentState) -> str:
    """
    Cyclic Self-Correction Controller:
    - If Evaluator passes -> transition to END.
    - If max revisions exceeded -> graceful fallback to END.
    - Otherwise -> loop back to reviser node.
    """
    if state["decision"] == "PASS":
        return "done"
    if state["revision_count"] >= MAX_REVISIONS:
        return "done"
    return "revise"


# -----------------------------------------------------------------------------
# StateGraph Assembly & Compilation
# -----------------------------------------------------------------------------
try:
    builder = StateGraph(AgentState)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("reviser", reviser_node)

    builder.add_edge(START, "writer")
    builder.add_edge("writer", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "revise": "reviser",
            "done": END,
        },
    )
    builder.add_edge("reviser", "reviewer")

    # Compile executable state graph
    multi_agent_graph = builder.compile()
except Exception:
    multi_agent_graph = None


# -----------------------------------------------------------------------------
# Execution Drivers
# -----------------------------------------------------------------------------
def run_workflow(topic: str) -> Dict[str, Any]:
    """
    Execute the multi-agent graph synchronously and aggregate event stream trace.
    """
    initial_state: AgentState = {
        "topic": topic,
        "draft": "",
        "feedback": "",
        "decision": "",
        "score": 0.0,
        "revision_count": 0,
        "review_history": [],
    }

    final_state = initial_state.copy()
    events: List[Dict[str, Any]] = []

    if multi_agent_graph is not None:
        for update in multi_agent_graph.stream(initial_state, stream_mode="updates"):
            for node_name, values in update.items():
                final_state.update(values)
                events.append({
                    "agent": node_name,
                    "draft": values.get("draft", ""),
                    "decision": values.get("decision", ""),
                    "score": values.get("score", 0.0),
                    "feedback": values.get("feedback", ""),
                    "revision_count": final_state.get("revision_count", 0),
                })
    else:
        # Standalone cyclic state machine execution
        w_out = writer_node(final_state)
        final_state.update(w_out)
        events.append({"agent": "writer", **w_out})

        while True:
            rev_out = reviewer_node(final_state)
            final_state.update(rev_out)
            events.append({"agent": "reviewer", **rev_out})

            next_step = route_after_review(final_state)
            if next_step == "done":
                break

            reviser_out = reviser_node(final_state)
            final_state.update(reviser_out)
            events.append({"agent": "reviser", **reviser_out})

    return {
        "topic": topic,
        "provider": LLM_PROVIDER,
        "final_answer": final_state.get("draft", ""),
        "final_decision": final_state.get("decision", ""),
        "final_score": final_state.get("score", 0.0),
        "total_revisions": final_state.get("revision_count", 0),
        "events": events,
        "review_history": final_state.get("review_history", []),
    }


if __name__ == "__main__":
    import sys
    test_topic = sys.argv[1] if len(sys.argv) > 1 else "What is a LangGraph StateGraph?"
    print(f"\n🚀 Running Self-Correcting Multi-Agent System on: '{test_topic}'\n")
    res = run_workflow(test_topic)
    print("=" * 60)
    print(f"Final Decision: {res['final_decision']} (Score: {res['final_score']})")
    print(f"Revisions Used: {res['total_revisions']}")
    print("=" * 60)
    print("\nFinal Output:\n", res["final_answer"])
