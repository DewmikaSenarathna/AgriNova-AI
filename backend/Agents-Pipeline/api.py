"""
api.py
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import agent_config
from agent_orchestrator import AgentOrchestrator
from rag_bridge import VectorStoreEmpty, VectorStoreUnavailable, rag_config

logging.basicConfig(
    level=getattr(logging, agent_config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AgriNova AI — Agents Pipeline API",
    description="Multi-agent (Planner + 7 specialists + Report Agent) API for AgriNova AI.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=agent_config.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Built once and reused across requests — every knowledge agent shares
# one embedding model + ChromaDB connection + LLM client (see
# agent_orchestrator.AgentOrchestrator.__init__), which are all
# expensive to (re)initialize per request.
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The farmer's question.")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional hints, e.g. {\"crop\": \"tomato\", \"location\": \"Kurunegala\"}.",
    )


class AgentResultOut(BaseModel):
    agent_name: str
    summary: str
    details: str
    grounded: bool
    sources: List[Dict]
    data: Dict[str, Any]
    error: Optional[str]


class PlanStepOut(BaseModel):
    need: str
    agent: str
    reason: str


class PlanOut(BaseModel):
    agents_to_run: List[str]
    reasoning: str
    method: str
    steps: List[PlanStepOut]


class AskResponse(BaseModel):
    question: str
    plan: PlanOut
    agent_results: List[AgentResultOut]
    final_report: AgentResultOut


@app.get("/health")
def health() -> Dict:
    """Reports whether the API, the knowledge base, and the LLM backend are ready."""
    try:
        orchestrator = get_orchestrator()
        chunk_count = orchestrator.agent_registry["disease_agent"].retriever.store.count()
        return {
            "status": "ok",
            "knowledge_base_chunks": chunk_count,
            "llm_provider": rag_config.LLM_PROVIDER,
            "planner_mode": agent_config.PLANNER_MODE,
            "agents": list(orchestrator.agent_registry.keys()) + ["planner_agent", "report_agent"],
        }
    except (VectorStoreUnavailable, VectorStoreEmpty) as e:
        return {"status": "degraded", "reason": str(e), "llm_provider": rag_config.LLM_PROVIDER}


@app.get("/api/agents")
def list_agents() -> Dict:
    """Lists every registered agent and its single responsibility."""
    orchestrator = get_orchestrator()
    agents = {
        name: agent.description
        for name, agent in orchestrator.agent_registry.items()
    }
    agents["planner_agent"] = "Decides which specialist agent(s) a question needs."
    agents["report_agent"] = orchestrator.report_agent.description
    return {"agents": agents}


@app.post("/api/agents/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    The main agentic endpoint:
        Farmer asks -> Planner Agent -> specialist agent(s) -> Report Agent
    """
    try:
        result = get_orchestrator().handle(request.question, context=request.context)
    except Exception as e:  # last-resort safety net so the API never 500s silently
        logger.exception("Unexpected error while handling an agentic request.")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}") from e

    return AskResponse(**result.to_dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=agent_config.API_HOST, port=agent_config.API_PORT, reload=True)
