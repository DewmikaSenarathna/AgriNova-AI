"""
api.py
======
PHASE 7/10/11/13 — Agents Pipeline exposed as a FastAPI service

This is the agentic counterpart to `RAG-Pipeline/api.py`. It runs on
its own port (default 8001) so both can be run side by side:

    GET    /health                     -> service + knowledge base + agent status
    GET    /api/agents                  -> lists every registered specialist agent
    POST   /api/agents/ask               -> { "question": "...", "context": {...},
                                              "session_id": "..." }
                                             -> full plan + every specialist's
                                                findings + one consolidated report +
                                                a Phase 13 Recommendation/Reason/
                                                Supporting-documents/Confidence/
                                                References explanation
    GET    /api/memory/{session_id}       -> PHASE 11 — what AgriNova AI currently
                                             remembers about that farmer/session
    DELETE /api/memory/{session_id}       -> PHASE 11 — forget that session
                                             ("my crop failed, start over")

Run it with:

    uvicorn api:app --reload --host 0.0.0.0 --port 8001

Then, e.g.:

    curl -X POST http://localhost:8001/api/agents/ask \\
         -H "Content-Type: application/json" \\
         -d '{"question": "my tomato leaves have brown spots and it might rain this week, what should I do?", \\
              "session_id": "farmer-42"}'

PHASE 11 — pass the SAME `session_id` on every request from the same
farmer/device/login and AgriNova AI will stop asking them to repeat
crop/location/previous findings that were already established in an
earlier request. Omitting `session_id` entirely (Phase 7-10 behaviour)
skips conversation memory for that request.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import agent_config
from agent_orchestrator import AgentOrchestrator
from rag_bridge import VectorStoreEmpty, VectorStoreUnavailable, rag_config
from tools import TOOL_REGISTRY

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
    # Phase 9 — convenience top-level field so a frontend can attach a
    # crop photo without needing to know the internal context key name.
    # Accepts either a bare base64 string or a full data: URL
    # ("data:image/jpeg;base64,...") — tools/image_model_tool.py strips
    # the data: prefix if present. Forwarded into context["image_base64"]
    # below so it flows through exactly like any other context hint.
    image_base64: Optional[str] = Field(
        default=None,
        description="Optional base64-encoded crop photo, routed to the Image Agent.",
    )
    # PHASE 11 — identifies which farmer's ongoing conversation this
    # question belongs to. Pass the SAME session_id on every request
    # from the same farmer/device/login to get conversation memory
    # (crop, location, previous findings, weather history carried
    # forward without the farmer repeating themselves); omit it to opt
    # out and get Phase 7-10's stateless behaviour for this request.
    session_id: Optional[str] = Field(
        default=None,
        description="Optional conversation/session identifier for memory across turns.",
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


class ConfidenceOut(BaseModel):
    """PHASE 13 — see explainability.py's `compute_confidence` for the formula."""
    level: str  # "Low" | "Medium" | "High"
    score: float  # 0.0 - 1.0
    factors: List[str]


class ReferenceOut(BaseModel):
    """PHASE 13 — one numbered, de-duplicated bibliography entry."""
    n: int
    label: str
    agent: Optional[str] = None
    similarity: Optional[float] = None
    detail: Optional[str] = None


class ExplanationOut(BaseModel):
    """PHASE 13 — Recommendation -> Reason -> Supporting documents ->
    Confidence -> References, built by explainability.py's
    `build_explanation()` from the Report Agent's structured output."""
    recommendation: str
    reason: str
    next_steps: str
    supporting_documents: List[Dict[str, Any]]
    confidence: ConfidenceOut
    references: List[ReferenceOut]


class AskResponse(BaseModel):
    question: str
    plan: PlanOut
    agent_results: List[AgentResultOut]
    final_report: AgentResultOut
    # PHASE 10 — "sequential" (agents collaborated in the Planner's chain
    # order, each seeing earlier agents' findings) or "parallel" (Phase 7
    # fan-out). See agent_config.COLLABORATION_MODE.
    collaboration_mode: str
    # PHASE 11 — echoes the session_id this request used (None if the
    # caller didn't pass one), and exactly which facts were recalled
    # from earlier turns and used to answer THIS question.
    session_id: Optional[str] = None
    recalled_memory: Dict[str, Any] = Field(default_factory=dict)
    # PHASE 13 — see ExplanationOut above.
    explanation: Optional[ExplanationOut] = None


class MemoryOut(BaseModel):
    """PHASE 11 — what AgriNova AI currently remembers about one session."""
    session_id: str
    crop: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    field_name: Optional[str] = None
    last_disease: Optional[Dict[str, str]] = None
    last_fertilizer: Optional[Dict[str, str]] = None
    last_pest: Optional[Dict[str, str]] = None
    last_soil_note: Optional[Dict[str, str]] = None
    weather_history: List[Dict[str, str]] = Field(default_factory=list)
    turns: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str
    updated_at: str


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
            "collaboration_mode": agent_config.COLLABORATION_MODE,
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


@app.get("/api/tools")
def list_tools() -> Dict:
    """PHASE 9 — lists every external tool this pipeline can reach, and
    which agent(s) use it (Weather API, Market Price API, Government PDF
    Search, Vector Database, Image Model)."""
    return {"tools": {name: spec.to_dict() for name, spec in TOOL_REGISTRY.items()}}


@app.post("/api/agents/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    The main agentic endpoint:
        Farmer asks -> Planner Agent -> specialist agent(s) -> Report Agent

    Phase 9: if `image_base64` is set, it's merged into `context` as
    `image_base64` so the Image Agent (and, transitively, the Disease/
    Pest/etc. agents that read `context["image_description"]` once the
    Image Agent has run) can use it — see agent_orchestrator.py.

    Phase 11: if `session_id` is set, this question is answered with
    (and updates) that session's conversation memory — see
    agent_orchestrator.py's module docstring and conversation_memory.py.
    """
    context = dict(request.context or {})
    if request.image_base64:
        context["image_base64"] = request.image_base64

    try:
        result = get_orchestrator().handle(
            request.question, context=context, session_id=request.session_id
        )
    except Exception as e:  # last-resort safety net so the API never 500s silently
        logger.exception("Unexpected error while handling an agentic request.")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}") from e

    return AskResponse(**result.to_dict())


@app.get("/api/memory/{session_id}", response_model=MemoryOut)
def get_memory(session_id: str) -> MemoryOut:
    """PHASE 11 — inspect exactly what AgriNova AI currently remembers
    about one farmer's conversation (crop, location, previous findings,
    weather history, recent questions). Returns an all-empty memory
    (not a 404) for a session_id that has never been used, since "no
    memory yet" is a normal, expected state, not an error."""
    memory = get_orchestrator().memory_store.get(session_id)
    return MemoryOut(**memory.to_dict())


@app.delete("/api/memory/{session_id}")
def delete_memory(session_id: str) -> Dict:
    """PHASE 11 — forgets everything remembered about one session (e.g.
    the farmer starts a new crop cycle, or wants a clean slate)."""
    deleted = get_orchestrator().memory_store.delete(session_id)
    return {"session_id": session_id, "deleted": deleted}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=agent_config.API_HOST, port=agent_config.API_PORT, reload=True)
