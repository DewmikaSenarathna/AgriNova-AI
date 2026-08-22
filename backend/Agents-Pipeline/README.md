# Agents-Pipeline (Phase 7)

Turns the single-shot question-answering assistant from `RAG-Pipeline`
(Phase 6) into **Agentic AI**: instead of one generalist retriever, a
**Planner Agent** decides which specialized agent(s) a farmer's
question needs, each does its one job, and a **Report Agent** combines
everything into one consolidated, source-cited recommendation.

```
Farmer asks
     │
     ▼
┌────────────────────── Planner Agent ───────────────────────────┐
│  planner_agent.py → decides WHICH agent(s) below should run      │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌───────────┬───────────┬───────────┬─────────────┬───────────┬─────────────┬───────────┐
│  Disease  │  Weather  │  Market   │ Government  │   Soil    │ Fertilizer  │   Pest    │
│  Agent    │  Agent    │  Agent    │  Agent      │  Agent    │  Agent      │  Agent    │
└───────────┴───────────┴───────────┴─────────────┴───────────┴─────────────┴───────────┘
     │ (each agent runs independently — one failing never stops the others)
     ▼
┌────────────────────── Report Agent ────────────────────────────┐
│  report_agent.py → combines every agent's findings into ONE      │
│                     consolidated, source-cited recommendation     │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
        Reliable farming recommendation
```

If the Planner can't confidently match a question to any specialist,
it routes to the **General Agent** — the plain Phase 6 `RAGPipeline`
over the whole knowledge base — so the farmer always gets a grounded
answer instead of an empty routing table.

## Each agent, single responsibility

| Agent | File | Responsibility | Evidence source |
|---|---|---|---|
| Planner Agent | `planner_agent.py` | Decides which specialist(s) a question needs | keyword rules, or the LLM (`PLANNER_MODE=llm`) |
| Disease Agent | `disease_agent.py` | Crop disease diagnosis & treatment | knowledge base (RAG) |
| Weather Agent | `weather_agent.py` | Short-range forecast & farming implications | live Open-Meteo API call |
| Market Agent | `market_agent.py` | Crop market prices & sell/hold guidance | local price dataset, then knowledge base |
| Government Agent | `government_agent.py` | Schemes, subsidies, official guidelines | knowledge base (RAG) |
| Soil Agent | `soil_agent.py` | Soil health, pH, land preparation | knowledge base (RAG) |
| Fertilizer Agent | `fertilizer_agent.py` | Fertilizer type, dosage, timing | knowledge base (RAG) |
| Pest Agent | `pest_agent.py` | Pest identification & management | knowledge base (RAG) |
| General Agent | `general_agent.py` | Fallback for anything else | knowledge base (RAG), whole KB |
| Report Agent | `report_agent.py` | Combines every agent's findings into one report | the other agents' `AgentResult`s |

The five knowledge-backed specialists (Disease, Fertilizer, Pest, Soil,
Government) share one implementation, `knowledge_agent.py` — each is a
~15-line subclass supplying only what makes it different: a domain
label, query-expansion hints, and a domain-specific system prompt. This
keeps "single responsibility" honest without five near-duplicate
retrieval implementations.

## How this reuses Phase 6

Nothing about embedding, ChromaDB, or the LLM client is duplicated.
`rag_bridge.py` is the single choke point that imports
`../RAG-Pipeline`'s `embedder.py`, `vector_store.py`, `retriever.py`,
`llm_client.py` and `rag_pipeline.py` and re-exports them here. One
shared `Retriever` and one shared `LLMClient` are built once in
`agent_orchestrator.py` and passed to every agent that needs them —
just like `RAGPipeline` avoids reloading the embedding model per
request.

> Agents-Pipeline's own settings file is named `agent_config.py`, not
> `config.py` — seeing two same-named `config.py` modules on
> `sys.path` at once (one per pipeline folder) would let whichever
> gets imported first silently shadow the other. See the comment at
> the top of `rag_bridge.py` for the full explanation.

## Prerequisites

1. **Document-Processing-Pipeline** has processed at least one PDF (Phase 3).
2. **Chunking-Embedding-Pipeline** has chunked and embedded it into
   `../../vector_db` (Phase 4 + 5).
3. **RAG-Pipeline** is configured — copy `../RAG-Pipeline/.env.example`
   to `../RAG-Pipeline/.env` and make sure an LLM backend (Ollama /
   Groq / OpenAI-compatible) is reachable. Agents-Pipeline reuses those
   exact settings via `rag_bridge.py`.

## Run it

```bash
pip install -r ../RAG-Pipeline/requirements.txt   # embedding model, ChromaDB, FastAPI, ...
pip install -r requirements.txt                    # this pipeline's own extras (requests)
cp .env.example .env                                # optional — defaults work out of the box

# Interactive CLI:
python main.py

# One-off question:
python main.py "my tomato leaves have brown spots and it might rain this week, what should I do?"

# Serve it as an API for the frontend (separate port from RAG-Pipeline/api.py):
uvicorn api:app --reload --host 0.0.0.0 --port 8001
```

### API example

```bash
curl -X POST http://localhost:8001/api/agents/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "my tomato leaves have brown spots and it might rain this week, what should I do?"}'
```

```json
{
  "question": "my tomato leaves have brown spots and it might rain this week, what should I do?",
  "plan": {
    "agents_to_run": ["disease_agent", "weather_agent"],
    "reasoning": "disease_agent <- ['spots on leaves']; weather_agent <- ['rain']",
    "method": "keyword"
  },
  "agent_results": [
    { "agent_name": "disease_agent", "grounded": true, "details": "...", "sources": [...] },
    { "agent_name": "weather_agent", "grounded": true, "details": "...", "data": {"current": {...}} }
  ],
  "final_report": {
    "agent_name": "report_agent",
    "details": "One combined, source-cited recommendation...",
    "grounded": true,
    "sources": [...]
  }
}
```

`GET /health` reports API + knowledge base + LLM readiness. `GET
/api/agents` lists every registered agent and its one-line
responsibility.

## Planner Agent modes

* **`PLANNER_MODE=keyword`** (default) — fast, free, deterministic.
  Matches the question against a per-agent keyword list
  (`_KEYWORD_MAP` in `planner_agent.py`). Zero extra latency and zero
  extra LLM spend; the trade-off is it only recognizes phrasing it has
  keywords for.
* **`PLANNER_MODE=llm`** — asks the configured LLM to choose agents as
  strict JSON. Copes better with unusual phrasing or multi-intent
  questions, at the cost of one extra LLM call per question. Falls
  back to keyword routing automatically if the LLM call or its JSON
  parsing fails.

Both modes cap the number of agents run per request at
`PLANNER_MAX_AGENTS_PER_REQUEST` (default 4) to keep latency and LLM
spend bounded, and both fall back to the General Agent when nothing
matches confidently.

## Files

| File | Responsibility |
|---|---|
| `agent_config.py` | Every Phase-7-specific setting (planner mode, weather/market config, API port) |
| `rag_bridge.py` | Re-exports Phase 6's embedder/vector store/retriever/LLM client/RAG pipeline |
| `agent_types.py` | Shared `AgentRequest` / `AgentResult` / `PlanDecision` dataclasses |
| `base_agent.py` | Abstract base class every agent implements; catches per-agent failures |
| `knowledge_agent.py` | Shared retrieval + grounded-generation logic for the 5 RAG-backed agents |
| `planner_agent.py` | Decides which agent(s) should run |
| `disease_agent.py` | Crop disease diagnosis & treatment |
| `fertilizer_agent.py` | Fertilizer type, dosage, timing |
| `pest_agent.py` | Pest identification & management |
| `soil_agent.py` | Soil health, pH, land preparation |
| `government_agent.py` | Government schemes, subsidies, guidelines |
| `weather_agent.py` | Live forecast + farming implications (Open-Meteo) |
| `market_agent.py` | Crop market prices + sell/hold guidance |
| `general_agent.py` | Fallback: plain Phase 6 RAG over the whole knowledge base |
| `report_agent.py` | Combines every agent's findings into one final report |
| `agent_orchestrator.py` | The main orchestrator — `AgentOrchestrator().handle(question)` |
| `main.py` | Interactive CLI entry point |
| `api.py` | FastAPI service (`/api/agents/ask`, `/api/agents`, `/health`) |
| `data/market_prices_sample.json` | Demo price dataset used by the Market Agent |

## Design notes

* **Single responsibility, enforced by the type contract.** Every
  agent takes an `AgentRequest` and returns an `AgentResult` — the
  Planner and Report Agent don't know or care how any specialist does
  its job internally, which is what makes it easy to add an eighth,
  ninth, tenth specialist later without touching the others.
* **One agent's failure never cancels the request.** `BaseAgent.execute()`
  catches any exception a specific agent raises and turns it into an
  `AgentResult.error` instead of letting it propagate — a down weather
  API degrades that one section of the report, it doesn't 500 the
  whole endpoint.
* **Honest grounding, all the way through.** Every `AgentResult` (and
  the final report) carries `grounded: bool` exactly like `RAGAnswer`
  does in Phase 6 — the Report Agent can tell the farmer plainly when
  a specialist found nothing relevant instead of quietly papering over
  the gap.
* **Deterministic-first, LLM-optional planning.** Keyword routing is
  the default specifically so the system is fast, free, and doesn't
  depend on the LLM being reachable just to decide who should answer;
  `PLANNER_MODE=llm` is there for when phrasing is too varied for
  keywords to catch reliably.
