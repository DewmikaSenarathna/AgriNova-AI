

Turns the single-shot question-answering assistant from `RAG-Pipeline`
(Phase 6) into **Agentic AI**: instead of one generalist retriever, a
**Planner Agent** decides which specialized agent(s) a farmer's
question needs, each does its one job, and a **Report Agent** combines
everything into one consolidated, source-cited recommendation.

Phase 8 upgrades the Planner from a flat router into "the manager": it
now produces a visible reasoning **chain** — "Need X → Need Y → ... →
Need a recommendation" — including needs the farmer didn't explicitly
ask for but a good agronomist would still check (see
[Planner Agent modes](#planner-agent-modes) below for the canonical
"Should I apply fertilizer tomorrow?" example).


```
Farmer asks (+ optional photo)
     │
     ▼
┌────────────────────── Planner Agent ───────────────────────────┐
│  planner_agent.py → decides WHICH agent(s) below should run    │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌───────────┬───────────┬───────────┬─────────────┬───────────┬─────────────┬───────────┬───────────┐
│  Disease  │  Weather  │  Market   │ Government  │   Soil    │ Fertilizer  │   Pest    │   Image   │
│  Agent    │  Agent    │  Agent    │  Agent      │  Agent    │  Agent      │  Agent    │   Agent   │
│    │      │    │      │    │      │   │    │    │    │      │             │           │     │     │
│    ▼      │    ▼      │    ▼      │   ▼    ▼    │    ▼      │      ▼      │     ▼     │     ▼     │
│ Vector DB │ Weather   │ Market    │ Vector DB    │ Vector DB│  Vector DB  │ Vector DB │  Image    │
│  tool     │ API tool  │ Price API │ + Gov PDF    │  tool    │    tool     │   tool    │  Model    │
│           │           │  tool     │ Search tools │          │             │           │   tool    │
└───────────┴───────────┴───────────┴─────────────┴───────────┴─────────────┴───────────┴───────────┘
     │ (each agent runs independently — one failing never stops the others)
     ▼
┌────────────────────── Report Agent ────────────────────────────┐
│  report_agent.py → combines every agent's findings into ONE    │
│                     consolidated, source-cited recommendation  │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
        Reliable farming recommendation
```

If the Planner can't confidently match a question to any specialist,
it routes to the **General Agent** — the plain Phase 6 `RAGPipeline`
over the whole knowledge base — so the farmer always gets a grounded
answer instead of an empty routing table.

## Each agent, single responsibility

| Agent | File | Responsibility | Tool(s) it calls (Phase 9) |
|---|---|---|---|
| Planner Agent | `planner_agent.py` | Decides which specialist(s) a question needs | — (keyword rules, or the LLM if `PLANNER_MODE=llm`) |
| Disease Agent | `disease_agent.py` | Crop disease diagnosis & treatment | `vector_database` |
| Weather Agent | `weather_agent.py` | Short-range forecast & farming implications | `weather_api` |
| Market Agent | `market_agent.py` | Crop market prices & sell/hold guidance | `market_price_api`, then `vector_database` |
| Government Agent | `government_agent.py` | Schemes, subsidies, official guidelines | `government_pdf_search` **and** `vector_database` |
| Soil Agent | `soil_agent.py` | Soil health, pH, land preparation | `vector_database` |
| Fertilizer Agent | `fertilizer_agent.py` | Fertilizer type, dosage, timing | `vector_database` |
| Pest Agent | `pest_agent.py` | Pest identification & management | `vector_database` |
| Image Agent | `image_agent.py` | Describes a farmer-submitted crop photo | `image_model` |
| General Agent | `general_agent.py` | Fallback for anything else | `vector_database` (via `RAGPipeline`) |
| Report Agent | `report_agent.py` | Combines every agent's findings into one report | — (reasons over other agents' `AgentResult`s) |

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

Attach a photo (Phase 9) with the top-level `image_base64` field —
either a bare base64 string or a full `data:image/jpeg;base64,...` URL:

```bash
curl -X POST http://localhost:8001/api/agents/ask \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"what is wrong with this plant?\", \"image_base64\": \"$(base64 -w0 leaf.jpg)\"}"
```

`GET /health` reports API + knowledge base + LLM readiness. `GET
/api/agents` lists every registered agent and its one-line


## Planner Agent: the manager (Phase 8)

Given **"Should I apply fertilizer tomorrow?"** — a question that only
mentions fertilizer — the Planner produces this reasoning chain:

```
Need the weather outlook            → weather_agent
    (rain shortly after application can wash fertilizer away)
        ↓
Need the fertilizer type, dosage and timing → fertilizer_agent
    (the farmer's core question)
        ↓
Need the crop's growth stage         → soil_agent
    (fertilizer needs change with growth stage)
        ↓
Need rainfall in the coming days      → weather_agent
    (confirms conditions stay dry long enough)
        ↓
Need a final recommendation            → report_agent
    (combines everything above)
```

Every `PlanStep` in that chain (`agent_types.PlanStep`) carries a
`need`, the `agent` that supplies it, and a `reason` — the chain is
inspectable end-to-end via `PlanDecision.steps`, not just a flat
`agents_to_run` list. Fertilizer, pest and disease questions each have
a hand-authored chain like this in `planner_agent._REASONING_CHAINS`,
because those three domains most often need a weather check the
farmer didn't think to ask for. Simpler domains (market, soil,
government, weather itself) get a single-step chain. `agents_to_run`
is the flattened, de-duplicated, execution-order list the orchestrator
actually runs — `report_agent` is excluded from it since the
orchestrator runs the Report Agent separately (see
`agent_orchestrator.py`).

**Two routing modes**, chosen via `PLANNER_MODE`:

* **`keyword`** (default) — fast, free, deterministic. Matches the
  question against a per-agent keyword list, then looks up that
  domain's reasoning chain. Zero extra latency and zero extra LLM
  spend; the trade-off is it only recognizes phrasing it has keywords
  for, and its dependency chains are fixed in advance.
* **`llm`** — asks the configured LLM to produce the WHOLE reasoning
  chain (needs, agents, reasons) as strict JSON, the way an agronomist
  would think out loud. Copes better with unusual phrasing,
  multi-intent questions, or dependencies the hand-authored chains
  don't cover, at the cost of one extra LLM call per question. Falls
  back to keyword routing automatically if the LLM call or its JSON
  parsing fails.

Both modes cap the number of agents *run* per request at
`PLANNER_MAX_AGENTS_PER_REQUEST` (default 4) to keep latency and LLM
spend bounded, and both fall back to the General Agent when nothing
matches confidently. The Report Agent also receives the full plan (via
`agent_orchestrator.py`), so the final report can be framed the way
the plan intended instead of re-guessing why each specialist was
consulted.


## Files

| File | Responsibility |
|---|---|
| `agent_config.py` | Every Phase-7/8/9-specific setting (planner mode, weather/market/gov-search/image config, API port) |
| `rag_bridge.py` | Re-exports Phase 6's embedder/vector store/retriever/LLM client/RAG pipeline |
| `agent_types.py` | Shared `AgentRequest` / `AgentResult` / `PlanDecision` / `PlanStep` dataclasses |
| `base_agent.py` | Abstract base class every agent implements; catches per-agent failures |
| `tools/` | Phase 9 — `BaseTool` contract + Weather API / Market Price API / Government PDF Search / Vector Database / Image Model tools |
| `knowledge_agent.py` | Shared retrieval (via `tools.vector_db_tool`) + grounded-generation logic for the RAG-backed agents |
| `planner_agent.py` | Decides which agent(s) should run |
| `disease_agent.py` | Crop disease diagnosis & treatment |
| `fertilizer_agent.py` | Fertilizer type, dosage, timing |
| `pest_agent.py` | Pest identification & management |
| `soil_agent.py` | Soil health, pH, land preparation |
| `government_agent.py` | Government schemes, subsidies, guidelines (vector DB + PDF search) |
| `weather_agent.py` | Live forecast + farming implications (via `tools.weather_tool`) |
| `market_agent.py` | Crop market prices + sell/hold guidance (via `tools.market_price_tool`) |
| `image_agent.py` | Describes an attached crop photo (via `tools.image_model_tool`) |
| `general_agent.py` | Fallback: plain Phase 6 RAG over the whole knowledge base |
| `report_agent.py` | Combines every agent's findings into one final report |
| `agent_orchestrator.py` | The main orchestrator — `AgentOrchestrator().handle(question, context)` |
| `main.py` | Interactive CLI entry point (`--image <path>` to attach a photo) |
| `api.py` | FastAPI service (`/api/agents/ask`, `/api/agents`, `/api/tools`, `/health`) |
| `data/market_prices_sample.json` | Demo price dataset used by the Market Agent / `MarketPriceTool` |

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
* **Reasoning chains, not just routing (Phase 8).** The Planner infers
  needs the farmer didn't state — a fertilizer-timing question always
  gets a weather check, a pest/disease question always gets a
  spraying-conditions check — because that's what a competent
  agronomist does automatically. Making that chain a first-class,
  inspectable `PlanStep` list (rather than baking the dependency logic
  invisibly into which agents happen to fire) is what keeps the
  Planner's decisions explainable to a farmer, a developer, and this
  README all at once.

