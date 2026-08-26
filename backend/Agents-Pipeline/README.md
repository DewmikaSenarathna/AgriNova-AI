# Agents-Pipeline (Phase 7 + Phase 8 + Phase 9 + Phase 10)

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

Phase 9 gives agents **tools**: a `tools/` package that formalizes
every external capability an agent reaches for (an HTTP API, a local
dataset, a full-text search, a vector database, a vision model) behind
one shared interface — see [Connect External Tools](#connect-external-tools-phase-9)
below.

Phase 10 is **Multi-Agent Collaboration** — the heart of Agentic AI:
by default, the agents the Planner selects no longer run in isolation.
They run ONE AT A TIME, in the Planner's chain order, and every agent
is handed every EARLIER agent's findings so it can genuinely build on
them — see [Multi-Agent Collaboration](#multi-agent-collaboration-phase-10)
below.

```
Farmer asks (+ optional photo)
     │
     ▼
┌────────────────────── Planner Agent ─────────────────────────────┐
│  planner_agent.py → decides WHICH agent(s) run, and in what ORDER│
└──────────────────────────────────────────────────────────────────┘
     │
     ▼  (Phase 10 default: SEQUENTIAL — each agent below receives every
     │   earlier agent's findings via request.context["prior_findings"])
     ▼
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌─────────────┐
│  Disease  │──▶│  Weather  │──▶│   Soil     │──▶│ Fertilizer  │   ...and so on for
│  Agent    │    │  Agent    │    │  Agent    │    │  Agent      │   whichever agents
└───────────┘    └───────────┘    └───────────┘    └─────────────┘   the Planner picked
     │                 │                │                  │        (Market, Government,
     ▼                 ▼                ▼                  ▼         Pest, Image, ...)
  Vector DB       Weather API       Vector DB           Vector DB
    tool              tool            tool                 tool
     │
     │ (one agent failing never stops the chain — see base_agent.execute())
     ▼
┌────────────────────── Report Agent ────────────────────────────┐
│  report_agent.py → the Planner's closing step: combines every  │
│                     agent's findings into ONE consolidated,    │
│                     source-cited recommendation                │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
        Reliable farming recommendation ("Final Answer")
```

Example (the canonical Phase 10 scenario):

> **Farmer asks:** "My tomato plants are turning yellow. Should I water them today?"
>
> **Chain:** Planner → Disease Agent → Weather Agent → Soil Agent →
> Fertilizer Agent → Planner (Report Agent) → Final Answer
>
> Every agent contributes: Disease Agent checks whether yellowing
> matches a known disease; Weather Agent checks whether recent/upcoming
> rain already explains it (and whether watering today makes sense);
> Soil Agent checks moisture/drainage (over- or under-watered soil also
> yellows leaves); Fertilizer Agent checks whether it's actually a
> nutrient deficiency. Each one sees what the agents before it already
> found (see [Multi-Agent Collaboration](#multi-agent-collaboration-phase-10)).

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
responsibility. `GET /api/tools` (Phase 9) lists every external tool
and which agent(s) use it.

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

## Connect External Tools (Phase 9)

Agents become powerful when they can use tools. Every external
capability an agent reaches for now lives in `tools/`, behind one
shared contract (`tools/base_tool.py`'s `BaseTool` — same
"single-responsibility, never raise out of the public entry point"
design as `base_agent.py`, one layer down):

| Tool | File | Used by | External capability |
|---|---|---|---|
| Weather API | `tools/weather_tool.py` | Weather Agent | Open-Meteo geocoding + forecast |
| Market Price API | `tools/market_price_tool.py` | Market Agent | Local price dataset (swap for a live pricing API) |
| Government PDF Search | `tools/government_pdf_search_tool.py` | Government Agent | Full-text keyword search over processed official PDFs |
| Vector Database | `tools/vector_db_tool.py` | Disease / Pest / Fertilizer / Soil / Government / Market / General Agent | Shared ChromaDB similarity search |
| Image Model | `tools/image_model_tool.py` | Image Agent | Vision-capable LLM (crop photo → plain-language description) |

```
Agent
  │
  ▼
tool.execute(**kwargs)   ← never raises; degrades to ToolResult(ok=False, error=...)
  │
  ▼
tool.run(**kwargs)        ← subclass implements the one external call here
  │
  ▼
ToolResult(ok, data, text, source, error)
```

Why tools are a layer separate from agents at all: several agents need
the *same* capability. Every knowledge-backed agent needs the vector
database; the Government Agent needs it *and* a PDF full-text search.
Putting each capability behind one tool means there's exactly one
place to swap Open-Meteo for a different weather provider, swap the
local market JSON for a live pricing API, or point the vision model at
a different backend — without touching the agents that call it.

**Government Agent now calls two tools.** It merges semantic search
(`vector_database` — good at "what's this about" even when wording
doesn't match) with literal keyword search over the original PDF text
(`government_pdf_search` — good at an exact scheme name or clause, and
works independently of the embedding model / ChromaDB being healthy).
`government_pdf_search` reads `Document-Processing-Pipeline`'s own
`output/clean_text/*.txt` + `output/metadata/*.json`, filtered to
documents whose `document_type` is `"Government Scheme"` (see that
pipeline's `metadata.py` — the keywords `government`/`scheme`/
`subsidy`/`gazette`/`ministry` all map to that label).

**Image Agent + Image Model tool.** A farmer can attach a photo
(`context["image_base64"]`, or the API's top-level `image_base64`
field). `image_agent.py` sends it to a vision-capable LLM via
`LLMClient.generate_vision()` (see `../RAG-Pipeline/llm_client.py`,
configured through `OLLAMA_VISION_MODEL` / `GROQ_VISION_MODEL` /
`OPENAI_COMPATIBLE_VISION_MODEL` in `../RAG-Pipeline/.env`) and returns
a plain-language description of what's visible — never a diagnosis
itself, and always `grounded=False` since it's model inference, not
retrieved evidence. `agent_orchestrator.py` runs the Image Agent
*first* whenever one is selected (or auto-injects it into the plan the
moment a photo is attached, even if the question's wording didn't
mention one) and folds its description into every other selected
agent's context as `context["image_description"]` — so a photo of
yellowing, spotted leaves can help Disease Agent find the right
knowledge-base sources even if the farmer's own words never said
"yellowing" or "spots" (see `knowledge_agent.py`'s use of that key).

**Nothing about existing agent behaviour changed** — Weather/Market
Agent's outward behaviour is identical to Phase 7/8, they just call
`WeatherTool`/`MarketPriceTool` instead of doing the HTTP/JSON work
inline. `GET /api/tools` lists every registered tool and which
agent(s) use it, the same way `GET /api/agents` already did for
agents.

## Multi-Agent Collaboration (Phase 10)

Phases 7–9 ran every agent the Planner selected **independently** —
each one only ever saw the farmer's raw question (plus, since Phase 9,
an attached photo's description). That's fan-out, not collaboration:
the Fertilizer Agent had no idea what the Soil Agent or Weather Agent
had just found for the very same question.

Phase 10 changes the default execution mode
(`agent_config.COLLABORATION_MODE`, default `"sequential"`) so agents
run **one at a time, in the Planner's chain order**, and every agent
after the first is handed the accumulated findings of every agent that
ran before it:

```python
# agent_orchestrator.py — AgentOrchestrator._run_sequential_collaboration()
for agent_name in ordered_agent_names:
    request_context = {**enriched_context, "prior_findings": list(prior_findings)}
    result = self.agent_registry[agent_name].execute(AgentRequest(query=question, context=request_context))
    agent_results.append(result)
    prior_findings.append({
        "agent_name": result.agent_name, "summary": result.summary,
        "details": result.details, "grounded": result.grounded,
    })
```

Every knowledge-backed agent (`knowledge_agent.py`, and therefore
Disease/Pest/Fertilizer/Soil/Government/Market Agent) and the Weather
Agent (`weather_agent.py`) read `request.context["prior_findings"]`
via the shared `agent_types.format_prior_findings()` helper and fold
it into their LLM prompt *ahead of* their own retrieved sources — so
e.g. the Fertilizer Agent's prompt literally includes what the Soil
Agent and Weather Agent already concluded, and is told to build on
it rather than repeat or ignore it.

**Chain order comes from the Planner, unchanged from Phase 8** — see
`planner_agent.py`'s `_REASONING_CHAINS`. The canonical example:

```
"My tomato plants are turning yellow. Should I water them today?"

Planner
  │
  ▼
Disease Agent    → is this a known disease matching "yellowing"?
  │  (finding passed forward)
  ▼
Weather Agent    → does recent/upcoming rain already explain this,
  │                 and is today even a good day to water?
  │  (both findings passed forward)
  ▼
Soil Agent       → is the soil itself waterlogged / poorly drained /
  │                 actually dry — given what Disease + Weather found?
  │  (all three findings passed forward)
  ▼
Fertilizer Agent → could this be a nutrient deficiency instead,
  │                 given everything found so far?
  ▼
Planner (Report Agent) → combines all four into ONE final answer
  │
  ▼
Final Answer
```

Run it and see the chain for yourself:

```bash
python main.py "My tomato plants are turning yellow. Should I water them today?"
```

**The Image Agent is a special case, unchanged from Phase 9:** its
photo description still flows through `context["image_description"]`
(read directly by `knowledge_agent.py`), not through
`prior_findings` — it's raw visual evidence, not a specialist's
conclusion, and every knowledge agent already knew how to use it.

**One agent failing never breaks the chain.** `BaseAgent.execute()`
(Phase 7) still turns any exception into an honest `AgentResult.error`
— if, say, the Weather API is down, Soil/Fertilizer Agent still run
next, they just won't have a weather finding to build on.

**Parallel mode (Phase 7's original fan-out) is still available**,
e.g. for lower latency on simple, single-domain questions where true
collaboration isn't needed:

```bash
# .env or shell
COLLABORATION_MODE=parallel
```

`GET /health` and every `/api/agents/ask` response report which mode
actually produced the answer (`"collaboration_mode": "sequential" |
"parallel"`), and the CLI's `PLANNER'S REASONING` header shows it too.

Unit tests for this (no LLM / network required — uses in-memory fake
agents) live in `tests/test_phase10_collaboration.py`:

```bash
python -m unittest tests.test_phase10_collaboration -v
```

## Files

| File | Responsibility |
|---|---|
| `agent_config.py` | Every Phase-7/8/9/10-specific setting (planner mode, weather/market/gov-search/image config, `COLLABORATION_MODE`, API port) |
| `rag_bridge.py` | Re-exports Phase 6's embedder/vector store/retriever/LLM client/RAG pipeline |
| `agent_types.py` | Shared `AgentRequest` / `AgentResult` / `PlanDecision` / `PlanStep` dataclasses + Phase 10's `format_prior_findings()` |
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
* **Tools are a separate layer from agents (Phase 9).** An agent
  *reasons about* what a tool returns; it doesn't know or care how the
  tool reaches an external system. That split is what let the
  Government Agent gain a second evidence source (PDF search) and the
  whole pipeline gain an Image Agent without touching any other
  agent's code, and it's the one place to swap a demo integration
  (the local market-price JSON, Open-Meteo) for a production one
  later.
