# Agents-Pipeline (Phase 7 + Phase 8 + Phase 9 + Phase 10 + Phase 11 + Phase 13)

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

Phase 11 is **Conversation Memory** — farmers shouldn't have to repeat
themselves. Pass the same `session_id` on every request from one
farmer and AgriNova AI remembers their crop, location, previous
disease/fertilizer findings and recent weather across turns (and
across CLI runs / API calls), instead of starting from zero every
question — see [Conversation Memory](#conversation-memory-phase-11)
below.

Phase 13 is **Explainable AI** — never answer without showing
evidence. Every final answer is broken into
`Recommendation -> Reason -> Supporting documents -> Confidence -> References`
instead of one block of prose, so a farmer (or an auditor) can actually
check it — see [Explainable AI](#explainable-ai-phase-13) below.
(Phase 12, the React frontend that renders all of this, lives in
`frontend/` — see its own README.)

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
│  Disease  │──▶│  Weather   │──▶│   Soil    │──▶│ Fertilizer  │   ...and so on for
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
  },
  "collaboration_mode": "sequential",
  "session_id": null,
  "recalled_memory": {},
  "explanation": {
    "recommendation": "Apply a copper-based fungicide and avoid overhead irrigation this week.",
    "reason": "Your tomato crop shows signs consistent with early blight [Disease Agent, Source 1]...",
    "next_steps": "- Apply copper fungicide\n- Avoid overhead irrigation\n- Monitor daily",
    "supporting_documents": [...],
    "confidence": {"level": "High", "score": 0.81, "factors": ["..."]},
    "references": [{"n": 1, "label": "Tomato Disease Guide", "agent": "disease_agent", "similarity": 0.82}]
  }
}
```

See [Explainable AI](#explainable-ai-phase-13) below for what builds
`explanation` and why.

Attach a photo (Phase 9) with the top-level `image_base64` field —
either a bare base64 string or a full `data:image/jpeg;base64,...` URL:

```bash
curl -X POST http://localhost:8001/api/agents/ask \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"what is wrong with this plant?\", \"image_base64\": \"$(base64 -w0 leaf.jpg)\"}"
```

Carry conversation memory across requests (Phase 11) with a
`session_id` — see [Conversation Memory](#conversation-memory-phase-11)
below for the full picture:

```bash
curl -X POST http://localhost:8001/api/agents/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "should I irrigate today?", "session_id": "farmer-42"}'
```

`GET /health` reports API + knowledge base + LLM readiness. `GET
/api/agents` lists every registered agent and its one-line
responsibility. `GET /api/tools` (Phase 9) lists every external tool
and which agent(s) use it. `GET /api/memory/{session_id}` /
`DELETE /api/memory/{session_id}` (Phase 11) inspect or clear one
session's remembered facts.

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

## Conversation Memory (Phase 11)

Farmers should not have to repeat themselves:

```
Day 1
  Farmer: "My tomato crop in Kurunegala has yellowing leaves."
  AI: [diagnoses early blight] — and REMEMBERS: crop=Tomato,
      location=Kurunegala, last_disease=early blight

Day 2
  Farmer: "Should I irrigate today?"
  AI already knows:
      Crop                → Tomato
      Location             → Kurunegala
      Field                → (whatever the farmer told it, if anything)
      Previous disease      → Early blight
      Previous fertilizer    → (whatever was last recommended)
      Weather history        → the last few forecasts checked
  → answers directly, and can factor in "since your tomatoes had
    early blight recently, avoid overhead irrigation."
```

**How it's identified:** a `session_id` string (one per farmer /
device / login — the caller decides what that maps to). Pass the SAME
`session_id` on every request from the same farmer to get memory
across turns:

```bash
python main.py "my tomato crop in Kurunegala has yellowing leaves" --session farmer-42
python main.py "should I irrigate today?" --session farmer-42
# ^ the second call already knows the crop and location from the first
```

```bash
curl -X POST http://localhost:8001/api/agents/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "should I irrigate today?", "session_id": "farmer-42"}'
```

Omitting `session_id` entirely skips memory load/save altogether —
existing callers that never pass one get exactly Phase 7-10's
stateless behaviour, unchanged.

**What gets remembered** (`conversation_memory.py`'s `FarmerMemory`):

| Fact | Where it comes from |
|---|---|
| `crop` | Explicit `context["crop"]`, else keyword-matched from the question text (e.g. "my tomato leaves..." → `Tomato`) |
| `location` / `latitude` / `longitude` | Explicit `context`, else the Weather Agent's own **geocoded** location (`tools/weather_tool.py`) fills the gap — more reliable than guessing from free text |
| `field` | Explicit `context["field"]` only (no reliable way to guess a plot name from text) |
| `last_disease` / `last_fertilizer` / `last_pest` / `last_soil_note` | The most recent grounded finding from that specialist agent |
| `weather_history` | A short rolling log (capped at `MEMORY_MAX_WEATHER_HISTORY`, default 5) of past forecasts checked |
| `turns` | A short rolling log (capped at `MEMORY_MAX_TURNS`, default 10) of recent questions, for light conversational continuity |

**How it flows through the pipeline** — the same shape Phase 10 already
established for cross-AGENT context, just one level up, across
QUESTIONS instead of across agents within one question:

```
Phase 10 — agents within ONE question share findings via
           context["prior_findings"]  (agent_types.format_prior_findings)
Phase 11 — questions within ONE conversation share facts via
           context["memory_summary"]  (FarmerMemory.to_prompt_block)
           + recalled context keys: crop, location, latitude,
             longitude, field
```

```python
# agent_orchestrator.py — AgentOrchestrator.handle()
memory = self.memory_store.get(session_id)              # 1. recall
context = {**memory.known_context(), **explicit_context}  # 2. merge (explicit wins)
context["memory_summary"] = memory.to_prompt_block()
...                                                        # 3. run the plan as normal
memory = self.memory_store.record_turn(                  # 4. persist what's new
    session_id, question, explicit_context, agent_results, final_report
)
```

Because `crop`/`location`/`latitude`/`longitude` are merged straight
into `context` under the SAME keys those agents already read
(`market_agent.py`'s `context.get("crop")`,
`tools/weather_tool.py`'s `context["location"]`/`latitude`/`longitude`),
**no existing agent needed to change** to benefit from recalled facts
— e.g. the Weather Agent automatically checks the remembered location
without the Planner or the farmer mentioning it again. The knowledge
agents (`knowledge_agent.py`), the Weather Agent, and the Report Agent
additionally read `context["memory_summary"]` and fold it into their
LLM prompt (ahead of `prior_findings`), so the language itself can say
"since your tomato crop had early blight recently..." instead of
treating every question as a first-ever one.

**Explicit context for THIS question always wins over older memory** —
a farmer switching crops mid-conversation (or an API caller passing an
explicit `context={"crop": "Rice"}`) is not stuck on a stale fact; it's
simply recorded as the new "last known" value going forward.

**Storage** is deliberately simple: one small JSON file per session
under `agent_config.MEMORY_DIR` (default `output/memory/`) — no
database server to stand up for a portfolio/demo project, easy to
inspect by hand, and every caller only ever talks to
`ConversationMemoryStore`, never the files directly, so the backend
can change later without touching `agent_orchestrator.py`, `api.py`,
or `main.py`.

**Inspecting / resetting memory:**

```bash
curl http://localhost:8001/api/memory/farmer-42       # what's remembered
curl -X DELETE http://localhost:8001/api/memory/farmer-42   # forget it
python main.py "..." --session farmer-42 --reset-memory     # same, from the CLI
```

In interactive CLI mode, a fresh session ID is generated automatically
each run (so a single sitting naturally shares memory across
questions) — pass `--session <id>` to resume a specific farmer's
memory from an earlier run instead.

Configurable via `agent_config.py` / environment variables:

```bash
# .env or shell
MEMORY_ENABLED=true            # set false to disable Phase 11 entirely
MEMORY_MAX_TURNS=10             # how many recent questions to keep per session
MEMORY_MAX_WEATHER_HISTORY=5    # how many past forecasts to keep per session
```

Unit tests (no LLM / network / vector DB required) live in
`tests/test_phase11_memory.py`, including a full replay of the Day 1 /
Day 2 scenario above through `AgentOrchestrator.handle()`:

```bash
python -m unittest tests.test_phase11_memory -v
```

## Explainable AI (Phase 13)

Never answer without showing evidence. Instead of handing a farmer a
bare instruction —

```
Use fertilizer X.
```

— every final answer is broken into five distinct stages a farmer (or
an auditor) can actually check:

```
Recommendation
     │
     ▼
  Reason
     │
     ▼
Supporting documents
     │
     ▼
Confidence
     │
     ▼
References
```

```bash
curl -X POST http://localhost:8001/api/agents/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "my tomato leaves have brown spots, what should I do?"}' | python -m json.tool
```

```json
{
  "explanation": {
    "recommendation": "Apply a copper-based fungicide and avoid overhead irrigation this week.",
    "reason": "Your tomato crop shows signs consistent with early blight [Disease Agent, Source 1]. Humid conditions are expected for the next 3 days [Weather Agent, Source 1], which favors fungal spread.",
    "next_steps": "- Apply copper fungicide\n- Avoid overhead irrigation\n- Monitor daily",
    "supporting_documents": [ { "agent": "disease_agent", "heading": "Tomato Disease Guide", "similarity": 0.82, "...": "..." } ],
    "confidence": {
      "level": "High",
      "score": 0.81,
      "factors": [
        "2 of 2 specialist finding(s) were grounded in real evidence",
        "retrieved knowledge-base passages matched the question with 82% average similarity",
        "the consolidated report is grounded in cited sources"
      ]
    },
    "references": [ { "n": 1, "label": "Tomato Disease Guide", "agent": "disease_agent", "similarity": 0.82 } ]
  }
}
```

**Why this isn't another LLM call.** `recommendation` / `reason` /
`next_steps` are PARSED out of the Report Agent's already-written
text — `report_agent.py`'s prompt now requires exactly that structure
(`## Recommendation` / `## Reason` / `## Recommended next steps`), so
`explainability.py`'s `split_recommendation_and_reason()` can reliably
pull them back apart (falling back to "first paragraph = recommendation,
rest = reason" for any answer that doesn't use the headers, so the UI
is never left with nothing). `confidence` is computed with a plain,
additive, fully-inspectable **formula**, deliberately NOT another LLM
call asked to grade its own answer:

```
+0.50 x (grounded specialists / specialists that ran)
+0.30 x (average retrieval similarity of cited sources — redistributed
         onto the grounded-ratio term when there's no similarity data
         at all, e.g. a purely tool-backed weather/market answer)
+0.20 if the consolidated report itself is grounded
-0.15 per specialist that errored out entirely
```

...clipped to `[0, 1]` and bucketed into **Low** (`<0.4`) / **Medium**
(`<0.7`) / **High** (`>=0.7`). An LLM can write a fluent,
confident-*sounding* recommendation whether or not it's actually
well-supported — a farmer's trust in the confidence score shouldn't
hinge on the same model also being an honest judge of its own
homework. Every score comes with the plain-language `factors` that
explain it, so the confidence number is itself explainable, not a
black box.

**Supporting documents vs. References — two different jobs.**
`supporting_documents` is the raw, combined evidence list exactly as
the specialist agents produced it (one entry per retrieved chunk / API
result / dataset hit). `references` is that same evidence,
de-duplicated and numbered (`build_references()`) into a
bibliography — the same `[Source N]` numbers the Report Agent's
`Reason` text cites inline, via
`report_agent.py`'s `_build_findings_block()`.

**How it fits the pipeline** — `agent_orchestrator.py` calls
`build_explanation()` once, right after the Report Agent produces
`final_report`, and attaches the result to
`OrchestratedAnswer.explanation`:

```python
# agent_orchestrator.py — AgentOrchestrator.handle()
final_report = self.report_agent.execute(report_request)
explanation = build_explanation(agent_results, final_report)
```

`api.py` serializes it straight through as `AskResponse.explanation`;
`main.py`'s CLI prints all five stages separately; the frontend's
`RecommendationLedger.jsx` renders them as five numbered, visually
separated stages (with the Phase 10 agent-chain graft line kept as a
secondary, collapsed "how this was produced" detail underneath) —
see `frontend/README.md`.

Unit tests (no LLM / network required) live in
`tests/test_phase13_explainability.py`: section-splitting (including
the no-headers fallback), the confidence formula's arithmetic,
reference de-duplication/numbering, and an
`AgentOrchestrator.handle()` integration test confirming a populated
`Explanation` is actually attached to the answer:

```bash
python -m unittest tests.test_phase13_explainability -v
```

## Files

| File | Responsibility |
|---|---|
| `agent_config.py` | Every Phase-7/8/9/10/11-specific setting (planner mode, weather/market/gov-search/image config, `COLLABORATION_MODE`, memory settings, API port) |
| `rag_bridge.py` | Re-exports Phase 6's embedder/vector store/retriever/LLM client/RAG pipeline |
| `agent_types.py` | Shared `AgentRequest` / `AgentResult` / `PlanDecision` / `PlanStep` dataclasses + Phase 10's `format_prior_findings()` |
| `conversation_memory.py` | Phase 11 — `FarmerMemory` + `ConversationMemoryStore`: persists/recalls per-session facts across turns |
| `explainability.py` | Phase 13 — `build_explanation()`: parses the Report Agent's text + computes confidence into the five-stage `Explanation` structure |
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
| `agent_orchestrator.py` | The main orchestrator — `AgentOrchestrator().handle(question, context, session_id)` |
| `main.py` | Interactive CLI entry point (`--image <path>` to attach a photo, `--session <id>` / `--reset-memory` for conversation memory) |
| `api.py` | FastAPI service (`/api/agents/ask`, `/api/agents`, `/api/tools`, `/api/memory/{session_id}`, `/health`) |
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
* **Memory recall reuses existing context keys, on purpose (Phase 11).**
  `FarmerMemory.known_context()` returns `crop`/`location`/`latitude`/
  `longitude`/`field` under the exact same keys `market_agent.py` and
  `tools/weather_tool.py` already read — so remembering a farmer's crop
  or location required zero changes to those agents. Only the
  richer, prose-shaped facts (previous disease/fertilizer findings,
  weather history) needed a new channel (`context["memory_summary"]`),
  and that reuses the same pattern Phase 10 established for
  `prior_findings` rather than inventing a new one.
* **Confidence is a formula, not a vibe (Phase 13).** It would be one
  extra LLM call to just ask the model "how confident are you?" —
  and it would be worthless, because the same model that wrote a
  fluent-sounding recommendation is a poor judge of whether that
  recommendation is actually well-supported. Computing `confidence`
  from `AgentResult.grounded` flags, retrieval similarity scores, and
  error counts instead means the number can't be fooled by confident
  phrasing, and the `factors` list means it never has to be trusted
  blindly either.
