# RAG-Pipeline (Phase 6)

Turns the knowledge base built by `Chunking-Embedding-Pipeline` (Phase 4 +
5) into an actual question-answering assistant, grounded in retrieved
evidence instead of guesses.

```
Farmer asks
     │
     ▼
┌──────────────── Step 1 — Embedding ─────────────────┐
│  embedder.py     → question → BGE query vector      │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────── Step 2 — Similarity Search ──────────┐
│  vector_store.py → nearest chunks in ChromaDB        │
│  retriever.py    → filters out low-similarity matches│
└──────────────────────────────────────────────────────┘
     │
     ▼
              Top 5 documents (relevance-filtered)
     │
     ▼
┌──────────────── Step 3 — Send to LLM ─────────────────┐
│  prompt_builder.py → numbered, source-labelled context│
└───────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────── Step 4 — Generate answer ─────────────┐
│  llm_client.py  → Ollama / Groq / OpenAI-compatible   │
│  rag_pipeline.py → orchestrates all 4 steps, returns  │
│                    { answer, sources, grounded }      │
└───────────────────────────────────────────────────────┘
```

## Without RAG vs. With RAG

| | Without RAG | With RAG (this module) |
|---|---|---|
| Input | Question | Question |
| Step | LLM guesses from memory | Question → relevant documents → LLM reasons over them |
| Output | Answer, no evidence, may hallucinate | Answer with evidence, cited as `[Source N]` |
| Unknown topics | Confidently makes something up | Says so plainly, suggests an extension officer |

Run `python eval_demo.py "<question>"` to see both side by side.

## Prerequisites

1. **Document-Processing-Pipeline** has processed at least one PDF (Phase 3).
2. **Chunking-Embedding-Pipeline** has chunked and embedded it into
   `../../vector_db` (Phase 4 + 5):
   ```bash
   cd ../Chunking-Embedding-Pipeline
   python main.py
   ```
3. An **LLM backend** is reachable. Default is a local [Ollama](https://ollama.com)
   install (free, no API key):
   ```bash
   ollama pull llama3
   ```
   Prefer a hosted option instead? Copy `.env.example` to `.env` and set
   `LLM_PROVIDER=groq` with a free key from https://console.groq.com/keys
   (fastest to get running with zero local setup), or point
   `LLM_PROVIDER=openai_compatible` at any OpenAI-chat-format endpoint
   (OpenRouter, Together AI, LM Studio, vLLM, ...).

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env    # optional — defaults work with local Ollama

# Interactive CLI:
python main.py

# One-off question:
python main.py "how do I treat aphids on tomato plants"

# Compare without-RAG vs with-RAG for one question:
python eval_demo.py "what fertilizer should I use for rice"

# Serve it as an API for the frontend:
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### API example

```bash
curl -X POST http://localhost:8000/api/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "how do I treat aphids on tomato plants", "top_k": 5}'
```

```json
{
  "question": "how do I treat aphids on tomato plants",
  "answer": "Based on [Source 1] and [Source 2], ...",
  "grounded": true,
  "sources": [
    {"chunk_id": "...", "doc_id": "pest_guide", "heading": "Aphid Control", "text": "...", "similarity": 0.81}
  ]
}
```

`GET /health` reports whether the API and the vector database behind it
are ready — useful for the frontend to show a "knowledge base not ready
yet" state instead of a confusing error.

## Files

| File | Step | Responsibility |
|---|---|---|
| `config.py` | all | Every tunable setting (top_k, similarity threshold, LLM provider, paths) |
| `embedder.py` | 1 | Embeds the farmer's question with the same BGE model used in Phase 5 |
| `vector_store.py` | 2 | Read-only ChromaDB search wrapper (same collection Phase 5 wrote to) |
| `retriever.py` | 1+2 | Orchestrates embed → search → drops chunks below `MIN_SIMILARITY` |
| `prompt_builder.py` | 3 | Builds the numbered, citable source context + system prompts |
| `llm_client.py` | 4 | Pluggable LLM backend: Ollama / Groq / any OpenAI-compatible endpoint |
| `rag_pipeline.py` | 1-4 | The main orchestrator — `RAGPipeline().answer(question)` |
| `main.py` | — | Interactive CLI entry point |
| `api.py` | — | FastAPI service (`/api/ask`, `/health`) for the frontend to call |
| `eval_demo.py` | — | Side-by-side "without RAG" vs "with RAG" demo |

## Design notes

* **Relevance filtering, not just top-K.** `MIN_SIMILARITY` (default
  `0.35`) drops chunks that were merely the *closest available* match,
  not an actually relevant one — this is what stops the pipeline from
  confidently citing an unrelated document when the knowledge base
  genuinely doesn't cover a topic.
* **Honest fallback.** If nothing relevant is retrieved, the pipeline
  still calls the LLM, but with an explicit instruction to say so and
  point the farmer to a human expert — never a silent hallucination
  dressed up as a sourced answer.
* **Provider-agnostic LLM layer.** Swapping `LLM_PROVIDER` in `.env` is
  the only change needed to move from local Ollama to a hosted API — no
  code changes, matching the project's "Llama 3 / Gemma / Mistral" stack.
* **Read-only w.r.t. the vector DB.** This pipeline never writes to
  ChromaDB — only `Chunking-Embedding-Pipeline` does. If it can't find
  the collection, it fails with an actionable message telling you to run
  that pipeline first, rather than a confusing empty-results response.
