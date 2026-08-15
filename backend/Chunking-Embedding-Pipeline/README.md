# Chunking-Embedding-Pipeline (Phase 4 + Phase 5)

Turns the cleaned documents produced by `Document-Processing-Pipeline`
(Phase 3) into a searchable knowledge base, as two separate, independently
runnable stages:

```
Phase 3 output (clean_text/*.txt + metadata/*.json)
        │
        ▼
┌───────────────────── PHASE 4 — Chunking ─────────────────────┐
│  loader.py     → loads clean text + metadata                 │
│  chunker.py    → splits into ~500-word overlapping chunks     │
│  chunk_store.py→ saves chunks to output/chunks/<doc_id>.json  │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────── PHASE 5 — Embedding ────────────────────┐
│  manifest.py   → skip documents unchanged since last run      │
│  embedder.py   → BGE model → 768-dimensional vector per chunk │
│  vector_store.py → stores vectors in ChromaDB (../../vector_db)│
└─────────────────────────────────────────────────────────────┘
```

## Why two separate phases?

Chunking is cheap (just text splitting). Embedding is expensive (it loads
a transformer model and runs inference on every chunk). Keeping them
separate — with Phase 4's output persisted to disk as plain JSON — means:

* You can re-run Phase 4 any time Phase 3 adds/updates a PDF, without
  touching the vector database.
* Phase 5 only pays the embedding cost for documents that are **new or
  changed**, via a content-hash manifest (`output/embedding_manifest.json`).
  Re-running `generate_embeddings.py` on an unchanged knowledge base does
  no work at all — exactly the "you only do this once unless your
  knowledge changes" behaviour.

## Run it

```bash
pip install -r requirements.txt

# First time (or whenever you want to do both steps at once):
python main.py

# Day-to-day, once your knowledge base exists:
python run_chunking.py          # after Phase 3 processes new/updated PDFs
python generate_embeddings.py   # embeds only what actually changed

# Test retrieval:
python search_demo.py "aphids on tomato plants"
```

## Files

| File | Phase | Responsibility |
|---|---|---|
| `config.py` | both | Every tunable setting (chunk size, model name, paths) |
| `loader.py` | 4 | Reads Phase 3's cleaned `.txt` + `.json` output |
| `chunker.py` | 4 | Sentence-aware, section-aware ~500-word chunking with overlap |
| `chunk_store.py` | 4 → 5 | Persists chunks to `output/chunks/*.json` and reloads them |
| `run_chunking.py` | 4 | Standalone Phase 4 entry point |
| `manifest.py` | 5 | Content-hash cache so unchanged documents are skipped |
| `embedder.py` | 5 | BGE embedding model wrapper (768-dim; passages vs. queries) |
| `vector_store.py` | 5 | ChromaDB read/write/search wrapper |
| `generate_embeddings.py` | 5 | Standalone Phase 5 entry point |
| `pipeline.py` | both | Convenience wrapper chaining Phase 4 → Phase 5 |
| `main.py` | both | CLI entry point (runs `pipeline.py`) |
| `search_demo.py` | — | Test script for semantic search against ChromaDB |

## Design notes

* **500-word chunks, 75-word overlap** — long enough to keep an
  explanation together, short enough for several chunks to fit an LLM's
  context window; overlap stops a sentence being cut exactly at a
  boundary.
* **Section-aware** — chunking runs per detected heading (from Phase 3),
  so one chunk never mixes two unrelated topics.
* **768-dimensional vectors** — `BAAI/bge-base-en-v1.5`. Query embeddings
  get a BGE-specific instruction prefix; passage/chunk embeddings do not
  (`embedder.py` exposes this as two distinct methods so it can't be
  mixed up by accident).
* **Idempotent storage** — chunk IDs are deterministic, and Phase 5
  deletes a document's old vectors before writing fresh ones, so
  re-processing a changed document never leaves stale chunks behind.
