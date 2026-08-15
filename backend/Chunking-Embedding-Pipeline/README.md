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
