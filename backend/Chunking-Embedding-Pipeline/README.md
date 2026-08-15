# Chunking-Embedding-Pipeline (Phase 4)

Turns the cleaned documents produced by `Document-Processing-Pipeline`
into a searchable knowledge base:

```
100-page PDF (already cleaned in Phase 3)
        │
        ▼
   loader.py        →  loads clean_text/*.txt + metadata/*.json
        │
        ▼
   chunker.py        →  splits into ~500-word overlapping chunks
        │
        ▼
   embedder.py        →  BGE model turns each chunk into a vector
        │
        ▼
   vector_store.py    →  ChromaDB (../../vector_db)
```

## Run it

```bash
pip install -r requirements.txt
python main.py                                   # chunk + embed + store everything
python search_demo.py "aphids on rice plants"   # test semantic search
```

## Files

| File | Responsibility |
|---|---|
| `config.py` | Every tunable setting (chunk size, overlap, model name, DB paths) |
| `loader.py` | Reads Phase 3's cleaned `.txt` + `.json` output |
| `chunker.py` | Sentence-aware, section-aware ~500-word chunking with overlap |
| `embedder.py` | BGE embedding model wrapper (passages vs. queries) |
| `vector_store.py` | ChromaDB read/write/search wrapper |
| `pipeline.py` | Orchestrates load → chunk → embed → store for every document |
| `main.py` | CLI entry point |
| `search_demo.py` | Standalone script to test retrieval quality |

## Why chunks are ~500 words with overlap

* **500 words** keeps a full explanation together while staying small
  enough that several chunks fit in an LLM's context window at once.
* **Overlap (75 words)** prevents a sentence - e.g. a fertilizer dosage -
  from being sliced in half exactly at a chunk boundary.
* **Section-aware** splitting (using the headings Phase 3 detects) stops a
  single chunk from mixing two unrelated topics together.
