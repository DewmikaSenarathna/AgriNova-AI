"""
pipeline.py
===========
The Conductor for Phase 4.

    100-page PDF (already cleaned by Phase 3)
        |
        v
    LOAD    (loader.py)     -> cleaned text + metadata
        |
        v
    SPLIT   (chunker.py)    -> ~500-word overlapping chunks
        |
        v
    EMBED   (embedder.py)   -> one vector per chunk
        |
        v
    STORE   (vector_store.py) -> ChromaDB, ready for semantic search
"""

import logging
from typing import Dict, List

import config
import loader
from chunker import chunk_document
from embedder import Embedder, EmbeddingModelUnavailable
from vector_store import VectorStore, VectorStoreUnavailable

logger = logging.getLogger(__name__)


def process_single_document(
    doc: "loader.LoadedDocument",
    embedder: Embedder,
    store: VectorStore,
) -> Dict:
    """
    Runs ONE loaded document through split -> embed -> store.
    Returns a summary dict so main.py can print a final report.
    """
    logger.info(f"\n{'-'*70}\nCHUNKING & EMBEDDING: {doc.doc_id}\n{'-'*70}")

    # STEP 2 — Split into ~500-word chunks (section-aware if available)
    sections = doc.metadata.get("sections")  # optional, if Phase 3 exported them
    chunks = chunk_document(doc.doc_id, doc.clean_text, sections=sections)

    if not chunks:
        logger.warning(f"'{doc.doc_id}' produced no chunks — skipping.")
        return {"doc_id": doc.doc_id, "status": "SKIPPED", "chunk_count": 0}

    # STEP 3 — Generate one embedding vector per chunk
    try:
        embeddings = embedder.embed_passages([c.text for c in chunks])
    except EmbeddingModelUnavailable as e:
        logger.error(str(e))
        return {"doc_id": doc.doc_id, "status": "ERROR", "reason": str(e)}

    # STEP 4 — Remove any stale chunks from a previous run, then store fresh
    try:
        store.delete_document(doc.doc_id)
        written = store.add_chunks(chunks, embeddings)
    except VectorStoreUnavailable as e:
        logger.error(str(e))
        return {"doc_id": doc.doc_id, "status": "ERROR", "reason": str(e)}

    avg_words = sum(c.word_count for c in chunks) / len(chunks)

    return {
        "doc_id": doc.doc_id,
        "status": "SUCCESS",
        "chunk_count": len(chunks),
        "chunks_written": written,
        "avg_chunk_words": round(avg_words, 1),
    }


def run_pipeline() -> List[Dict]:
    """
    The main function main.py calls. Loads every processed document from
    Phase 3's output and runs each one through split -> embed -> store.
    """
    documents = loader.load_all_documents()
    if not documents:
        logger.warning(
            "Nothing to chunk. Make sure Phase 3 "
            "(Document-Processing-Pipeline) has produced output in "
            f"'{config.CLEAN_TEXT_DIR}' first."
        )
        return []

    # One embedder + one store, reused across every document, so the
    # (relatively expensive) model load and DB connection each happen once.
    embedder = Embedder()
    store = VectorStore()

    results = [process_single_document(doc, embedder, store) for doc in documents]

    logger.info(f"\nVector database now holds {store.count()} chunk(s) in total.")
    return results
