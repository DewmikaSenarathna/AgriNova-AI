"""
generate_embeddings.py
========================
PHASE 5 — Generate Embeddings

    Text (chunk)
        |
        v
    Embedding Model   (BAAI/bge-base-en-v1.5)
        |
        v
    768-dimensional vector
        |
        v
    Store inside ChromaDB

    "You only do this once unless your knowledge changes."

Run this after Phase 4 has produced chunks:

    python generate_embeddings.py

WHY this step checks a manifest before embedding anything:
Embedding is the expensive part of the whole pipeline — it means loading
a transformer model and running a forward pass over every chunk. If this
script blindly re-embedded every document on every run, adding one new
PDF to a 500-document knowledge base would mean re-embedding all 501
documents every time. Instead, manifest.py remembers a content hash for
every document already embedded, so each run only pays the embedding
cost for documents that are NEW or have CHANGED since last time.
"""

import logging

import config
import manifest
from chunk_store import load_all_chunk_files
from embedder import Embedder, EmbeddingModelUnavailable
from vector_store import VectorStore, VectorStoreUnavailable

logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run() -> list:
    """
    Step A — Load every document's chunks (written by Phase 4).
    Step B — For each one, check the manifest: skip if unchanged.
    Step C — Otherwise: embed -> store in ChromaDB -> update the manifest.
    """
    documents = load_all_chunk_files()
    if not documents:
        print(
            f"Nothing to embed. Run Phase 4 first:\n    python run_chunking.py"
        )
        return []

    doc_manifest = manifest.load_manifest()
    embedder = Embedder()
    store = VectorStore()

    results = []
    for doc_id, chunks in documents:
        content_hash = manifest.compute_content_hash(chunks)

        if not manifest.needs_embedding(doc_id, content_hash, doc_manifest):
            logger.info(f"'{doc_id}' unchanged since last run — skipping (already embedded).")
            results.append({"doc_id": doc_id, "status": "SKIPPED", "chunk_count": len(chunks)})
            continue

        logger.info(f"'{doc_id}' is new or changed — generating embeddings for {len(chunks)} chunk(s)...")
        try:
            embeddings = embedder.embed_passages([c.text for c in chunks])
        except EmbeddingModelUnavailable as e:
            logger.error(str(e))
            results.append({"doc_id": doc_id, "status": "ERROR", "reason": str(e)})
            continue

        try:
            # Clear any previous (now-outdated) chunks for this document
            # before writing the fresh ones, so a document that shrank
            # doesn't leave orphaned old chunks behind in ChromaDB.
            store.delete_document(doc_id)
            store.add_chunks(chunks, embeddings)
        except VectorStoreUnavailable as e:
            logger.error(str(e))
            results.append({"doc_id": doc_id, "status": "ERROR", "reason": str(e)})
            continue

        doc_manifest[doc_id] = content_hash
        results.append({"doc_id": doc_id, "status": "EMBEDDED", "chunk_count": len(chunks)})

    manifest.save_manifest(doc_manifest)
    return results


def print_summary(results):
    print(f"\n{'='*70}\nPHASE 5 SUMMARY — Embedding Generation\n{'='*70}")
    if not results:
        return

    for r in results:
        icon = {"EMBEDDED": " NEW", "SKIPPED": "SAME", "ERROR": "FAIL"}.get(r["status"], "?")
        line = f"[{icon}] {r['doc_id']:<40}"
        if r["status"] == "ERROR":
            line += f" {r.get('reason', '')}"
        else:
            line += f" {r['chunk_count']} chunk(s)"
        print(line)

    embedded = sum(1 for r in results if r["status"] == "EMBEDDED")
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    errored = sum(1 for r in results if r["status"] == "ERROR")
    print(
        f"\n{embedded} document(s) embedded, {skipped} unchanged (skipped), "
        f"{errored} failed."
    )
    print(f"Vector DB location: {config.VECTOR_DB_DIR}")
    print(f"Embedding model: {config.EMBEDDING_MODEL_NAME} ({config.EMBEDDING_DIMENSION}-dim)")


if __name__ == "__main__":
    configure_logging()
    print_summary(run())
