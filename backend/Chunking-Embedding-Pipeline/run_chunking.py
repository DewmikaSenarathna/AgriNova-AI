"""
run_chunking.py
================
PHASE 4 — Document Chunking (standalone entry point)

    100-page PDF (cleaned by Phase 3)  ->  split  ->  ~500-word chunks
                                                            |
                                                            v
                                        output/chunks/<doc_id>.json

Run this whenever Phase 3 has produced new or updated documents:

    python run_chunking.py

This step is intentionally cheap and safe to re-run as often as you like
— it only reads text and writes small JSON files, no embedding model is
loaded and nothing is sent to the vector database. Phase 5
(generate_embeddings.py) is the step that costs real time/compute, and
it decides separately (via manifest.py) whether anything actually needs
re-embedding.
"""

import logging

import config
import loader
from chunker import chunk_document
from chunk_store import save_chunks


def configure_logging():
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run() -> list:
    documents = loader.load_all_documents()
    if not documents:
        print(
            f"No documents to chunk. Run Phase 3 (Document-Processing-Pipeline) "
            f"first so '{config.CLEAN_TEXT_DIR}' has content."
        )
        return []

    results = []
    for doc in documents:
        sections = doc.metadata.get("sections")  # section-aware chunking, if available
        chunks = chunk_document(doc.doc_id, doc.clean_text, sections=sections)
        save_chunks(doc.doc_id, chunks)
        results.append({"doc_id": doc.doc_id, "chunk_count": len(chunks)})
    return results


def print_summary(results):
    print(f"\n{'='*70}\nPHASE 4 SUMMARY — Document Chunking\n{'='*70}")
    if not results:
        return
    for r in results:
        print(f"  {r['doc_id']:<40} {r['chunk_count']:>4} chunk(s)")
    total = sum(r["chunk_count"] for r in results)
    print(f"\n{len(results)} document(s) chunked, {total} chunk(s) total.")
    print(f"Chunks saved to: {config.CHUNKS_DIR}")
    print("Next: run `python generate_embeddings.py` (Phase 5).")


if __name__ == "__main__":
    configure_logging()
    print_summary(run())
