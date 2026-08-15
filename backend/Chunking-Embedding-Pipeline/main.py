"""
main.py
=======
Run this file to execute Phase 4 end-to-end:

    python main.py

It will:
  1. Load every cleaned document Phase 3 produced
  2. Split each one into ~500-word overlapping chunks
  3. Embed every chunk with the BGE model
  4. Store everything in the project's ChromaDB vector database

Prints a final summary table so you can see exactly what happened.
"""

import logging

import config
from pipeline import run_pipeline


def configure_logging():
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def print_summary(results):
    print(f"\n{'='*70}\nPHASE 4 SUMMARY — Document Chunking & Embedding\n{'='*70}")

    if not results:
        print("No documents were processed.")
        return

    total_chunks = 0
    for r in results:
        status_icon = {"SUCCESS": "OK", "ERROR": "FAIL", "SKIPPED": "SKIP"}.get(r["status"], "?")
        line = f"[{status_icon:>4}] {r['doc_id']:<40}"
        if r["status"] == "SUCCESS":
            line += f" {r['chunk_count']:>4} chunks  (avg {r['avg_chunk_words']} words/chunk)"
            total_chunks += r["chunk_count"]
        elif r["status"] == "ERROR":
            line += f" {r.get('reason', 'unknown error')}"
        print(line)

    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    print(f"\n{success_count}/{len(results)} document(s) processed successfully.")
    print(f"{total_chunks} chunk(s) embedded and stored in this run.")
    print(f"Vector DB location: {config.VECTOR_DB_DIR}")


if __name__ == "__main__":
    configure_logging()
    results = run_pipeline()
    print_summary(results)
