"""
main.py
=======
Runs Phase 4 (chunking) and Phase 5 (embedding) back-to-back — the
easiest way to process everything the first time:

    python main.py

Once your knowledge base is up and running, you can also run each phase
on its own (e.g. from a cron job / CI pipeline):

    python run_chunking.py          # re-chunk after Phase 3 adds new PDFs
    python generate_embeddings.py   # only embeds documents that changed

`python main.py` is always safe to re-run too — Phase 5 skips any
document whose content hash hasn't changed since it was last embedded.
"""

import logging

import config
from pipeline import run_full_pipeline


def configure_logging():
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def print_summary(results):
    chunking = results.get("chunking", [])
    embedding = results.get("embedding", [])

    print(f"\n{'='*70}\nPHASE 4 — Chunking\n{'='*70}")
    if not chunking:
        print("  (nothing chunked)")
    for r in chunking:
        print(f"  {r['doc_id']:<40} {r['chunk_count']:>4} chunk(s)")

    print(f"\n{'='*70}\nPHASE 5 — Embedding\n{'='*70}")
    if not embedding:
        print("  (nothing embedded)")
    for r in embedding:
        icon = {"EMBEDDED": " NEW", "SKIPPED": "SAME", "ERROR": "FAIL"}.get(r["status"], "?")
        line = f"  [{icon}] {r['doc_id']:<38}"
        line += f" {r.get('reason', '')}" if r["status"] == "ERROR" else f" {r.get('chunk_count', 0)} chunk(s)"
        print(line)

    embedded = sum(1 for r in embedding if r["status"] == "EMBEDDED")
    skipped = sum(1 for r in embedding if r["status"] == "SKIPPED")
    print(f"\n{embedded} document(s) newly embedded, {skipped} unchanged (skipped).")
    print(f"Vector DB location: {config.VECTOR_DB_DIR}")


if __name__ == "__main__":
    configure_logging()
    print_summary(run_full_pipeline())
