"""
chunk_store.py
===============
Persistence layer for Phase 4's output.

Phase 4 (chunking) and Phase 5 (embedding) are deliberately two SEPARATE
steps that don't have to run back-to-back:

    Phase 4:  clean text  ->  chunks  ->  saved to output/chunks/*.json
    Phase 5:  output/chunks/*.json  ->  embeddings  ->  ChromaDB

Splitting them like this means re-running Phase 4 (e.g. because a new
PDF was added) doesn't force you to re-embed documents that haven't
changed — Phase 5 decides that separately, using manifest.py.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import config
from chunker import Chunk

logger = logging.getLogger(__name__)


def save_chunks(doc_id: str, chunks: List[Chunk]) -> Path:
    """
    Step (Phase 4 output) — Writes one document's chunks to
    output/chunks/<doc_id>.json as a simple JSON array.
    """
    out_path = config.CHUNKS_DIR / f"{doc_id}.json"
    payload = [c.to_dict() for c in chunks]
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def load_chunks(doc_id: str) -> List[Chunk]:
    """Loads back the chunks for one document, previously saved by save_chunks()."""
    path = config.CHUNKS_DIR / f"{doc_id}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk.from_dict(item) for item in data]


def load_all_chunk_files() -> List[Tuple[str, List[Chunk]]]:
    """
    Step (Phase 5 input) — The main function generate_embeddings.py calls.

    Reads every chunks/*.json file written by Phase 4 and returns
    (doc_id, chunks) pairs, ready to be embedded.
    """
    if not config.CHUNKS_DIR.exists():
        logger.error(f"'{config.CHUNKS_DIR}' does not exist — run Phase 4 (run_chunking.py) first.")
        return []

    chunk_files = sorted(config.CHUNKS_DIR.glob("*.json"))
    if not chunk_files:
        logger.warning(
            f"No chunk files found in {config.CHUNKS_DIR} — "
            f"run Phase 4 (run_chunking.py) before generating embeddings."
        )
        return []

    documents = []
    for path in chunk_files:
        doc_id = path.stem
        chunks = load_chunks(doc_id)
        if chunks:
            documents.append((doc_id, chunks))

    logger.info(f"Loaded chunks for {len(documents)} document(s) from {config.CHUNKS_DIR}")
    return documents
