"""
manifest.py
============
Change detection for Phase 5.

    "You only do this once unless your knowledge changes."

Generating embeddings costs real time (and, with a paid API-based model,
real money) — so Phase 5 should NEVER blindly re-embed every document on
every run. This module gives it a memory: a small JSON file mapping
    doc_id -> a hash of that document's chunk content
Before embedding a document, generate_embeddings.py checks: "does this
doc_id already have a matching hash in the manifest?" If yes, the
document's content is byte-for-byte the same as last time — skip it.
If the hash differs (or doc_id is new), the content has changed —
re-embed it and update the manifest.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

import config
from chunker import Chunk

logger = logging.getLogger(__name__)


def compute_content_hash(chunks: List[Chunk]) -> str:
    """
    Builds one fingerprint for a document's entire chunk set. Any change
    to the chunk text (different PDF content, different chunk_size_words
    setting, an extra chunk, etc.) produces a different hash.
    """
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk.text.encode("utf-8"))
    return hasher.hexdigest()


def load_manifest() -> Dict[str, str]:
    """Loads the {doc_id: content_hash} manifest, or an empty one if this is the first run."""
    if not config.EMBEDDING_MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(config.EMBEDDING_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Embedding manifest was corrupted — starting a fresh one.")
        return {}


def save_manifest(manifest: Dict[str, str]) -> None:
    config.EMBEDDING_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def needs_embedding(doc_id: str, content_hash: str, manifest: Dict[str, str]) -> bool:
    """
    True if this document is new, or has changed since it was last
    embedded. False if it's already in the vector database, unchanged.
    """
    return manifest.get(doc_id) != content_hash
