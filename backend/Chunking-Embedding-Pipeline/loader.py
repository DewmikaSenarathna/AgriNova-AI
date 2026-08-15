"""
loader.py
=========
Step 1 — Document Loader

Phase 3 (Document-Processing-Pipeline) already turned every messy PDF into
a clean .txt file plus a .json metadata file. This module's only job is to
find matching (.txt, .json) pairs and hand back a simple, predictable
Python object for the chunker to work with — it does NOT touch PDFs,
OCR, or cleaning at all. Keeping this boundary strict is what lets Phase 4
be re-run/tested independently of Phase 3.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class LoadedDocument:
    """
    One fully-processed document, ready to be chunked.

    doc_id is a filesystem-safe identifier (the .txt file's stem) used to
    build stable chunk IDs later, e.g. "soil_report_2024" -> chunk IDs like
    "soil_report_2024::chunk_0007".
    """
    doc_id: str
    source_file: str
    clean_text: str
    metadata: dict = field(default_factory=dict)


def _read_metadata_for(txt_path: Path) -> dict:
    """
    Step 1a — Load the sidecar metadata.json for a given clean_text.txt,
    matched purely by filename stem (e.g. "soil_report.txt" <-> "soil_report.json").

    If no metadata file exists yet (e.g. someone dropped a .txt in manually,
    or the exporter step hasn't been wired up), we fail soft: build a
    minimal metadata dict from the filename instead of crashing the whole
    pipeline over one file.
    """
    metadata_path = config.METADATA_DIR / f"{txt_path.stem}.json"

    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read metadata for {txt_path.name}: {e}")

    logger.warning(
        f"No metadata.json found for '{txt_path.name}' — using filename-only metadata."
    )
    return {
        "source_file": str(txt_path),
        "file_name": txt_path.name,
        "title": txt_path.stem.replace("_", " ").title(),
        "document_type": "General Document",
        "language": "unknown",
    }


def load_single_document(txt_path: Path) -> Optional[LoadedDocument]:
    """
    Step 1b — Load ONE cleaned document from disk.
    Returns None (and logs a warning) instead of raising, so one bad file
    never stops the whole batch.
    """
    try:
        text = txt_path.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.error(f"Could not read {txt_path.name}: {e}")
        return None

    if not text:
        logger.warning(f"Skipping {txt_path.name} — file is empty.")
        return None

    metadata = _read_metadata_for(txt_path)

    return LoadedDocument(
        doc_id=txt_path.stem,
        source_file=metadata.get("source_file", str(txt_path)),
        clean_text=text,
        metadata=metadata,
    )


def load_all_documents(clean_text_dir: Path = None) -> List[LoadedDocument]:
    """
    Step 1c — The main function pipeline.py calls.
    Loads every processed document waiting to be chunked and embedded.
    """
    clean_text_dir = clean_text_dir or config.CLEAN_TEXT_DIR

    if not clean_text_dir.exists():
        logger.error(
            f"'{clean_text_dir}' does not exist. Run Phase 3 "
            f"(Document-Processing-Pipeline) first so there is cleaned "
            f"text for Phase 4 to chunk."
        )
        return []

    txt_files = sorted(clean_text_dir.glob("*.txt"))
    if not txt_files:
        logger.warning(f"No .txt files found in {clean_text_dir}.")
        return []

    documents = []
    for txt_path in txt_files:
        doc = load_single_document(txt_path)
        if doc:
            documents.append(doc)

    logger.info(f"Loaded {len(documents)}/{len(txt_files)} document(s) for chunking.")
    return documents
