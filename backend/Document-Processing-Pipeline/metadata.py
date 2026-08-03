"""
metadata.py
===========
Step 6 - Metadata Extractor

"""

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # makes language detection consistent every run
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed. Run: pip install langdetect")


# Keywords to guess a document's category from its file path.
DOCUMENT_TYPE_KEYWORDS = {
    "soil": "Soil Report",
    "crop": "Crop Guide",
    "pest": "Pest Management",
    "irrigation": "Irrigation Guide",
    "fertilizer": "Fertilizer Guide",
    "weather": "Weather Report",
    "market": "Market Report",
    "research": "Research Paper",
    "manual": "Technical Manual",
}


def guess_title(clean_text: str) -> str:
    """
    Step 6a — Guess the title.

    Simple, reliable rule: the first non-empty line of the cleaned
    document is usually the title or the main heading.
    """
    for line in clean_text.split("\n"):
        line = line.strip()
        if len(line) > 3:
            return line[:150]  # keep titles reasonably short
    return "Untitled Document"


def detect_language(clean_text: str) -> str:
    """
    Step 6b — Detect the document's language automatically.
    Returns a language code like "en" (English), "si" (Sinhala), etc.
    """
    if not LANGDETECT_AVAILABLE:
        return "unknown"

    sample = clean_text[:1000]  # first 1000 characters is enough to detect
    if not sample.strip():
        return "unknown"

    try:
        return detect(sample)
    except Exception:
        return "unknown"


def guess_document_type(file_path: Path) -> str:
    """
    Step 6c — Guess what KIND of document this is, using the folder
    name and file name (e.g. ".../soil_reports/report1.pdf" -> "Soil Report").
    """
    path_text = str(file_path).lower()

    for keyword, doc_type in DOCUMENT_TYPE_KEYWORDS.items():
        if keyword in path_text:
            return doc_type

    return "General Document"


def extract_metadata(file_path: Path, clean_text: str, page_count: int) -> Dict:
    """
    Step 6d — The main function pipeline.py calls.
    Builds one dictionary with everything we know about this document.
    """
    metadata = {
        "source_file": str(file_path),
        "file_name": file_path.name,
        "file_size_kb": round(file_path.stat().st_size / 1024, 1) if file_path.exists() else None,
        "page_count": page_count,
        "title": guess_title(clean_text),
        "language": detect_language(clean_text),
        "document_type": guess_document_type(file_path),
        "word_count": len(clean_text.split()),
        "character_count": len(clean_text),
    }
    return metadata
