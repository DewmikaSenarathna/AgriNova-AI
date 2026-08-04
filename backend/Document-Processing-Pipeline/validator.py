"""
validator.py
============
Step 9 — Document Validator
"""

import logging
import re
from typing import Dict

import config

logger = logging.getLogger(__name__)

# Characters we consider "normal" (letters, numbers, common punctuation, spaces)
JUNK_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\s.,;:!?()\-\'\"%/]")


def calculate_junk_ratio(text: str) -> float:
    """
    Step 9a — What fraction of the characters look like junk/garbage?
    A high ratio usually means OCR failed or the PDF encoding was broken.
    """
    if not text:
        return 1.0
    junk_chars = JUNK_CHAR_PATTERN.findall(text)
    return len(junk_chars) / max(len(text), 1)


def validate_document(clean_text: str, metadata: Dict) -> Dict:
    """
    Step 9b — The main function pipeline.py calls.

    """
    issues = []

    word_count = metadata.get("word_count", 0)
    if word_count < config.MIN_WORD_COUNT:
        issues.append(f"Too short: only {word_count} words (minimum {config.MIN_WORD_COUNT})")

    junk_ratio = calculate_junk_ratio(clean_text)
    if junk_ratio > config.MAX_JUNK_CHAR_RATIO:
        issues.append(f"Too much junk text: {junk_ratio:.0%} of characters look broken")

    if metadata.get("language") == "unknown":
        issues.append("Could not detect a language — text may be badly extracted")

    status = "FAIL" if issues else "PASS"

    if status == "FAIL":
        logger.warning(f"Validation FAILED for {metadata.get('file_name')}: {issues}")

    return {
        "file_name": metadata.get("file_name"),
        "status": status,
        "word_count": word_count,
        "junk_ratio": round(junk_ratio, 3),
        "issues": issues,
    }
