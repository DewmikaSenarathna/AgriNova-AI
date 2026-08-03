"""
section_splitter.py
====================
Step 7 — Section Detector
"""

import logging
import re
from typing import List, Dict

import config

logger = logging.getLogger(__name__)

NUMBERED_HEADING_PATTERN = re.compile(r"^\s*(\d+(\.\d+)*)[\.\)]?\s+[A-Za-z]")
MARKDOWN_HEADING_PATTERN = re.compile(r"^\s*#{1,6}\s+")


def looks_like_heading(line: str) -> bool:
    """
    Step 7a - Decide if ONE line looks like a heading.
    """
    line = line.strip()
    if not line:
        return False

    word_count = len(line.split())
    if word_count > config.MAX_HEADING_WORD_COUNT:
        return False  # too long to be a heading, it's a normal sentence

    # Rule 1: markdown-style "# Heading"
    if MARKDOWN_HEADING_PATTERN.match(line):
        return True

    # Rule 2: numbered heading "1. Introduction" or "2.3 Soil Testing"
    if NUMBERED_HEADING_PATTERN.match(line):
        return True

    # Rule 3: ALL CAPS line
    letters_only = re.sub(r"[^A-Za-z]", "", line)
    if len(letters_only) >= 4 and line.upper() == line and letters_only.isalpha():
        return True

    # Rule 4: Title Case line without ending punctuation
    if (
        line[0].isupper()
        and not line.endswith((".", ",", ";"))
        and word_count <= 8
        and sum(1 for w in line.split() if w[:1].isupper()) >= max(1, word_count // 2)
    ):
        return True

    return False


def split_into_sections(clean_text: str) -> List[Dict]:
    """
    Step 7b — The main function pipeline.py calls.

    Splits the cleaned document text into a list of sections:
        [
            {"heading": "Introduction", "content": "..."},
            {"heading": "Soil Testing", "content": "..."},
        ]

    If NO headings are detected at all, the whole document becomes
    one single section called "Full Document" — so nothing is ever lost.
    """
    lines = clean_text.split("\n")
    sections = []
    current_heading = "Introduction"
    current_content_lines = []

    for line in lines:
        if looks_like_heading(line):
            # save the section we were building before starting a new one
            if current_content_lines:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_content_lines).strip(),
                })
            current_heading = line.strip()
            current_content_lines = []
        else:
            current_content_lines.append(line)

    # save the last section
    if current_content_lines:
        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_content_lines).strip(),
        })

    # remove any sections that ended up empty
    sections = [s for s in sections if s["content"]]

    if not sections:
        sections = [{"heading": "Full Document", "content": clean_text}]

    logger.info(f"Detected {len(sections)} section(s)")
    return sections
