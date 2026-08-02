"""
cleaner.py
==========
Step 5 - Text Cleaner

"""

import logging
import re
from collections import Counter
from typing import List

import config

logger = logging.getLogger(__name__)

# Small helper patterns
PAGE_NUMBER_PATTERN = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
MULTIPLE_SPACES_PATTERN = re.compile(r"[ \t]{2,}")
MULTIPLE_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")


def is_page_number_line(line: str) -> bool:
    """A line like '12', '- 12 -', 'Page 12' is almost certainly a page number."""
    if PAGE_NUMBER_PATTERN.match(line):
        return True
    if re.match(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", line, re.IGNORECASE):
        return True
    return False


def find_repeated_lines(all_pages: List[str]) -> set:
    """
    Step 5a — Detect headers & footers automatically.

    Idea: a real header/footer (like "AgriNovaAI Technical Manual")
    repeats on MANY pages, word-for-word. Normal body text almost
    never repeats exactly like that. So: count how many times each
    line appears across all pages; if a line appears more than
    config.REPEATED_LINE_MIN_OCCURRENCES times, it's a header/footer.
    """
    line_counter = Counter()

    for page_text in all_pages:
        # Use a set() per page so a line counts once per page,
        # even if it accidentally appears twice on the same page.
        lines_on_this_page = set(
            line.strip() for line in page_text.split("\n") if line.strip()
        )
        line_counter.update(lines_on_this_page)

    repeated_lines = {
        line for line, count in line_counter.items()
        if count >= config.REPEATED_LINE_MIN_OCCURRENCES
    }

    if repeated_lines:
        logger.info(f"Detected {len(repeated_lines)} repeated header/footer line(s)")

    return repeated_lines


def clean_single_page(page_text: str, repeated_lines: set) -> str:
    """
    Step 5b — Clean ONE page of text using the rules above.
    """
    cleaned_lines = []

    for raw_line in page_text.split("\n"):
        line = raw_line.strip()

        if not line:
            continue  # remove empty lines

        if is_page_number_line(line):
            continue  # remove page numbers

        if line in repeated_lines:
            continue  # remove repeated headers/footers

        if len(line) < config.MIN_LINE_LENGTH_TO_KEEP:
            continue  # remove tiny junk lines like "." or "-"

        # remove extra internal spaces ("soil   pH" -> "soil pH")
        line = MULTIPLE_SPACES_PATTERN.sub(" ", line)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_document(pages_text: List[str]) -> str:
    """
    Step 5c — The main function pipeline.py calls.

    Takes all pages of ONE document, removes headers/footers/page numbers,
    and joins everything into a single clean text block.
    """
    repeated_lines = find_repeated_lines(pages_text)

    cleaned_pages = [
        clean_single_page(page, repeated_lines) for page in pages_text
    ]

    full_text = "\n\n".join(p for p in cleaned_pages if p.strip())

    # Collapse 3+ blank lines down to a single blank line
    full_text = MULTIPLE_BLANK_LINES_PATTERN.sub("\n\n", full_text)

    return full_text.strip()
