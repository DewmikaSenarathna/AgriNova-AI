"""
table_extractor.py
===================
Step 8 — Table Extractor

"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not installed. Run: pip install pdfplumber")


def table_to_markdown(table: List[List]) -> str:
    """
    Step 8a — Convert one raw table (list of rows) into a Markdown table string.
    """
    if not table or not table[0]:
        return ""

    # Replace any None cells with empty string and strip whitespace
    clean_rows = [
        [str(cell).strip() if cell is not None else "" for cell in row]
        for row in table
    ]

    header = clean_rows[0]
    body = clean_rows[1:]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in body:
        # pad short rows so the table doesn't break
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")

    return "\n".join(lines)


def extract_tables_from_pdf(pdf_path: Path) -> List[str]:
    """
    Step 8b — The main function pipeline.py calls.

    Opens the PDF with pdfplumber and pulls out every table it finds,
    across every page, as a list of Markdown table strings.
    """
    if not PDFPLUMBER_AVAILABLE:
        return []

    markdown_tables = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table in tables:
                    md = table_to_markdown(table)
                    if md:
                        markdown_tables.append(
                            f"### Table (page {page_number})\n\n{md}"
                        )
    except Exception as e:
        logger.warning(f"Could not extract tables from {pdf_path.name}: {e}")

    if markdown_tables:
        logger.info(f"Extracted {len(markdown_tables)} table(s) from {pdf_path.name}")

    return markdown_tables
