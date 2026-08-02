"""
pipeline.py
===========
The Conductor - runs EVERY step, in order, for EVERY PDF.

"""

import logging
from pathlib import Path
from typing import Dict, List

import config
import pdf_reader
import ocr
import cleaner
import metadata as metadata_module
import section_splitter
import table_extractor
import validator
import exporter

logger = logging.getLogger(__name__)


def process_single_pdf(pdf_path: Path) -> Dict:
    """
    Runs ONE pdf through the entire pipeline, station by station.
    Returns a summary dictionary so main.py can print a final report.
    """
    logger.info(f"\n{'='*70}\nPROCESSING: {pdf_path.name}\n{'='*70}")

    # STEP 3 — Read raw text from the PDF
    pages_text = pdf_reader.read_pdf_pages(pdf_path)
    if not pages_text:
        logger.error(f"Skipping {pdf_path.name} — could not open/read the file.")
        return {"file_name": pdf_path.name, "status": "ERROR", "reason": "unreadable file"}

    # Check BEFORE OCR runs: does this PDF actually need OCR at all?
    pages_needing_ocr_before = [p for p in pages_text if ocr.page_needs_ocr(p)]

    # STEP 4 — Fill in OCR text for any scanned / image-only pages
    pages_text = ocr.apply_ocr_where_needed(pdf_path, pages_text)

    # STEP 5 — Clean headers, footers, page numbers, spacing
    clean_text = cleaner.clean_document(pages_text)

    if not clean_text.strip():
        if pages_needing_ocr_before and not ocr.OCR_AVAILABLE:
            # This PDF is scanned/image-only and OCR is not set up correctly.
            # Give a specific, actionable reason instead of a vague one.
            reason = (
                "This looks like a SCANNED / image-only PDF, but OCR is not "
                "fully set up on this machine. See the 'OCR IS DISABLED' "
                "warning above (or run check_ocr_setup.py) to see exactly "
                "what's missing."
            )
        elif pages_needing_ocr_before:
            reason = "This is a scanned PDF — OCR ran but could not read any text from it (image may be too low quality/blank)."
        else:
            reason = "No text extracted — the PDF may be empty or corrupted."

        logger.warning(f"{pdf_path.name} produced NO usable text after cleaning.")
        return {"file_name": pdf_path.name, "status": "ERROR", "reason": reason}

    # STEP 6 — Extract metadata (title, language, type, source info)
    doc_metadata = metadata_module.extract_metadata(pdf_path, clean_text, len(pages_text))

    # STEP 7 — Split into logical sections using detected headings
    sections = section_splitter.split_into_sections(clean_text)

    # STEP 8 — Extract any tables and convert them to Markdown
    tables = table_extractor.extract_tables_from_pdf(pdf_path)

    # STEP 9 — Validate quality (word count, junk ratio, language)
    validation = validator.validate_document(clean_text, doc_metadata)

    # STEP 10 — Save clean .txt, .md, and .json outputs
    saved_paths = exporter.export_document(
        original_path=pdf_path,
        clean_text=clean_text,
        sections=sections,
        tables=tables,
        metadata=doc_metadata,
        validation=validation,
    )

    return {
        "file_name": pdf_path.name,
        "status": validation["status"],
        "word_count": doc_metadata["word_count"],
        "section_count": len(sections),
        "table_count": len(tables),
        "issues": validation["issues"],
        "outputs": {k: str(v) for k, v in saved_paths.items()},
    }


def run_pipeline(input_folder: Path = None) -> List[Dict]:
    """
    Runs the entire pipeline for EVERY PDF found inside input_folder
    (searching sub-folders too). This is what handles "hundreds of PDFs".
    """
    input_folder = input_folder or config.INPUT_DIR
    pdf_files = pdf_reader.find_all_pdfs(input_folder)

    if not pdf_files:
        logger.warning(
            f"No PDF files found in {input_folder}. "
            f"Put your AgriNovaAI PDFs in that folder and run again."
        )
        return []

    results = []
    for pdf_path in pdf_files:
        result = process_single_pdf(pdf_path)
        results.append(result)

    return results
