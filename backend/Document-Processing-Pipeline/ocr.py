"""
ocr.py
======
Step 4 - OCR Support

"""

import logging
import shutil
from pathlib import Path
from typing import List

import config

logger = logging.getLogger(__name__)

try:
    import pytesseract
    from pdf2image import convert_from_path
    PYTHON_LIBS_INSTALLED = True
except ImportError:
    PYTHON_LIBS_INSTALLED = False

TESSERACT_BINARY_FOUND = shutil.which("tesseract") is not None
POPPLER_BINARY_FOUND = shutil.which("pdftoppm") is not None

OCR_AVAILABLE = PYTHON_LIBS_INSTALLED and TESSERACT_BINARY_FOUND and POPPLER_BINARY_FOUND

if not OCR_AVAILABLE:
    missing = []
    if not PYTHON_LIBS_INSTALLED:
        missing.append("Python packages (run: pip install pytesseract pdf2image pillow)")
    if not TESSERACT_BINARY_FOUND:
        missing.append("the 'tesseract' program (run: sudo apt-get install tesseract-ocr, or brew install tesseract)")
    if not POPPLER_BINARY_FOUND:
        missing.append("the 'poppler' program (run: sudo apt-get install poppler-utils, or brew install poppler)")

    logger.warning(
        "OCR IS DISABLED — scanned/image-only PDFs will fail with "
        "'no text extracted'. Missing: " + "; ".join(missing)
    )


def page_needs_ocr(page_text: str) -> bool:
    """
    Step 4a — Decide if a page is "scanned" (image) or "real text".

    Simple rule: if the extracted text is shorter than
    config.OCR_MIN_CHARS_PER_PAGE characters, assume it's a
    scanned image with little or no real text, so it needs OCR.
    """
    cleaned = page_text.strip()
    return len(cleaned) < config.OCR_MIN_CHARS_PER_PAGE


def ocr_pdf_pages(pdf_path: Path, page_numbers: List[int]) -> dict:
    """
    Step 4b — Run OCR on specific pages of a PDF.

    page_numbers: list like [1, 3] meaning "only OCR page 1 and page 3"
    (only OCR the pages that actually need it - this saves a lot of time
    instead of OCR-ing every page of every PDF).

    Returns: {1: "text found on page 1", 3: "text found on page 3"}
    """
    results = {}

    if not OCR_AVAILABLE:
        logger.warning(f"Skipping OCR for {pdf_path.name} — OCR libraries missing.")
        return results

    if not page_numbers:
        return results

    logger.info(f"Running OCR on {pdf_path.name} pages {page_numbers} ...")

    try:
        # Convert only the needed page range into images
        first_page = min(page_numbers)
        last_page = max(page_numbers)
        images = convert_from_path(
            str(pdf_path),
            dpi=config.OCR_DPI,
            first_page=first_page,
            last_page=last_page,
        )

        for offset, image in enumerate(images):
            page_num = first_page + offset
            if page_num in page_numbers:
                text = pytesseract.image_to_string(image, lang=config.OCR_LANGUAGE)
                results[page_num] = text

    except Exception as e:
        logger.error(f"OCR failed for {pdf_path.name}: {e}")

    return results


def apply_ocr_where_needed(pdf_path: Path, pages_text: List[str]) -> List[str]:
    """
    Step 4c — The main function pipeline.py calls.

    Takes the list of page texts from pdf_reader.py, checks each page,
    and replaces any "empty / scanned" page with OCR-extracted text.
    """
    pages_needing_ocr = [
        i + 1 for i, text in enumerate(pages_text) if page_needs_ocr(text)
    ]

    if not pages_needing_ocr:
        return pages_text  # nothing to do, all pages already had real text

    ocr_results = ocr_pdf_pages(pdf_path, pages_needing_ocr)

    updated_pages = list(pages_text)
    for page_num, ocr_text in ocr_results.items():
        updated_pages[page_num - 1] = ocr_text

    return updated_pages
