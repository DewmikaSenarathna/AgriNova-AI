"""
pdf_reader.py

Step 3 — PDF Reader

"""

import logging
from pathlib import Path
from typing import List, Dict

import pypdf

logger = logging.getLogger(__name__)


def find_all_pdfs(root_folder: Path) -> List[Path]:
    """
    Step 3a — Find every PDF file inside root_folder, including sub-folders.

    "rglob" means "recursive glob" -> search this folder AND every folder inside it.
    Returns a simple list of file paths.
    """
    pdf_files = sorted(root_folder.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF file(s) inside {root_folder}")
    return pdf_files


def read_pdf_pages(pdf_path: Path) -> List[str]:
    """
    Step 3b — Open ONE pdf and return a list of strings,
    where each string is the raw text of one page.

    """
    pages_text = []
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"  Could not read page {page_number} of {pdf_path.name}: {e}")
                text = ""
            pages_text.append(text)
    except Exception as e:
        logger.error(f"Failed to open {pdf_path.name}: {e}")
        return []

    return pages_text


def read_all_pdfs(root_folder: Path) -> Dict[str, List[str]]:
    """
    Step 3c — The main function other files will call.

    Finds every PDF in root_folder, reads each one and returns a dictionary:

    This lets pipeline.py process "hundreds of PDFs" in one loop later.
    """
    pdf_paths = find_all_pdfs(root_folder)
    all_documents = {}

    for pdf_path in pdf_paths:
        logger.info(f"Reading: {pdf_path.name}")
        pages = read_pdf_pages(pdf_path)
        # store the FULL path as key text so later steps know exactly
        # where the file came from (needed for metadata "source").
        all_documents[str(pdf_path)] = pages

    return all_documents


# Quick manual test run "python pdf_reader.py" to test this file alone
if __name__ == "__main__":
    import config
    logging.basicConfig(level=logging.INFO)
    docs = read_all_pdfs(config.INPUT_DIR)
    for name, pages in docs.items():
        print(f"{name}: {len(pages)} page(s)")
