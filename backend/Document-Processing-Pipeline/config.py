"""
config.py

Step 2 — Configuration

"""

from pathlib import Path

# 1. FOLDER PATHS

# BASE_DIR = the main project folder 
BASE_DIR = Path(__file__).resolve().parent

# Where raw, messy PDFs live 
INPUT_DIR = BASE_DIR / "input_pdfs"

# Where the pipeline will save the final clean output
OUTPUT_DIR = BASE_DIR / "output"

# Sub-folders inside output/ for each type of result
CLEAN_TEXT_DIR = OUTPUT_DIR / "clean_text"      # plain .txt files
MARKDOWN_DIR = OUTPUT_DIR / "markdown"          # .md files with headings
METADATA_DIR = OUTPUT_DIR / "metadata"          # .json files with info
REPORTS_DIR = OUTPUT_DIR / "reports"            # validation reports

# Make sure these folders exist automatically 
for folder in [INPUT_DIR, OUTPUT_DIR, CLEAN_TEXT_DIR, MARKDOWN_DIR,
               METADATA_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 2. OCR SETTINGS (for scanned / image-based PDFs)
# If a PDF page has fewer than this many real text characters, treat it as a "scanned image" and send it to OCR instead.
OCR_MIN_CHARS_PER_PAGE = 20

# OCR language. "eng" = English. 
OCR_LANGUAGE = "eng"

# Image resolution used when converting PDF pages to pictures for OCR.
# Higher = better accuracy but slower. 300 is a good balance.
OCR_DPI = 300

# 3. TEXT CLEANING SETTINGS
# Lines shorter than this (like page numbers "12" or "-3-") are removed
MIN_LINE_LENGTH_TO_KEEP = 3

# If the exact same short line repeats on many pages, treat it as a header/footer
# and remove it everywhere.
REPEATED_LINE_MIN_OCCURRENCES = 3

# 4. SECTION DETECTION SETTINGS
# A line is treated as a heading if it's short and looks like a title
MAX_HEADING_WORD_COUNT = 12

# 5. VALIDATION SETTINGS (quality control)
# A document with fewer words than this is flagged as "too short / low quality"
MIN_WORD_COUNT = 50

# If more than this % of characters are "junk" (symbols, broken encoding),
# the document is flagged as low quality.
MAX_JUNK_CHAR_RATIO = 0.15

# 6. LOGGING
LOG_LEVEL = "INFO"  # change to "DEBUG" if you want more detailed logs
