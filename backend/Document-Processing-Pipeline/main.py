"""
main.py
=======
  1. Look inside document_pipeline/input_pdfs/ for all your PDFs
  2. Run every one of them through all 10 steps automatically
  3. Print a clear summary report at the end (PASS/FAIL per file)
  4. Save a full report as JSON in output/reports/
  5. Leave clean .txt / .md / .json files in output/, ready for RAG
"""

import json
import logging
import sys
from datetime import datetime

import config
from pipeline import run_pipeline


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def print_summary(results):
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY — AgriNovaAI Document Processing (Phase 3)")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "PASS")
    failed = sum(1 for r in results if r.get("status") == "FAIL")
    errored = sum(1 for r in results if r.get("status") == "ERROR")

    print(f"Total PDFs processed : {total}")
    print(f"  PASS  (good quality): {passed}")
    print(f"  FAIL  (low quality) : {failed}")
    print(f"  ERROR (unreadable)  : {errored}")
    print("-" * 70)

    for r in results:
        status = r.get("status")
        name = r.get("file_name")
        if status == "PASS":
            print(f"[OK]   {name}  ({r.get('word_count')} words, {r.get('section_count')} sections)")
        elif status == "FAIL":
            print(f"[WARN] {name}  -> {', '.join(r.get('issues', []))}")
        else:
            print(f"[FAIL] {name}  -> {r.get('reason')}")

    print("=" * 70)
    print(f"Clean text saved to : {config.CLEAN_TEXT_DIR}")
    print(f"Markdown saved to   : {config.MARKDOWN_DIR}")
    print(f"Metadata saved to   : {config.METADATA_DIR}")
    print("=" * 70)
    print("Your documents are now READY FOR RAG.\n")


def save_report(results):
    report_path = config.REPORTS_DIR / f"run_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Full run report saved to: {report_path}")


def main():
    setup_logging()
    print("Starting AgriNovaAI Document Processing Pipeline (Phase 3)...")
    print(f"Reading PDFs from: {config.INPUT_DIR}\n")

    results = run_pipeline()

    if not results:
        print("No PDFs were processed. Add PDF files to the input_pdfs/ folder and try again.")
        sys.exit(0)

    print_summary(results)
    save_report(results)


if __name__ == "__main__":
    main()
