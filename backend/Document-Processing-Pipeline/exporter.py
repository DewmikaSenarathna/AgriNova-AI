"""
exporter.py
===========
Step 10 — Save Clean Documents
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import config

logger = logging.getLogger(__name__)


def safe_filename(name: str) -> str:
    """Turn a document name into a safe filename (no weird characters)."""
    stem = Path(name).stem
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem).strip()


def save_clean_text(base_name: str, clean_text: str) -> Path:
    """Step 10a — Save the plain clean text (.txt)."""
    out_path = config.CLEAN_TEXT_DIR / f"{base_name}.txt"
    out_path.write_text(clean_text, encoding="utf-8")
    return out_path


def save_markdown(base_name: str, sections: List[Dict], tables: List[str]) -> Path:
    """Step 10b — Save a Markdown version with real headings (##) per section."""
    lines = []
    for section in sections:
        lines.append(f"## {section['heading']}")
        lines.append("")
        lines.append(section["content"])
        lines.append("")

    if tables:
        lines.append("## Tables")
        lines.append("")
        for table_md in tables:
            lines.append(table_md)
            lines.append("")

    out_path = config.MARKDOWN_DIR / f"{base_name}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def save_metadata(base_name: str, metadata: Dict, validation: Dict, section_count: int) -> Path:
    """Step 10c — Save metadata + validation results together as JSON."""
    combined = {
        **metadata,
        "section_count": section_count,
        "validation": validation,
    }
    out_path = config.METADATA_DIR / f"{base_name}.json"
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def export_document(
    original_path: Path,
    clean_text: str,
    sections: List[Dict],
    tables: List[str],
    metadata: Dict,
    validation: Dict,
) -> Dict[str, Path]:
    """
    Step 10d — The main function pipeline.py calls.
    Saves all 3 output files for ONE document and returns their paths.
    """
    base_name = safe_filename(original_path.name)

    txt_path = save_clean_text(base_name, clean_text)
    md_path = save_markdown(base_name, sections, tables)
    json_path = save_metadata(base_name, metadata, validation, len(sections))

    logger.info(f"Saved outputs for {original_path.name} -> {base_name}.[txt/md/json]")

    return {"txt": txt_path, "md": md_path, "json": json_path}
