"""
government_pdf_search_tool.py
==============================
PHASE 9 — "Government Agent -> Government PDF Search" tool.

The Government Agent already gets grounded answers from the shared
vector database (semantic search, same as every other KnowledgeAgent
subclass — see vector_db_tool.py). This tool adds a SECOND, literal
full-text search over the plain-text output that
`Document-Processing-Pipeline` already wrote for every processed PDF
(`output/clean_text/*.txt` + `output/metadata/*.json`), filtered to
documents whose `document_type` looks official (see
`Document-Processing-Pipeline/metadata.py`'s "Government Scheme"
keywords: government/scheme/subsidy/gazette/ministry).

Why have both:
  - Semantic search (vector DB) is great at "what's this about" even
    when the farmer's wording doesn't match the document's wording.
  - A farmer asking about a specific scheme, form, or exact eligibility
    clause often benefits from an exact keyword hit inside the ORIGINAL
    document text too — chunking (Phase 4/5) can occasionally split a
    clause away from the sentence that names the scheme. A plain-text
    search over the untouched clean_text/*.txt catches that case, and
    doesn't depend on the embedding model or ChromaDB being healthy at
    all — it's a completely independent code path, so government
    guidance stays available even if the vector database is down.

This tool never invents anything: it returns exact snippets straight
out of files a PDF the project ingested, with the source file name
attached for a citation.
"""

import json
import logging
import re
from pathlib import Path
from typing import List

import agent_config
from tools.base_tool import BaseTool
from tools.tool_types import PDFSearchHit, ToolResult

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"[A-Za-z]{3,}")


class GovernmentPDFSearchTool(BaseTool):
    name = "government_pdf_search"
    description = (
        "Full-text keyword search over official government scheme / subsidy PDFs "
        "already processed by the Document-Processing-Pipeline, independent of the "
        "vector database."
    )

    def __init__(self):
        self.clean_text_dir = agent_config.DOCUMENT_CLEAN_TEXT_DIR
        self.metadata_dir = agent_config.DOCUMENT_METADATA_DIR
        self.doc_type_labels = agent_config.GOVERNMENT_DOCUMENT_TYPE_LABELS

    # -- Step A — Which processed documents look official? ----------------------
    def _list_government_documents(self) -> List[dict]:
        if not self.metadata_dir.exists():
            return []
        docs = []
        for meta_path in sorted(self.metadata_dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Government PDF Search: could not read {meta_path.name}: {e}")
                continue
            if meta.get("document_type") in self.doc_type_labels:
                docs.append(meta)
        return docs

    # -- Step B — Keyword-search each candidate document's clean text -----------
    def run(self, query: str, top_k: int = None, snippet_chars: int = None) -> ToolResult:
        top_k = top_k or agent_config.GOVERNMENT_PDF_SEARCH_TOP_K
        snippet_chars = snippet_chars or agent_config.GOVERNMENT_PDF_SEARCH_SNIPPET_CHARS

        documents = self._list_government_documents()
        if not documents:
            return ToolResult(
                ok=False,
                error=(
                    "No government/scheme documents have been processed yet. Run the "
                    "Document-Processing-Pipeline on official PDFs first (see its README)."
                ),
            )

        keywords = [w.lower() for w in _WORD_PATTERN.findall(query or "")]
        if not keywords:
            return ToolResult(ok=False, error="No searchable keywords in the query.")

        hits: List[PDFSearchHit] = []
        for meta in documents:
            file_name = meta.get("file_name", "unknown.pdf")
            stem = Path(file_name).stem
            text_path = self.clean_text_dir / f"{stem}.txt"
            if not text_path.exists():
                continue
            try:
                text = text_path.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning(f"Government PDF Search: could not read {text_path.name}: {e}")
                continue

            text_lower = text.lower()
            score = sum(text_lower.count(kw) for kw in keywords)
            if score <= 0:
                continue

            snippet = self._best_snippet(text, text_lower, keywords, snippet_chars)
            hits.append(PDFSearchHit(
                file_name=file_name,
                title=meta.get("title", stem),
                document_type=meta.get("document_type", "Government Scheme"),
                snippet=snippet,
                score=score,
            ))

        if not hits:
            return ToolResult(
                ok=False,
                error="No official documents matched this question's keywords.",
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:top_k]

        text_block = "\n\n".join(
            f"[{h.file_name}] ({h.document_type}) \"{h.title}\":\n{h.snippet}" for h in hits
        )
        return ToolResult(
            ok=True,
            data={"hits": [h.to_dict() for h in hits]},
            text=text_block,
            source={
                "source": "Government PDF Search (full-text, Document-Processing-Pipeline output)",
                "documents_searched": len(documents),
                "matches": len(hits),
            },
        )

    @staticmethod
    def _best_snippet(text: str, text_lower: str, keywords: List[str], snippet_chars: int) -> str:
        """Returns a window of the document text centered on the first
        keyword hit, so the LLM sees real surrounding context instead of
        just a bare match count."""
        first_index = min(
            (text_lower.find(kw) for kw in keywords if text_lower.find(kw) != -1),
            default=0,
        )
        start = max(0, first_index - snippet_chars // 2)
        end = min(len(text), start + snippet_chars)
        snippet = text[start:end].strip()
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(text) else ""
        return f"{prefix}{snippet}{suffix}"
