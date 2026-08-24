"""
vector_db_tool.py
"""

import logging
from typing import List, Optional

from tools.base_tool import BaseTool
from tools.tool_types import ToolResult
from rag_bridge import Retriever, RetrievedChunk, VectorStoreEmpty, VectorStoreUnavailable, build_context_block

logger = logging.getLogger(__name__)


class VectorDBTool(BaseTool):
    name = "vector_database"
    description = (
        "Embeds a query and runs a similarity search against the shared ChromaDB "
        "knowledge base built by the Chunking-Embedding-Pipeline (Phase 4/5)."
    )

    def __init__(self, retriever: Optional[Retriever] = None):
        self.retriever = retriever or Retriever()

    def run(self, query: str) -> ToolResult:
        try:
            chunks: List[RetrievedChunk] = self.retriever.retrieve(query)
        except (VectorStoreUnavailable, VectorStoreEmpty) as e:
            return ToolResult(ok=False, error=str(e))

        if not chunks:
            return ToolResult(ok=False, error="No matching chunks found in the vector database.")

        context_block, used_chunks = build_context_block(chunks)
        return ToolResult(
            ok=True,
            data={"chunks": [c.to_dict() for c in used_chunks]},
            text=context_block,
            source={"source": "AgriNova AI vector database (ChromaDB)", "chunk_count": len(used_chunks)},
        )
