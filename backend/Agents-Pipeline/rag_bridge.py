"""
rag_bridge.py
"""

import sys
from pathlib import Path

_AGENTS_PIPELINE_DIR = Path(__file__).resolve().parent
_RAG_PIPELINE_DIR = _AGENTS_PIPELINE_DIR.parent / "RAG-Pipeline"

if not _RAG_PIPELINE_DIR.exists():
    raise ImportError(
        f"Expected to find the Phase 6 RAG-Pipeline at '{_RAG_PIPELINE_DIR}' "
        f"(Agents-Pipeline is built on top of it), but that folder is missing."
    )

if str(_RAG_PIPELINE_DIR) not in sys.path:
    # Insert at the front so RAG-Pipeline's own bare imports (e.g. the
    # `import config` inside retriever.py) resolve to ITS config.py,
    # even if some other "config"-named module got imported earlier.
    sys.path.insert(0, str(_RAG_PIPELINE_DIR))

import config as rag_config  # noqa: E402  (Phase 6 config.py, explicitly aliased)
from embedder import Embedder  # noqa: E402
from vector_store import (  # noqa: E402
    VectorStore,
    VectorStoreEmpty,
    VectorStoreUnavailable,
)
from retriever import Retriever, RetrievedChunk  # noqa: E402
from llm_client import LLMClient, LLMError  # noqa: E402
from prompt_builder import build_context_block  # noqa: E402
from rag_pipeline import RAGPipeline, RAGAnswer  # noqa: E402

__all__ = [
    "rag_config",
    "Embedder",
    "VectorStore",
    "VectorStoreEmpty",
    "VectorStoreUnavailable",
    "Retriever",
    "RetrievedChunk",
    "LLMClient",
    "LLMError",
    "build_context_block",
    "RAGPipeline",
    "RAGAnswer",
]


def make_shared_retriever() -> Retriever:
    """
    Builds ONE Retriever (embedding model + ChromaDB connection) that every
    knowledge-backed agent can share for the lifetime of the process,
    instead of each agent separately loading its own copy of the BGE
    model. Callers (agent_orchestrator.py / api.py) build this once and
    pass it to every KnowledgeAgent subclass.
    """
    return Retriever(embedder=Embedder(), store=VectorStore())


def make_shared_llm() -> LLMClient:
    """Same idea as make_shared_retriever(), but for the LLM client."""
    return LLMClient()
