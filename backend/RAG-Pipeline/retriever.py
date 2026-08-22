"""
retriever.py
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import config
from embedder import Embedder, EmbeddingModelUnavailable
from vector_store import VectorStore, VectorStoreEmpty, VectorStoreUnavailable

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """One retrieved piece of evidence, ready to be cited."""
    chunk_id: str
    doc_id: str
    heading: str
    text: str
    similarity: float

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "heading": self.heading,
            "text": self.text,
            "similarity": round(self.similarity, 4),
        }


class Retriever:
    """
    Turns a farmer's natural-language question into a ranked, relevance-
    filtered list of RetrievedChunk objects.
    """

    def __init__(self, embedder: Embedder = None, store: VectorStore = None):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        where: Optional[Dict] = None,
    ) -> List[RetrievedChunk]:
        """
        The main function rag_pipeline.py calls.

        Step 1 — Embedding: turns `question` into a vector using the
                 SAME BGE model the knowledge base was built with.
        Step 2 — Similarity Search: asks ChromaDB for the `top_k`
                 nearest chunk vectors.
        Step 3 — Relevance filter: drops any chunk whose similarity is
                 below `min_similarity` — a low score means "closest
                 match we had", not "actually relevant", and feeding an
                 unrelated chunk to the LLM as if it were evidence is
                 exactly how RAG systems produce confident-sounding but
                 wrong answers.

        Returns an empty list if the question isn't a good match for
        anything in the knowledge base — `rag_pipeline.py` treats that
        as a signal to fall back to an honest "I don't know" response
        instead of guessing.
        """
        question = (question or "").strip()
        if not question:
            return []

        top_k = top_k or config.RETRIEVAL_TOP_K
        min_similarity = config.MIN_SIMILARITY if min_similarity is None else min_similarity

        try:
            query_vector = self.embedder.embed_query(question)
        except EmbeddingModelUnavailable as e:
            logger.error(str(e))
            raise

        try:
            raw_results = self.store.search(query_vector, top_k=top_k, where=where)
        except (VectorStoreUnavailable, VectorStoreEmpty) as e:
            logger.error(str(e))
            raise

        retrieved = []
        for r in raw_results:
            # ChromaDB returns cosine DISTANCE (0 = identical). Similarity
            # is the more intuitive number to reason about and to show
            # a farmer/dev in logs, so we convert once, here.
            similarity = 1.0 - r["distance"]
            if similarity < min_similarity:
                continue
            metadata = r.get("metadata") or {}
            retrieved.append(
                RetrievedChunk(
                    chunk_id=r["chunk_id"],
                    doc_id=metadata.get("doc_id", "unknown"),
                    heading=metadata.get("heading", "Untitled Section"),
                    text=r["text"],
                    similarity=similarity,
                )
            )

        logger.info(
            f"Retrieved {len(raw_results)} candidate chunk(s), "
            f"{len(retrieved)} passed the min_similarity={min_similarity} filter."
        )
        return retrieved
