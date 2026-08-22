"""
rag_pipeline.py

"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from embedder import Embedder
from llm_client import LLMClient, LLMError
from prompt_builder import (
    SYSTEM_PROMPT_NO_CONTEXT,
    SYSTEM_PROMPT_WITH_CONTEXT,
    build_context_block,
    build_no_context_prompt,
    build_user_prompt,
)
from retriever import Retriever, RetrievedChunk
from vector_store import VectorStore, VectorStoreEmpty, VectorStoreUnavailable

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    """Everything the frontend needs to render an answer WITH its evidence."""
    question: str
    answer: str
    grounded: bool  # True if the answer is backed by retrieved sources
    sources: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "grounded": self.grounded,
            "sources": self.sources,
        }


class RAGPipeline:
    """
    The end-to-end Retrieval-Augmented Generation pipeline described in
    the Phase 6 diagram: retrieval grounds generation in real, citable
    agricultural documents instead of letting the LLM guess.
    """

    def __init__(self, retriever: Retriever = None, llm: LLMClient = None):
        self.retriever = retriever or Retriever(embedder=Embedder(), store=VectorStore())
        self.llm = llm or LLMClient()

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> RAGAnswer:
        """
        The main function main.py / api.py call.

        Returns a RAGAnswer even in failure/edge cases (empty question,
        empty knowledge base, no relevant chunks) rather than raising —
        callers always get something safe to show the farmer, with
        `grounded` telling them whether it's backed by real sources.
        """
        question = (question or "").strip()
        if not question:
            return RAGAnswer(
                question=question,
                answer="Please ask a question so I can help.",
                grounded=False,
                sources=[],
            )

        # Step 1 + 2 — Embedding + Similarity Search -> Top-K documents
        try:
            chunks = self.retriever.retrieve(question, top_k=top_k, min_similarity=min_similarity)
        except (VectorStoreUnavailable, VectorStoreEmpty) as e:
            return RAGAnswer(
                question=question,
                answer=(
                    "The knowledge base isn't ready yet, so I can't look up trusted "
                    f"sources right now. ({e})"
                ),
                grounded=False,
                sources=[],
            )

        # Step 3 + 4 — no relevant evidence found: fall back honestly
        if not chunks:
            logger.info(f"No sufficiently relevant chunks found for: {question!r}")
            try:
                answer_text = self.llm.generate(
                    SYSTEM_PROMPT_NO_CONTEXT, build_no_context_prompt(question)
                )
            except LLMError as e:
                answer_text = self._llm_unavailable_message(e)
            return RAGAnswer(question=question, answer=answer_text, grounded=False, sources=[])

        # Step 3 — Send to LLM: build the grounded prompt
        context_block, used_chunks = build_context_block(chunks)
        user_prompt = build_user_prompt(question, context_block)

        # Step 4 — Generate answer
        try:
            answer_text = self.llm.generate(SYSTEM_PROMPT_WITH_CONTEXT, user_prompt)
        except LLMError as e:
            answer_text = self._llm_unavailable_message(e)
            return RAGAnswer(
                question=question,
                answer=answer_text,
                grounded=False,
                sources=[c.to_dict() for c in used_chunks],
            )

        return RAGAnswer(
            question=question,
            answer=answer_text,
            grounded=True,
            sources=[c.to_dict() for c in used_chunks],
        )

    @staticmethod
    def _llm_unavailable_message(error: LLMError) -> str:
        return (
            "I found relevant information but couldn't reach the language model to "
            f"turn it into an answer just now. ({error})"
        )
