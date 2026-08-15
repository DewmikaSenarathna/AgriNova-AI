"""
embedder.py
===========
Step 3 — Embedding Generator

    ~500-word chunks  ->  embedding vectors (lists of numbers)

An embedding is a fixed-length list of numbers that represents the
MEANING of a piece of text — chunks about "irrigation scheduling" end up
mathematically close to each other in this number-space, even if they
never share an exact word. That's what makes semantic search possible.

This module wraps a BGE (BAAI General Embedding) model via the
`sentence-transformers` library, matching the project's tech stack.

IMPORTANT BGE-specific detail:
BGE models were trained asymmetrically — the instructions are DIFFERENT
depending on which side of the search you're embedding:
    * Documents/passages being stored           -> embed as-is, NO prefix
    * A user's search query at retrieval time    -> prefix with an
      instruction string ("Represent this sentence for searching
      relevant passages: ")
Mixing these up is a common bug that quietly makes search results worse,
so this module exposes two clearly separate methods instead of one.
"""

import logging
from typing import List

import config

logger = logging.getLogger(__name__)


class EmbeddingModelUnavailable(RuntimeError):
    """Raised when sentence-transformers/torch isn't installed."""


class Embedder:
    """
    A thin, lazy-loading wrapper around the BGE sentence-transformers model.

    Lazy loading matters here: importing torch + downloading/loading a
    transformer model is slow (multiple seconds) and memory-heavy, so we
    only pay that cost once — the first time an embedding is actually
    requested — not just from importing this file.
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None,
        normalize: bool = None,
    ):
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.device = device or config.EMBEDDING_DEVICE
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        self.normalize = config.NORMALIZE_EMBEDDINGS if normalize is None else normalize
        self._model = None  # loaded on first use

    # Step 3a — Load the model (once) 
    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingModelUnavailable(
                "sentence-transformers is not installed. Run:\n"
                "    pip install sentence-transformers\n"
                "(this will also pull in torch, which is required)."
            ) from e

        resolved_device = self.device
        if resolved_device in (None, "auto"):
            try:
                import torch
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                resolved_device = "cpu"

        logger.info(f"Loading embedding model '{self.model_name}' on '{resolved_device}'...")
        self._model = SentenceTransformer(self.model_name, device=resolved_device)
        logger.info(
            f"Embedding model ready — output dimension: "
            f"{self._model.get_sentence_embedding_dimension()}"
        )
        return self._model

    # Step 3b — Embed CHUNKS being stored (no instruction prefix)
    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of document chunks for storage in the vector
        database. This is what pipeline.py calls for every document.
        """
        if not texts:
            return []

        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > self.batch_size,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    # Step 3c — Embed a user's search QUERY (with instruction prefix) 
    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single natural-language question for semantic search
        (used later in Phase 5's RAG retrieval step). Applies the BGE
        query instruction prefix — this is the asymmetric half of the
        BGE contract described in the module docstring above.
        """
        model = self._load_model()
        prefixed_query = f"{config.BGE_QUERY_INSTRUCTION}{query.strip()}"
        vector = model.encode(
            [prefixed_query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return vector[0].tolist()
