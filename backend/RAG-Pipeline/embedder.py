"""
embedder.py
===========
Step 1 — Embed the farmer's question

    "How do I treat aphids on my tomato plants?"
            |
            v
    768-dimensional vector

This mirrors `Chunking-Embedding-Pipeline/embedder.py`, trimmed to only
the QUERY side of the BGE model's asymmetric contract — the RAG pipeline
never embeds/stores passages, only farmer questions at ask-time.

IMPORTANT BGE-specific detail (see config.py's warning at the top):
Queries get an instruction prefix ("Represent this sentence for
searching relevant passages: "); the document chunks already sitting in
ChromaDB were embedded WITHOUT that prefix. Both sides must use the same
underlying model (config.EMBEDDING_MODEL_NAME) for the vectors to live
in the same space and similarity search to mean anything.
"""

import logging
from typing import List

import config

logger = logging.getLogger(__name__)


class EmbeddingModelUnavailable(RuntimeError):
    """Raised when sentence-transformers/torch isn't installed."""


class Embedder:
    """
    A thin, lazy-loading wrapper around the BGE sentence-transformers
    model, used here only to embed incoming farmer questions.

    Lazy loading matters: importing torch + loading a transformer model
    takes multiple seconds, so we only pay that cost once — the first
    time a question actually needs to be embedded, not on import.
    """

    def __init__(self, model_name: str = None, device: str = None, normalize: bool = None):
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.device = device or config.EMBEDDING_DEVICE
        self.normalize = config.NORMALIZE_EMBEDDINGS if normalize is None else normalize
        self._model = None  # loaded on first use

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

        actual_dim = self._model.get_sentence_embedding_dimension()
        expected_dim = getattr(config, "EMBEDDING_DIMENSION", None)
        if expected_dim and actual_dim != expected_dim:
            logger.warning(
                f"Configured EMBEDDING_DIMENSION ({expected_dim}) does not match "
                f"the loaded model's actual output ({actual_dim}). This usually "
                f"means EMBEDDING_MODEL_NAME here has drifted out of sync with "
                f"Chunking-Embedding-Pipeline/config.py — retrieval quality will "
                f"suffer until the two match again."
            )
        logger.info(f"Embedding model ready — output dimension: {actual_dim}")
        return self._model

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds one natural-language farmer question for semantic search.
        Applies the BGE query instruction prefix — the asymmetric half
        of the BGE contract described in the module docstring above.
        """
        model = self._load_model()
        prefixed_query = f"{config.BGE_QUERY_INSTRUCTION}{query.strip()}"
        vector = model.encode(
            [prefixed_query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return vector[0].tolist()
