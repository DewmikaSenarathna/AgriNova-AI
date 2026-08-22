"""
vector_store.py
"""

import logging
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)


class VectorStoreUnavailable(RuntimeError):
    """Raised when the chromadb package isn't installed."""


class VectorStoreEmpty(RuntimeError):
    """
    Raised when the collection exists but has zero chunks, or doesn't
    exist yet — i.e. Phase 4/5 (Chunking-Embedding-Pipeline) hasn't been
    run yet. This is a distinct, more actionable error than a generic
    "no results" — it means there is nothing to search at all.
    """


class VectorStore:
    """A thin, read-only wrapper around the project's persistent ChromaDB collection."""

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        self.persist_dir = str(persist_dir or config.VECTOR_DB_DIR)
        self.collection_name = collection_name or config.VECTOR_DB_COLLECTION_NAME
        self._client = None
        self._collection = None

    # -- Step 2a — Connect to the existing collection 
    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
        except ImportError as e:
            raise VectorStoreUnavailable(
                "chromadb is not installed. Run:\n    pip install chromadb"
            ) from e

        self._client = chromadb.PersistentClient(path=self.persist_dir)

        try:
            self._collection = self._client.get_collection(name=self.collection_name)
        except Exception as e:
            raise VectorStoreEmpty(
                f"Collection '{self.collection_name}' was not found at "
                f"'{self.persist_dir}'. Run Chunking-Embedding-Pipeline "
                f"(Phase 4 + 5) first:\n"
                f"    cd ../Chunking-Embedding-Pipeline && python main.py"
            ) from e

        count = self._collection.count()
        if count == 0:
            raise VectorStoreEmpty(
                f"Collection '{self.collection_name}' exists but is empty. "
                f"Run Chunking-Embedding-Pipeline (Phase 4 + 5) first:\n"
                f"    cd ../Chunking-Embedding-Pipeline && python main.py"
            )

        logger.info(
            f"Connected to ChromaDB collection '{self.collection_name}' "
            f"at '{self.persist_dir}' ({count} chunk(s) available for search)."
        )
        return self._collection

    # -- Step 2b — Semantic search 
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Returns the top_k chunks whose embeddings are closest to
        query_embedding, formatted as a flat list of dicts.
        """
        collection = self._get_collection()
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        results = []
        ids = raw.get("ids", [[]])[0]
        for i in range(len(ids)):
            results.append({
                "chunk_id": ids[i],
                "text": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
                "distance": raw["distances"][0][i],
            })
        return results

    def count(self) -> int:
        return self._get_collection().count()
