"""
vector_store.py
================
Step 4 — Vector Database

    embedding vectors + chunk text + metadata  ->  ChromaDB

ChromaDB stores three parallel lists per chunk, all lined up by ID:
    ids         -> unique string per chunk        ("demo::chunk_0007::a1b2c3d4")
    embeddings  -> the vector from embedder.py     ([0.0123, -0.0456, ...])
    documents   -> the original chunk TEXT         ("Apply 50kg/ha of urea...")
    metadatas   -> everything needed for citations ({"heading": "...", "source_file": "..."})

At search time (Phase 5), we embed the farmer's question with the SAME
model, ask ChromaDB for the nearest vectors, and get back the original
chunk text + metadata — ready to hand to the LLM as grounding context.
"""

import logging
from typing import Dict, List, Optional

import config
from chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStoreUnavailable(RuntimeError):
    """Raised when the chromadb package isn't installed."""


class VectorStore:
    """
    A thin wrapper around a persistent ChromaDB collection.
    "Persistent" means everything written here survives between runs —
    it's saved to disk at config.VECTOR_DB_DIR, not just kept in memory.
    """

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        self.persist_dir = str(persist_dir or config.VECTOR_DB_DIR)
        self.collection_name = collection_name or config.VECTOR_DB_COLLECTION_NAME
        self._client = None
        self._collection = None

    # Step 4a — Connect to (or create) the collection 
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
        # "hnsw:space": distance metric used for nearest-neighbour search.
        # Cosine similarity is the standard pairing with normalized
        # sentence embeddings (see embedder.py's NORMALIZE_EMBEDDINGS).
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": config.VECTOR_DB_DISTANCE_METRIC},
        )
        logger.info(
            f"Connected to ChromaDB collection '{self.collection_name}' "
            f"at '{self.persist_dir}' ({self._collection.count()} chunk(s) currently stored)"
        )
        return self._collection

    # Step 4b — Remove any old chunks for a document before re-adding 
    def delete_document(self, doc_id: str) -> None:
        """
        Deletes every chunk belonging to one document. Always call this
        before re-adding a document that may have already been processed
        (e.g. the source PDF changed and Phase 3 re-exported it) — without
        this, a shorter re-processed document would leave orphaned old
        chunks behind from its longer previous version.
        """
        collection = self._get_collection()
        collection.delete(where={"doc_id": doc_id})

    # Step 4c — Write chunks + embeddings into the database 
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        """
        The main "STORE" step. Writes chunks in batches (rather than one
        giant call) so memory usage stays flat even for a document that
        produced thousands of chunks.

        Uses `upsert` (not `add`) — since chunk IDs are deterministic
        (see chunker._make_chunk_id), re-running the pipeline on an
        unchanged document safely overwrites the same rows instead of
        erroring out on duplicate IDs.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        collection = self._get_collection()
        batch_size = config.VECTOR_DB_WRITE_BATCH_SIZE
        written = 0

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]
            batch_embeddings = embeddings[start:start + batch_size]

            collection.upsert(
                ids=[c.chunk_id for c in batch_chunks],
                embeddings=batch_embeddings,
                documents=[c.text for c in batch_chunks],
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "chunk_index": c.chunk_index,
                        "heading": c.heading,
                        "word_count": c.word_count,
                    }
                    for c in batch_chunks
                ],
            )
            written += len(batch_chunks)

        logger.info(f"Stored {written} chunk(s) in ChromaDB collection '{self.collection_name}'.")
        return written

    # Step 4d — Semantic search (used by Phase 5's RAG retriever) 
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Returns the top_k chunks whose embeddings are closest to
        query_embedding, formatted as a flat, easy-to-use list of dicts
        instead of ChromaDB's raw nested-lists response shape.
        """
        collection = self._get_collection()
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        results = []
        for i in range(len(raw["ids"][0])):
            results.append({
                "chunk_id": raw["ids"][0][i],
                "text": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
                "distance": raw["distances"][0][i],
            })
        return results

    def count(self) -> int:
        return self._get_collection().count()
