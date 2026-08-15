"""
search_demo.py
===============
A quick sanity check for everything Phase 4 built — run it after main.py
to prove the vector database actually works end-to-end:

    python search_demo.py "how do I treat aphids on tomato plants"

This mirrors exactly what Phase 5's RAG retriever will do later: embed
the farmer's question, search ChromaDB, and print back the most relevant
chunks with their source document for citation.
"""

import sys

from embedder import Embedder
from vector_store import VectorStore


def run_search(query: str, top_k: int = 5):
    embedder = Embedder()
    store = VectorStore()

    if store.count() == 0:
        print("The vector database is empty — run `python main.py` first.")
        return

    query_vector = embedder.embed_query(query)
    results = store.search(query_vector, top_k=top_k)

    print(f"\nTop {len(results)} result(s) for: \"{query}\"\n{'='*70}")
    for rank, r in enumerate(results, start=1):
        similarity = 1 - r["distance"]  # cosine distance -> similarity
        print(
            f"\n#{rank}  similarity={similarity:.3f}  "
            f"doc='{r['metadata']['doc_id']}'  section='{r['metadata']['heading']}'"
        )
        preview = r["text"][:280].replace("\n", " ")
        print(f"    {preview}{'...' if len(r['text']) > 280 else ''}")


if __name__ == "__main__":
    user_query = " ".join(sys.argv[1:]) or "how often should I irrigate rice fields"
    run_search(user_query)
