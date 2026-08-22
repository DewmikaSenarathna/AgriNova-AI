"""
api.py
"""

import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from rag_pipeline import RAGPipeline
from vector_store import VectorStoreEmpty, VectorStoreUnavailable

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AgriNova AI — RAG Pipeline API",
    description="Retrieval-Augmented Generation API for the AgriNova AI agricultural assistant.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Built once and reused across requests — the embedding model and the
# ChromaDB connection are both expensive to (re)initialize per request.
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The farmer's question.")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="How many source chunks to retrieve.")


class SourceOut(BaseModel):
    chunk_id: str
    doc_id: str
    heading: str
    text: str
    similarity: float


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    sources: List[SourceOut]


@app.get("/health")
def health() -> Dict:
    """Reports whether the API, and the vector database behind it, are ready."""
    try:
        count = get_pipeline().retriever.store.count()
        return {"status": "ok", "knowledge_base_chunks": count, "llm_provider": config.LLM_PROVIDER}
    except (VectorStoreUnavailable, VectorStoreEmpty) as e:
        return {"status": "degraded", "reason": str(e), "llm_provider": config.LLM_PROVIDER}


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    The main RAG endpoint:
        Farmer asks -> Embedding -> Similarity Search -> Top-K documents
                     -> Send to LLM -> Generate answer with evidence
    """
    try:
        result = get_pipeline().answer(request.question, top_k=request.top_k)
    except Exception as e:  # last-resort safety net so the API never 500s silently
        logger.exception("Unexpected error while answering a question.")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}") from e

    return AskResponse(**result.to_dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=True)
