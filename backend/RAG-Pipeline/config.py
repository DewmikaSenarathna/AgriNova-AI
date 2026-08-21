"""
config.py
=========
Step 0 — Configuration

Every "tunable number" for the RAG Pipeline (Phase 6) lives here, in ONE
place — matching the pattern already used by Document-Processing-Pipeline
and Chunking-Embedding-Pipeline.

IMPORTANT: The values under "EMBEDDING SETTINGS" and "VECTOR DATABASE
SETTINGS" below must stay IDENTICAL to the ones in
`../Chunking-Embedding-Pipeline/config.py`. Retrieval works by embedding
the farmer's question with the *same* model that embedded every document
chunk, then searching the *same* ChromaDB collection. If these drift out
of sync, similarity search will silently return garbage (wrong vector
space) or nothing (wrong collection name) — no hard error.
"""

import os
from pathlib import Path

# dotenv is optional: if it isn't installed, os.environ / real env vars
# still work fine, we just skip loading a local .env file.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 1. FOLDER PATHS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# The RAG pipeline is a READER, not a writer, of the project's vector
# database. Chunking-Embedding-Pipeline (Phase 4 + 5) is the only thing
# that ever writes to it — this file just needs to point at the exact
# same folder + collection name so it can query what's already there.
VECTOR_DB_DIR = BASE_DIR.parent.parent / "vector_db"
VECTOR_DB_COLLECTION_NAME = "agrinova_knowledge_base"

# Where this pipeline's own artifacts (query logs, run reports) go.
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = OUTPUT_DIR / "logs"

for folder in [OUTPUT_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. EMBEDDING SETTINGS  (must match Chunking-Embedding-Pipeline/config.py)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
NORMALIZE_EMBEDDINGS = True
EMBEDDING_DEVICE = "auto"  # "auto" picks GPU (cuda) if available, otherwise CPU

# ---------------------------------------------------------------------------
# 3. VECTOR DATABASE / RETRIEVAL SETTINGS
# ---------------------------------------------------------------------------

VECTOR_DB_DISTANCE_METRIC = "cosine"

# "Top 5 documents" as specified in the Phase 6 diagram.
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# Cosine similarity is 1 - distance. Below this, a retrieved chunk is
# considered "not actually relevant" and dropped rather than being fed
# to the LLM as if it were good evidence. This is what stops the RAG
# pipeline from confidently citing an unrelated document.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.35"))

# Hard cap on how many words of retrieved context get sent to the LLM,
# even if top_k chunks together would be larger. Keeps prompts fast and
# cheap regardless of how big individual chunks are.
MAX_CONTEXT_WORDS = int(os.getenv("MAX_CONTEXT_WORDS", "2500"))

# ---------------------------------------------------------------------------
# 4. LLM SETTINGS
# ---------------------------------------------------------------------------
# The project's tech stack targets open models (Llama 3 / Gemma / Mistral).
# Three interchangeable providers are supported — pick one with
# LLM_PROVIDER, no code changes needed:
#
#   "ollama"            -> free, fully local (default). Requires Ollama
#                           (https://ollama.com) running on this machine.
#   "groq"               -> free-tier cloud API serving Llama 3 / Gemma /
#                           Mistral at very high speed. Needs GROQ_API_KEY.
#   "openai_compatible"  -> any OpenAI-chat-format endpoint (OpenRouter,
#                           Together AI, LM Studio, vLLM, etc).
#
# All three are configured entirely through environment variables /
# .env — see .env.example.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

OPENAI_COMPATIBLE_BASE_URL = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")
OPENAI_COMPATIBLE_API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY", "")
OPENAI_COMPATIBLE_MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL", "")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# ---------------------------------------------------------------------------
# 5. API SERVER SETTINGS (api.py)
# ---------------------------------------------------------------------------

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
# Comma-separated list of allowed frontend origins, e.g.
# "http://localhost:3000,https://myapp.com". "*" allows any origin
# (fine for local development, tighten this before deploying).
API_CORS_ORIGINS = [o.strip() for o in os.getenv("API_CORS_ORIGINS", "*").split(",")]

# ---------------------------------------------------------------------------
# 6. LOGGING
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
