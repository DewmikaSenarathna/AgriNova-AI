"""
config.py
=========
Step 0 — Configuration
"""

from pathlib import Path

# 1. FOLDER PATHS

# BASE_DIR = this module's own folder
BASE_DIR = Path(__file__).resolve().parent

# Phase 4 reads its input from Phase 3's output (the cleaned, validated
# documents produced by the Document-Processing-Pipeline). Keeping the two
# pipelines as separate folders — but wired together like this — means each
# phase can be run, tested and re-run independently.
DOCUMENT_PIPELINE_DIR = BASE_DIR.parent / "Document-Processing-Pipeline"
CLEAN_TEXT_DIR = DOCUMENT_PIPELINE_DIR / "output" / "clean_text"
METADATA_DIR = DOCUMENT_PIPELINE_DIR / "output" / "metadata"

# Where this pipeline's own logs / reports go
OUTPUT_DIR = BASE_DIR / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"

# The project-wide vector database folder (already exists at project root).
# ChromaDB will persist its files (SQLite + HNSW index segments) here.
VECTOR_DB_DIR = BASE_DIR.parent.parent / "vector_db"

for folder in [OUTPUT_DIR, REPORTS_DIR, VECTOR_DB_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 2. CHUNKING SETTINGS

# Target chunk size, in WORDS. 500 words (~650-700 tokens) is the sweet spot
# for agricultural manuals: long enough to keep a full explanation together,
# short enough that the LLM's context window can hold several chunks at once.
CHUNK_SIZE_WORDS = 500

# How many words of the PREVIOUS chunk are repeated at the start of the NEXT
# chunk. Overlap stops a sentence (e.g. a fertilizer dosage instruction) from
# being sliced exactly in half at a chunk boundary and losing its meaning.
CHUNK_OVERLAP_WORDS = 75

# If a leftover "tail" chunk ends up smaller than this, it gets merged into
# the previous chunk instead of being stored as its own tiny, low-value chunk.
MIN_CHUNK_WORDS = 40

# 3. EMBEDDING SETTINGS

# BGE (BAAI General Embedding) — matches the project's tech stack.
# "small" = 384 dimensions, fast, runs fine on CPU. Swap to
# "BAAI/bge-base-en-v1.5" (768-dim) later for higher retrieval accuracy.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models are trained so that queries need a special instruction prefix
# but the documents/passages being stored do NOT. Getting this backwards is
# the single most common mistake when using BGE for retrieval.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# How many chunks are embedded per batch. Bigger = faster but more RAM.
EMBEDDING_BATCH_SIZE = 32

# Cosine similarity works best when embeddings are L2-normalized.
NORMALIZE_EMBEDDINGS = True

# "auto" picks GPU (cuda) automatically if available, otherwise CPU.
EMBEDDING_DEVICE = "auto"

# 4. VECTOR DATABASE SETTINGS

VECTOR_DB_COLLECTION_NAME = "agrinova_knowledge_base"

# "cosine" is the standard choice for normalized sentence-embedding search.
VECTOR_DB_DISTANCE_METRIC = "cosine"

# Number of chunks written to ChromaDB per batch (keeps memory usage flat
# even when processing hundreds of PDFs).
VECTOR_DB_WRITE_BATCH_SIZE = 100

# 5. LOGGING

LOG_LEVEL = "INFO"  # change to "DEBUG" for verbose step-by-step logs
