"""
pipeline.py
===========
Convenience wrapper that runs Phase 4 (chunking) immediately followed by
Phase 5 (embedding) in one command — what main.py uses for a first-time,
"just process everything" run.

For normal day-to-day use once your knowledge base is up and running,
prefer running the two phases separately:

    python run_chunking.py          # cheap, safe to re-run anytime
    python generate_embeddings.py   # only re-embeds new/changed documents

Since Phase 5 always checks manifest.py before embedding anything,
calling run_full_pipeline() repeatedly is still safe and cheap — it will
naturally skip any document that hasn't changed since last time.
"""

import logging
from typing import Dict, List

import run_chunking
import generate_embeddings

logger = logging.getLogger(__name__)


def run_full_pipeline() -> Dict[str, List[Dict]]:
    logger.info("=== PHASE 4: Chunking ===")
    chunking_results = run_chunking.run()

    logger.info("=== PHASE 5: Embedding ===")
    embedding_results = generate_embeddings.run()

    return {
        "chunking": chunking_results,
        "embedding": embedding_results,
    }
