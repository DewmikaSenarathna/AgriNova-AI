"""
prompt_builder.py
"""

from typing import List, Tuple

import config
from retriever import RetrievedChunk

SYSTEM_PROMPT_WITH_CONTEXT = """You are AgriNova AI, a trustworthy agricultural assistant that helps \
farmers with crop diseases, fertilizers, pests, irrigation and general farming practices.

You will be given a farmer's question and a set of numbered SOURCE excerpts retrieved from a \
trusted agricultural knowledge base. Follow these rules exactly:

1. Answer using ONLY the information in the SOURCE excerpts below. Do not use outside knowledge \
to fill in gaps.
2. Every factual claim you make must reference the source it came from, like [Source 2].
3. If the sources only partially answer the question, answer the part you can and clearly state \
what is missing.
4. If the sources do not contain enough information to answer confidently, say so plainly instead \
of guessing, and suggest the farmer consult a local agricultural extension officer.
5. Write in clear, plain language a farmer can act on. Prefer short paragraphs or a short list of \
concrete steps over long dense prose. Avoid unnecessary jargon.
6. Never invent dosages, chemical names, or figures that are not present in the sources.
"""

SYSTEM_PROMPT_NO_CONTEXT = """You are AgriNova AI, a trustworthy agricultural assistant.

No relevant information was found in the trusted agricultural knowledge base for this question. \
Tell the farmer plainly that you don't have grounded information on this in your knowledge base \
right now, so they know not to fully rely on what follows. Then, if you can, offer brief, clearly \
general (non-source-backed) guidance and recommend they confirm it with a local agricultural \
extension officer before acting on it. Keep it short and honest — do not present a guess as if it \
were established, sourced fact.
"""


def _truncate_to_word_limit(text: str, words_remaining: int) -> Tuple[str, int]:
    """Trims `text` to at most `words_remaining` words. Returns (trimmed_text, words_used)."""
    words = text.split()
    if len(words) <= words_remaining:
        return text, len(words)
    trimmed = " ".join(words[:words_remaining])
    return trimmed + " [...]", words_remaining


def build_context_block(chunks: List[RetrievedChunk]) -> Tuple[str, List[RetrievedChunk]]:
    """
    Step 3a — Formats retrieved chunks into a numbered SOURCE block the
    LLM can cite by number (e.g. "[Source 3]"), and returns the subset
    of chunks actually included (in case the word budget cut some off).

    Chunks are numbered in the order they're passed in — `rag_pipeline.py`
    passes them already ranked by similarity (best first), so lower
    source numbers are always the strongest matches.
    """
    lines = []
    used_chunks = []
    words_remaining = config.MAX_CONTEXT_WORDS

    for i, chunk in enumerate(chunks, start=1):
        if words_remaining <= 0:
            break
        text, words_used = _truncate_to_word_limit(chunk.text, words_remaining)
        words_remaining -= words_used

        lines.append(
            f"[Source {i}] (document: {chunk.doc_id} | section: {chunk.heading} | "
            f"similarity: {chunk.similarity:.2f})\n{text}"
        )
        used_chunks.append(chunk)

    context_block = "\n\n".join(lines)
    return context_block, used_chunks


def build_user_prompt(question: str, context_block: str) -> str:
    """Step 3b — The final message sent to the LLM when context IS available."""
    return (
        f"SOURCES:\n{context_block}\n\n"
        f"---\n\n"
        f"FARMER'S QUESTION: {question.strip()}\n\n"
        f"Answer the farmer's question using only the SOURCES above, citing them as [Source N]."
    )


def build_no_context_prompt(question: str) -> str:
    """Step 3c — Fallback message when retrieval found nothing relevant enough to use."""
    return (
        f"FARMER'S QUESTION: {question.strip()}\n\n"
        f"No matching sources were found in the knowledge base for this question."
    )
