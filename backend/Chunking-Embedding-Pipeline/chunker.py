"""
chunker.py
==========
Step 2 — Document Chunker

    100-page PDF text  ->  split  ->  ~500-word chunks (with overlap)

WHY chunk at all?
An LLM can't (and shouldn't) read a 100-page manual just to answer one
question — it's slow, expensive, and the answer gets lost in the noise.
Instead we cut every document into small, self-contained pieces small
enough to embed and retrieve individually, so later (Phase 5 RAG) we only
feed the LLM the 3-5 chunks that actually answer the farmer's question.

WHY not just cut every 500 words blindly?
A naive "word[0:500], word[500:1000], ..." split will happily cut a
sentence — or a fertilizer dosage table — exactly in half. This chunker
instead:
    1. Splits text into SENTENCES (never breaks mid-sentence)
    2. Packs sentences into a chunk until ~500 words is reached
    3. Carries the last ~75 words of context into the START of the next
       chunk (the "overlap"), so meaning never gets lost at a boundary
    4. Chunks section-by-section (using the headings Phase 3 detected),
       so a chunk never mixes unrelated topics like "Soil pH" and
       "Pest Control" together
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

# Splits on '.', '!', '?' followed by whitespace + a capital letter/quote,
# without breaking on common abbreviations like "e.g." or "Dr." or "5.5 pH".
_ABBREVIATIONS = r"(?<!\be\.g)(?<!\bi\.e)(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bvs)"
SENTENCE_BOUNDARY_PATTERN = re.compile(
    _ABBREVIATIONS + r"(?<=[.!?])\s+(?=[A-Z0-9\"'\u201c])"
)


@dataclass
class Chunk:
    """One retrieval-ready unit of text, plus everything needed to trace
    it back to its exact origin (used for source citations later in RAG)."""
    chunk_id: str
    doc_id: str
    chunk_index: int
    heading: str
    text: str
    word_count: int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.text.split())


# Step 2a — Sentence splitting

def split_into_sentences(text: str) -> List[str]:
    """
    Breaks one block of text into a list of sentences. Falls back to
    treating the whole block as one "sentence" if the text has no
    punctuation at all (e.g. a table row), so nothing is ever dropped.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = SENTENCE_BOUNDARY_PATTERN.split(text)
    return [s.strip() for s in sentences if s.strip()]


# Step 2b — Pack sentences into ~500-word windows, with overlap

def _pack_sentences(
    sentences: List[str],
    chunk_size_words: int,
    overlap_words: int,
) -> List[str]:
    """
    The sliding-window packer. Greedily adds sentences to the current
    chunk until adding the next one would exceed chunk_size_words, then
    starts a new chunk — seeded with the trailing sentences from the
    chunk we just closed, so context carries across the boundary.
    """
    chunks: List[str] = []
    current_sentences: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # A single sentence longer than the whole target chunk size (rare,
        # e.g. a giant run-on table row) is kept as its own chunk rather
        # than being torn apart.
        if sentence_word_count >= chunk_size_words:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences, current_word_count = [], 0
            chunks.append(sentence)
            continue

        if current_word_count + sentence_word_count > chunk_size_words and current_sentences:
            # This chunk is full — close it out.
            chunks.append(" ".join(current_sentences))

            # Build the overlap: walk backwards through the sentences we
            # just used, keeping whole sentences until we've gathered
            # ~overlap_words worth of trailing context.
            carry_over, carry_words = [], 0
            for s in reversed(current_sentences):
                w = len(s.split())
                if carry_words + w > overlap_words and carry_over:
                    break
                carry_over.insert(0, s)
                carry_words += w

            current_sentences = carry_over
            current_word_count = carry_words

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    # Merge a too-small trailing chunk into its predecessor instead of
    # shipping a near-empty, low-signal chunk to the vector database.
    if len(chunks) >= 2 and len(chunks[-1].split()) < config.MIN_CHUNK_WORDS:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    return chunks


# Step 2c — Chunk one document (section-aware)

def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    A short, stable, filesystem/DB-safe ID. Deterministic (same doc_id +
    index always produces the same ID) so re-running the pipeline on an
    unchanged document overwrites (upserts) the same vector DB rows
    instead of creating duplicates.
    """
    raw = f"{doc_id}::chunk_{chunk_index:04d}"
    short_hash = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"{raw}::{short_hash}"


def chunk_document(
    doc_id: str,
    clean_text: str,
    sections: Optional[List[Dict]] = None,
    chunk_size_words: int = None,
    overlap_words: int = None,
) -> List[Chunk]:
    """
    Step 2d — The main function pipeline.py calls.

    Turns one cleaned document into an ordered list of Chunk objects.

    If `sections` (from Phase 3's section_splitter — [{"heading","content"}])
    is supplied, chunking runs PER SECTION so chunk boundaries respect topic
    boundaries. Otherwise the whole document is chunked as a single section.
    """
    chunk_size_words = chunk_size_words or config.CHUNK_SIZE_WORDS
    overlap_words = overlap_words or config.CHUNK_OVERLAP_WORDS

    if not sections:
        sections = [{"heading": "Full Document", "content": clean_text}]

    chunks: List[Chunk] = []
    global_index = 0

    for section in sections:
        heading = section.get("heading", "Untitled Section")
        content = section.get("content", "")
        if not content.strip():
            continue

        sentences = split_into_sentences(content)
        if not sentences:
            continue

        text_windows = _pack_sentences(sentences, chunk_size_words, overlap_words)

        for window_text in text_windows:
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(doc_id, global_index),
                    doc_id=doc_id,
                    chunk_index=global_index,
                    heading=heading,
                    text=window_text,
                )
            )
            global_index += 1

    logger.info(
        f"'{doc_id}': {len(sections)} section(s) -> {len(chunks)} chunk(s) "
        f"(target {chunk_size_words} words, {overlap_words} word overlap)"
    )
    return chunks
