"""
chunker.py — Splits text into LLM-digestible chunks.

Uses character-based token estimation (1 token ≈ 4 chars for English).
No external tokenizer needed — fast, offline, accurate enough.
"""

from config import cfg

MAX_CHUNK_TOKENS = cfg["chunking"]["chunk_size"]
OVERLAP_TOKENS   = cfg["chunking"]["overlap"]

CHARS_PER_TOKEN  = 4  # conservative estimate for English text


def chunk(text: str, source_title: str = "") -> list[dict]:
    """
    Split text into chunks ready for LLM extraction.

    Returns list of dicts:
      {"text", "heading", "source_title", "chunk_index", "total_chunks"}
    """
    if not text or not text.strip():
        return []

    max_chars     = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN
    overlap_chars = OVERLAP_TOKENS   * CHARS_PER_TOKEN

    # Short text fits in one chunk — no splitting needed
    if len(text) <= max_chars:
        return [{
            "text":         text.strip(),
            "heading":      "Full Content",
            "source_title": source_title,
            "chunk_index":  1,
            "total_chunks": 1,
        }]

    # Split at paragraph boundaries when possible
    chunks = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    current       = []
    current_chars = 0

    for para in paragraphs:
        para_len = len(para)

        if current_chars + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            # Carry overlap into next chunk
            overlap_text  = _tail(current, overlap_chars)
            current       = [overlap_text] if overlap_text else []
            current_chars = len(overlap_text)

        current.append(para)
        current_chars += para_len

    if current:
        chunks.append("\n\n".join(current))

    total = len(chunks)
    return [
        {
            "text":         c,
            "heading":      f"Part {i + 1}",
            "source_title": source_title,
            "chunk_index":  i + 1,
            "total_chunks": total,
        }
        for i, c in enumerate(chunks)
    ]


def _tail(paragraphs: list[str], max_chars: int) -> str:
    """Return the last N chars worth of paragraphs for overlap."""
    result = []
    total  = 0
    for p in reversed(paragraphs):
        if total + len(p) > max_chars:
            break
        result.insert(0, p)
        total += len(p)
    return "\n\n".join(result)