"""
chunker.py — Splits preprocessed content into LLM-digestible chunks.

Strategy:
  - Respect section boundaries first (never split mid-section if avoidable)
  - Split large sections at paragraph boundaries
  - Maintain overlap between chunks to avoid losing context at edges
  - Each chunk carries its source section heading for the LLM prompt
"""

import re
from config import cfg


CHUNK_SIZE = cfg["chunking"]["chunk_size"]   # target tokens per chunk
OVERLAP    = cfg["chunking"]["overlap"]       # overlap tokens between chunks

# Rough token estimate: 1 token ≈ 4 characters (conservative for English)
CHARS_PER_TOKEN = 4


def chunk(content: dict) -> list[dict]:
    """
    Split preprocessed content into chunks ready for LLM extraction.

    Args:
        content: Output from any preprocessor (has "sections" and "metadata")

    Returns:
        List of chunk dicts:
        [
            {
                "text":        str,   # chunk text
                "heading":     str,   # source section heading
                "chunk_index": int,   # position in sequence
                "total_chunks": int,  # total number of chunks
                "source_title": str,  # document title
            },
            ...
        ]
    """
    sections    = content["sections"]
    source_title = content["metadata"].get("detected_title", "Unknown Source")

    raw_chunks = []

    for section in sections:
        heading = section.get("heading", "")
        text    = section.get("content", "").strip()

        if not text:
            continue

        section_chunks = _split_section(text, heading)
        raw_chunks.extend(section_chunks)

    # Merge very small chunks into their neighbor (avoid sending 50-word prompts)
    merged = _merge_small_chunks(raw_chunks)

    # Add index metadata
    total = len(merged)
    result = []
    for i, chunk_data in enumerate(merged):
        result.append({
            "text":         chunk_data["text"],
            "heading":      chunk_data["heading"],
            "chunk_index":  i + 1,
            "total_chunks": total,
            "source_title": source_title,
        })

    return result


# ── Section splitting ────────────────────────────────────────────────────────

def _split_section(text: str, heading: str) -> list[dict]:
    """
    Split a single section into chunks.
    Tries paragraph boundaries first, then sentence boundaries.
    """
    max_chars   = CHUNK_SIZE * CHARS_PER_TOKEN
    overlap_chars = OVERLAP * CHARS_PER_TOKEN

    # If section fits in one chunk — return as-is
    if len(text) <= max_chars:
        return [{"text": text, "heading": heading}]

    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks  = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # Single paragraph too large — split by sentences
        if para_len > max_chars:
            sentence_chunks = _split_by_sentences(para, heading, max_chars)
            # Flush current buffer first
            if current:
                chunks.append({"text": "\n\n".join(current), "heading": heading})
                current, current_len = [], 0
            chunks.extend(sentence_chunks)
            continue

        # Adding this paragraph would exceed limit — flush and start new chunk
        if current_len + para_len > max_chars and current:
            chunk_text = "\n\n".join(current)
            chunks.append({"text": chunk_text, "heading": heading})

            # Overlap: carry last paragraph(s) into next chunk
            overlap_text = _get_overlap(current, overlap_chars)
            current     = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)

        current.append(para)
        current_len += para_len

    # Flush remaining
    if current:
        chunks.append({"text": "\n\n".join(current), "heading": heading})

    return chunks


def _split_by_sentences(text: str, heading: str, max_chars: int) -> list[dict]:
    """Last resort: split a giant paragraph by sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks    = []
    current   = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append({"text": " ".join(current), "heading": heading})
            current, current_len = [], 0

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append({"text": " ".join(current), "heading": heading})

    return chunks


def _get_overlap(paragraphs: list[str], overlap_chars: int) -> str:
    """
    Return the tail of the current chunk up to overlap_chars.
    Works backwards through paragraphs.
    """
    result  = []
    total   = 0

    for para in reversed(paragraphs):
        if total + len(para) > overlap_chars:
            break
        result.insert(0, para)
        total += len(para)

    return "\n\n".join(result)


# ── Merge small chunks ───────────────────────────────────────────────────────

def _merge_small_chunks(chunks: list[dict]) -> list[dict]:
    """
    Merge consecutive chunks that are too small to be meaningful
    (less than 15% of target chunk size).
    """
    min_chars = CHUNK_SIZE * CHARS_PER_TOKEN * 0.15
    merged    = []

    for chunk_data in chunks:
        if merged and len(chunk_data["text"]) < min_chars:
            # Absorb into previous chunk
            prev = merged[-1]
            prev["text"] = prev["text"] + "\n\n" + chunk_data["text"]
        else:
            merged.append(dict(chunk_data))

    return merged


# ── Debug / inspection ───────────────────────────────────────────────────────

def describe(chunks: list[dict]) -> None:
    """Print a summary of chunk sizes and headings — useful during development."""
    print(f"\n{'─'*60}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"{'─'*60}")
    for c in chunks:
        words = len(c["text"].split())
        est_tokens = len(c["text"]) // CHARS_PER_TOKEN
        print(f"  [{c['chunk_index']:>2}/{c['total_chunks']}] "
              f"{c['heading'][:35]:<35} "
              f"{words:>5} words  "
              f"~{est_tokens:>4} tokens")
    print(f"{'─'*60}\n")
