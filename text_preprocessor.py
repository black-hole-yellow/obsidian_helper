"""
preprocessors/text_preprocessor.py

Handles plain text files and raw article content.
Cleans encoding issues, HTML artifacts, excessive whitespace,
and returns structured clean text with detected section markers.
"""

import re
import unicodedata
from pathlib import Path


def process(source: str | Path) -> dict:
    """
    Process a plain text file or raw string.

    Args:
        source: Path to a .txt file, OR a raw string of text

    Returns:
        {
            "text":     str,   # cleaned full text
            "sections": list,  # list of {"heading": str, "content": str}
            "metadata": dict   # detected title, word count, etc.
        }
    """
    # Load from file or use directly
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
        raw = Path(source).read_text(encoding="utf-8", errors="replace")
    else:
        raw = source

    # Clean the text
    text = _clean(raw)

    # Detect structure
    sections = _extract_sections(text)

    metadata = {
        "word_count":    len(text.split()),
        "char_count":    len(text),
        "section_count": len(sections),
        "detected_title": _detect_title(text),
    }

    return {"text": text, "sections": sections, "metadata": metadata}


# ── Cleaning ─────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    # Normalize unicode (fix smart quotes, em-dashes, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Remove HTML tags if present (articles copy-pasted from web)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs (they add noise, not knowledge)
    text = re.sub(r"https?://\S+", "", text)

    # Remove excessive special characters but keep punctuation
    text = re.sub(r"[^\w\s.,;:!?\"'()\-–—\[\]{}]", " ", text)

    # Collapse multiple spaces and blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── Section detection ────────────────────────────────────────────────────────

def _extract_sections(text: str) -> list[dict]:
    """
    Detect sections by looking for heading-like lines:
      - ALL CAPS lines
      - Lines followed by a blank line that are short (< 80 chars)
      - Markdown-style headings (# ## ###)
      - Numbered headings (1. Introduction, Chapter 3, etc.)
    """
    sections = []
    lines    = text.split("\n")
    current_heading = "Introduction"
    current_lines   = []

    heading_pattern = re.compile(
        r"^(#{1,4}\s.+|"           # Markdown headings
        r"Chapter\s+\d+.*|"        # Chapter N
        r"\d+[\.\)]\s+[A-Z].{3,}|" # 1. Heading or 1) Heading
        r"[A-Z][A-Z\s]{4,}[A-Z])$" # ALL CAPS LINE
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if heading_pattern.match(line) and len(line) < 120:
            # Save previous section
            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_lines).strip()
                })
            current_heading = _clean_heading(line)
            current_lines   = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_lines).strip()
        })

    # Fallback: no sections detected → treat whole text as one section
    if not sections:
        sections = [{"heading": "Full Content", "content": text}]

    return sections


def _clean_heading(line: str) -> str:
    """Strip markdown symbols and numbering from heading text."""
    line = re.sub(r"^#{1,4}\s+", "", line)       # remove ## prefix
    line = re.sub(r"^\d+[\.\)]\s+", "", line)    # remove 1. or 1) prefix
    return line.strip().title()


def _detect_title(text: str) -> str:
    """Best-guess at the document title: first non-empty short line."""
    for line in text.split("\n"):
        line = line.strip()
        if line and len(line) < 100:
            return _clean_heading(line)
    return "Untitled"
