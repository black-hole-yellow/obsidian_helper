"""
preprocessors/pdf_preprocessor.py

Extracts text from PDF files using pymupdf (fitz).
Preserves chapter/section structure, page numbers, and metadata.
Handles both text-based PDFs and basic layout recovery.
"""

from pathlib import Path

try:
    import fitz  # pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def process(source: str | Path) -> dict:
    """
    Extract text from a PDF file.

    Args:
        source: Path to a .pdf file

    Returns:
        {
            "text":     str,   # full extracted text
            "sections": list,  # list of {"heading": str, "content": str, "page": int}
            "metadata": dict   # title, author, page_count, word_count
        }
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError(
            "pymupdf is not installed. Run: pip install pymupdf"
        )

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(str(path))

    # Extract PDF metadata
    meta = doc.metadata or {}
    pdf_title  = meta.get("title", "").strip() or path.stem
    pdf_author = meta.get("author", "").strip()

    pages_text = []
    sections   = []
    current_heading = "Introduction"
    current_lines   = []
    current_page    = 1

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] != 0:  # 0 = text block
                continue

            for line in block["lines"]:
                line_text = " ".join(
                    span["text"] for span in line["spans"]
                ).strip()

                if not line_text:
                    continue

                # Detect headings by font size (larger = heading)
                avg_size = sum(s["size"] for s in line["spans"]) / len(line["spans"])
                is_bold  = any(s["flags"] & 16 for s in line["spans"])  # flag 16 = bold

                if _is_heading(line_text, avg_size, is_bold):
                    # Flush current section
                    if current_lines:
                        sections.append({
                            "heading": current_heading,
                            "content": "\n".join(current_lines).strip(),
                            "page":    current_page,
                        })
                    current_heading = line_text.strip()
                    current_lines   = []
                    current_page    = page_num
                else:
                    current_lines.append(line_text)

        pages_text.append(page.get_text("text"))

    # Flush last section
    if current_lines:
        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_lines).strip(),
            "page":    current_page,
        })

    doc.close()

    full_text = "\n\n".join(pages_text)
    full_text = _clean_pdf_text(full_text)

    # Fallback if no sections detected
    if not sections:
        sections = [{"heading": "Full Content", "content": full_text, "page": 1}]

    metadata = {
        "title":         pdf_title,
        "author":        pdf_author,
        "page_count":    len(pages_text),
        "section_count": len(sections),
        "word_count":    len(full_text.split()),
        "detected_title": pdf_title,
    }

    return {"text": full_text, "sections": sections, "metadata": metadata}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_heading(text: str, font_size: float, is_bold: bool) -> bool:
    """
    Heuristic: a line is a heading if it's:
    - Short enough to be a title
    - Large font OR bold
    - Not ending with common sentence punctuation
    """
    if len(text) > 120:
        return False
    if text.endswith((".", ",", ";", ":")):
        return False
    if font_size >= 13 or (is_bold and font_size >= 11):
        return True
    return False


def _clean_pdf_text(text: str) -> str:
    """Fix common PDF extraction artifacts."""
    import re

    # Fix hyphenated line breaks (word- \nword → word)
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Remove page numbers (isolated numbers on their own line)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
