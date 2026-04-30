"""
preprocessors/__init__.py — Single entry point for all source types.

All preprocessors return the same dict shape:
  {
      "text":     str,
      "sections": list[dict],
      "metadata": dict
  }
"""

from pathlib import Path


def load(source: str, source_type: str) -> dict:
    source_type = source_type.lower().strip()

    if source_type == "text":
        from text_preprocessor import process
        return process(source)

    elif source_type == "pdf":
        from pdf_preprocessor import process
        return process(source)

    elif source_type == "youtube":
        from youtube_preprocessor import process
        return process(source)

    else:
        raise ValueError(
            f"Unknown source type: '{source_type}'. "
            f"Choose from: text, pdf, youtube"
        )