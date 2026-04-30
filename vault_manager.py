"""
vault_manager.py — Manages vault_index.json and tag_index.json.

These two lightweight files are the system's memory:
  - vault_index.json  → {title, tags, file} for every note in the vault
  - tag_index.json    → flat list of all tags ever used

The LLM is shown ONLY these indexes (not full .md content),
keeping token usage minimal while enabling smart linking and tagging.
"""

import json
import os
from datetime import date
from pathlib import Path
from typing import Optional

from config import index_path, folder


# ── File paths ───────────────────────────────────────────────────────────────

VAULT_INDEX_FILE = "vault_index.json"
TAG_INDEX_FILE   = "tag_index.json"


# ── Vault Index ──────────────────────────────────────────────────────────────

def load_vault_index() -> list[dict]:
    """
    Load the vault index from disk.
    Returns a list of note records:
      [{"title": str, "tags": [str], "file": str}, ...]
    """
    path = index_path(VAULT_INDEX_FILE)
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_vault_index(index: list[dict]) -> None:
    path = index_path(VAULT_INDEX_FILE)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)


def add_note_to_index(title: str, tags: list[str], filename: str) -> None:
    """Add or update a note record in the vault index."""
    index = load_vault_index()

    # Update if already exists (re-processed source), otherwise append
    for record in index:
        if record["title"].lower() == title.lower():
            record["tags"]  = tags
            record["file"]  = filename
            save_vault_index(index)
            return

    index.append({"title": title, "tags": tags, "file": filename})
    save_vault_index(index)


def get_index_summary_for_llm() -> str:
    """
    Return a compact string representation of the vault index
    to be injected into LLM prompts.

    Format: "Title | tag1, tag2, tag3"
    One line per note.
    """
    index = load_vault_index()
    if not index:
        return "(vault is empty — no existing notes)"

    lines = [f"{r['title']} | {', '.join(r['tags'])}" for r in index]
    return "\n".join(lines)


def find_similar_titles(title: str, threshold: float = 0.0) -> list[str]:
    """
    Return existing note titles that share significant word overlap with `title`.
    Used as a fast pre-filter before embedding-based merge detection.
    """
    index  = load_vault_index()
    words  = set(title.lower().split())
    result = []

    for record in index:
        existing_words = set(record["title"].lower().split())
        overlap = len(words & existing_words) / max(len(words), 1)
        if overlap >= 0.5:  # at least 50% word overlap
            result.append(record["title"])

    return result


# ── Tag Index ────────────────────────────────────────────────────────────────

def load_tag_index() -> list[str]:
    """Return the flat list of all known tags."""
    path = index_path(TAG_INDEX_FILE)
    if not path.exists():
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("tags", [])


def save_tag_index(tags: list[str]) -> None:
    path = index_path(TAG_INDEX_FILE)
    with open(path, "w") as f:
        json.dump({"tags": sorted(tags), "last_updated": str(date.today())}, f, indent=2)


def add_tags_to_index(new_tags: list[str]) -> None:
    """Merge new tags into the global tag index (deduplicates automatically)."""
    existing = set(load_tag_index())
    merged   = existing | set(t.lower().strip("#") for t in new_tags)
    save_tag_index(sorted(merged))


def get_tag_index_for_llm() -> str:
    """
    Return known tags as a comma-separated string for LLM prompts.
    The LLM is instructed to reuse these before inventing new ones.
    """
    tags = load_tag_index()
    if not tags:
        return "(no existing tags — create new ones as needed)"
    return ", ".join(tags)


# ── Sync: rebuild index from actual vault files ──────────────────────────────

def rebuild_index_from_vault() -> None:
    """
    Walk the Concepts folder and rebuild vault_index.json from scratch.
    Useful if notes were manually edited/added/deleted in Obsidian.
    Reads only the YAML frontmatter of each .md file — fast and cheap.
    """
    import re

    concepts_folder = folder("concepts")
    index = []
    tag_pool: set[str] = set()

    for md_file in concepts_folder.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")

        # Extract YAML frontmatter between --- delimiters
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not fm_match:
            continue

        frontmatter = fm_match.group(1)

        # Extract tags line: "tags: [a, b, c]"
        tags_match = re.search(r"tags:\s*\[([^\]]+)\]", frontmatter)
        tags = []
        if tags_match:
            tags = [t.strip() for t in tags_match.group(1).split(",")]
            tag_pool.update(tags)

        title = md_file.stem  # filename without .md = note title

        index.append({"title": title, "tags": tags, "file": md_file.name})

    save_vault_index(index)
    save_tag_index(sorted(tag_pool))

    print(f"[vault_manager] Rebuilt index: {len(index)} notes, {len(tag_pool)} tags.")