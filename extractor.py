"""
extractor.py — Runs LLM extraction over chunked content.

This is the core processing layer. It:
  1. Loads vault context (index + tags) — lightweight, no full file reads
  2. Sends each chunk to the LLM with a structured prompt
  3. Parses and validates the JSON response
  4. Normalises tags against the existing tag vocabulary
  5. Resolves links against existing vault notes
  6. Returns a clean list of concept dicts ready for note_builder
"""

import json
import time
from typing import Optional

from config import cfg
import llm_client
from prompts import extraction_prompt, tag_normalisation_prompt
from vault_manager import (
    get_index_summary_for_llm,
    get_tag_index_for_llm,
    load_vault_index,
)


MIN_CONCEPT_LEN = cfg["processing"]["min_concept_length"]
MAX_TAGS        = cfg["tags"]["max_per_note"]


# ── Main entry ───────────────────────────────────────────────────────────────

def extract(chunks: list[dict], source_label: str = "") -> list[dict]:
    """
    Run LLM extraction over all chunks from a single source.

    Args:
        chunks:       Output from chunker.chunk()
        source_label: Human-readable source name (e.g. "Book: Deep Work")

    Returns:
        List of validated concept dicts:
        [
            {
                "title":          str,
                "summary":        str,
                "examples":       list[str],
                "tags":           list[str],
                "links":          list[dict],  # {"to": str, "bidirectional": bool}
                "source_section": str,
                "source_label":   str,
            },
            ...
        ]
    """
    # Load vault context once — shared across all chunk calls
    vault_summary  = get_index_summary_for_llm()
    existing_tags  = get_tag_index_for_llm()
    existing_titles = {r["title"].lower() for r in load_vault_index()}

    all_concepts: list[dict] = []
    seen_titles:  set[str]   = set()  # deduplicate within this run

    total = len(chunks)
    print(f"\n[extractor] Processing {total} chunk(s) from: {source_label or 'source'}")
    print(f"[extractor] Vault context: {len(existing_titles)} existing notes\n")

    for chunk in chunks:
        idx = chunk["chunk_index"]
        print(f"  Chunk {idx}/{total} — {chunk['heading']} ...", end=" ", flush=True)

        concepts = _process_chunk(
            chunk          = chunk,
            vault_summary  = vault_summary,
            existing_tags  = existing_tags,
            existing_titles = existing_titles,
            source_label   = source_label,
        )

        # Deduplicate against concepts already found in this run
        fresh = []
        for c in concepts:
            key = c["title"].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                fresh.append(c)

        all_concepts.extend(fresh)
        print(f"→ {len(fresh)} concept(s) found")

        # Update vault_summary mid-run so later chunks can link to
        # concepts discovered earlier in the same source
        for c in fresh:
            vault_summary += f"\n{c['title']} | {', '.join(c['tags'])}"
            existing_titles.add(c["title"].lower())

    print(f"\n[extractor] Done. Total concepts extracted: {len(all_concepts)}")
    return all_concepts


# ── Chunk processing ─────────────────────────────────────────────────────────

def _process_chunk(
    chunk: dict,
    vault_summary: str,
    existing_tags: str,
    existing_titles: set[str],
    source_label: str,
) -> list[dict]:
    """Extract concepts from a single chunk."""

    system, user = extraction_prompt(
        chunk_text      = chunk["text"],
        section_heading = chunk["heading"],
        source_title    = chunk["source_title"],
        chunk_index     = chunk["chunk_index"],
        total_chunks    = chunk["total_chunks"],
        existing_tags   = existing_tags,
        vault_summary   = vault_summary,
    )

    try:
        raw = llm_client.call_json(user, system=system)
    except (ValueError, RuntimeError) as e:
        print(f"\n  [extractor] LLM call failed: {e}")
        return []

    concepts = raw.get("concepts", [])
    if not isinstance(concepts, list):
        return []

    validated = []
    for raw_concept in concepts:
        concept = _validate_and_clean(
            raw_concept     = raw_concept,
            existing_titles = existing_titles,
            existing_tags   = existing_tags,
            source_label    = source_label,
            source_section  = chunk["heading"],
        )
        if concept:
            validated.append(concept)

    return validated


# ── Validation & cleaning ────────────────────────────────────────────────────

def _validate_and_clean(
    raw_concept: dict,
    existing_titles: set[str],
    existing_tags: str,
    source_label: str,
    source_section: str,
) -> Optional[dict]:
    """
    Validate a single raw concept dict from the LLM.
    Returns None if the concept fails minimum quality checks.
    """

    # ── Required fields ──────────────────────────────────────────────────────
    title   = str(raw_concept.get("title", "")).strip()
    summary = str(raw_concept.get("summary", "")).strip()

    if not title or len(title) < 3:
        return None
    if len(summary) < MIN_CONCEPT_LEN:
        return None

    # ── Examples ─────────────────────────────────────────────────────────────
    examples = raw_concept.get("examples", [])
    if not isinstance(examples, list):
        examples = []
    examples = [str(e).strip() for e in examples if str(e).strip()]

    # ── Tags ─────────────────────────────────────────────────────────────────
    raw_tags = raw_concept.get("tags", [])
    if not isinstance(raw_tags, list):
        raw_tags = []

    tags = _normalise_tags(raw_tags, existing_tags)

    # ── Links ─────────────────────────────────────────────────────────────────
    raw_links = raw_concept.get("links", [])
    links     = _validate_links(raw_links, existing_titles)

    return {
        "title":          title,
        "summary":        summary,
        "examples":       examples,
        "tags":           tags,
        "links":          links,
        "source_section": raw_concept.get("source_section", source_section),
        "source_label":   source_label,
    }


def _normalise_tags(raw_tags: list, existing_tags: str) -> list[str]:
    """
    Clean tags: lowercase, hyphenated, max 2 words, max MAX_TAGS total.
    Does a quick local pass first; LLM normalisation runs only if tags
    look messy (contains long tags or many unknowns).
    """
    cleaned = []
    for tag in raw_tags:
        tag = str(tag).lower().strip().strip("#").replace(" ", "-")
        words = tag.split("-")
        # Enforce 2-word max
        tag = "-".join(words[:2])
        if tag and len(tag) > 1:
            cleaned.append(tag)

    cleaned = list(dict.fromkeys(cleaned))  # deduplicate, preserve order

    # Only call LLM for tag normalisation if we have > MAX_TAGS or long tags
    needs_llm = len(cleaned) > MAX_TAGS or any(len(t) > 20 for t in cleaned)

    if needs_llm and existing_tags:
        try:
            system, user = tag_normalisation_prompt(cleaned, existing_tags)
            result = llm_client.call_json(user, system=system)
            normalised = result.get("normalised_tags", cleaned)
            if isinstance(normalised, list) and normalised:
                cleaned = [str(t).lower().strip() for t in normalised[:MAX_TAGS]]
        except Exception:
            pass  # fall back to local cleaning if LLM fails

    return cleaned[:MAX_TAGS]


def _validate_links(raw_links: list, existing_titles: set[str]) -> list[dict]:
    """
    Validate link targets against the known vault + in-run titles.
    Drop links to non-existent notes (LLM hallucinations).
    """
    if not isinstance(raw_links, list):
        return []

    valid = []
    for link in raw_links:
        if not isinstance(link, dict):
            continue

        target        = str(link.get("to", "")).strip()
        bidirectional = bool(link.get("bidirectional", False))

        if not target:
            continue

        # Accept if target exists in vault (case-insensitive)
        if target.lower() in existing_titles:
            valid.append({"to": target, "bidirectional": bidirectional})

    return valid


# ── Debug helpers ─────────────────────────────────────────────────────────────

def describe(concepts: list[dict]) -> None:
    """Print a readable summary of extracted concepts."""
    print(f"\n{'═'*60}")
    print(f"  Extracted {len(concepts)} concept(s)")
    print(f"{'═'*60}")
    for c in concepts:
        print(f"\n  ● {c['title']}")
        print(f"    Tags:  {', '.join(c['tags']) or '—'}")
        links = [lk['to'] for lk in c['links']]
        print(f"    Links: {', '.join(links) or '—'}")
        print(f"    {c['summary'][:120]}...")
    print(f"\n{'═'*60}\n")
