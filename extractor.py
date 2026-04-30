"""
extractor.py — Runs LLM extraction over batched chunks.

Performance improvements:
  Option 1 — Batch processing: groups N chunks into one LLM call,
             cutting round-trip overhead by ~60-70%.
  Option 3 — Capped context: vault summary and tag list are trimmed
             before injection, keeping prompt size stable as vault grows.

Both settings are tunable in config.yaml:
  chunking.batch_size          (default 3)
  processing.vault_context_limit (default 50 notes)
  processing.tag_context_limit   (default 60 tags)
"""

from typing import Optional

from config import cfg
import llm_client
from prompts import extraction_prompt, tag_normalisation_prompt
from vault_manager import (
    get_index_summary_for_llm,
    get_tag_index_for_llm,
    load_vault_index,
)


MIN_CONCEPT_LEN   = cfg["processing"]["min_concept_length"]
MAX_TAGS          = cfg["tags"]["max_per_note"]
BATCH_SIZE        = cfg["chunking"].get("batch_size", 3)
VAULT_CTX_LIMIT   = cfg["processing"].get("vault_context_limit", 50)
TAG_CTX_LIMIT     = cfg["processing"].get("tag_context_limit", 60)


# ── Main entry ────────────────────────────────────────────────────────────────

def extract(chunks: list[dict], source_label: str = "") -> list[dict]:
    """
    Extract concepts from all chunks using batched LLM calls.

    Args:
        chunks:       Output from chunker.chunk()
        source_label: Human-readable source name

    Returns:
        List of validated concept dicts
    """
    vault_summary   = _trim_vault_summary(get_index_summary_for_llm())
    existing_tags   = _trim_tags(get_tag_index_for_llm())
    existing_titles = {r["title"].lower() for r in load_vault_index()}

    all_concepts: list[dict] = []
    seen_titles:  set[str]   = set()

    batches     = _make_batches(chunks)
    total_chunks = len(chunks)

    print(f"\n[extractor] {total_chunks} chunk(s) → {len(batches)} batch(es) "
          f"(batch size: {BATCH_SIZE})")
    print(f"[extractor] Vault context: {len(existing_titles)} notes "
          f"(capped at {VAULT_CTX_LIMIT}) | "
          f"Tags capped at {TAG_CTX_LIMIT}\n")

    for i, batch in enumerate(batches, start=1):
        chunk_range = f"{batch[0]['chunk_index']}–{batch[-1]['chunk_index']}"
        print(f"  Batch {i}/{len(batches)} (chunks {chunk_range}) ...",
              end=" ", flush=True)

        concepts = _process_batch(
            batch           = batch,
            source_title    = source_label,
            vault_summary   = vault_summary,
            existing_tags   = existing_tags,
            existing_titles = existing_titles,
            source_label    = source_label,
        )

        fresh = []
        for c in concepts:
            key = c["title"].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                fresh.append(c)

        all_concepts.extend(fresh)
        print(f"→ {len(fresh)} concept(s)")

        # Update context so later batches can link to concepts found earlier
        for c in fresh:
            vault_summary   += f"\n{c['title']} | {', '.join(c['tags'])}"
            existing_titles.add(c["title"].lower())

    print(f"\n[extractor] Done — {len(all_concepts)} concept(s) total.")
    return all_concepts


# ── Batching ──────────────────────────────────────────────────────────────────

def _make_batches(chunks: list[dict]) -> list[list[dict]]:
    """Group chunks into batches of BATCH_SIZE."""
    return [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]


# ── Batch processing ──────────────────────────────────────────────────────────

def _process_batch(
    batch: list[dict],
    source_title: str,
    vault_summary: str,
    existing_tags: str,
    existing_titles: set[str],
    source_label: str,
) -> list[dict]:
    """Send one batch of chunks to the LLM and parse the response."""

    system, user = extraction_prompt(
        chunks        = batch,
        source_title  = source_title,
        existing_tags = existing_tags,
        vault_summary = vault_summary,
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
        concept = _validate(
            raw_concept     = raw_concept,
            existing_titles = existing_titles,
            existing_tags   = existing_tags,
            source_label    = source_label,
            fallback_section = batch[0]["heading"],
        )
        if concept:
            validated.append(concept)

    return validated


# ── Context trimming (Option 3) ───────────────────────────────────────────────

def _trim_vault_summary(summary: str) -> str:
    """
    Keep only the most recent VAULT_CTX_LIMIT note entries.
    Most recent = most likely to be relevant to the current source.
    """
    if not summary or summary.startswith("(vault"):
        return summary
    lines = [l for l in summary.strip().split("\n") if l.strip()]
    if len(lines) <= VAULT_CTX_LIMIT:
        return summary
    trimmed = lines[-VAULT_CTX_LIMIT:]  # keep most recent
    dropped = len(lines) - VAULT_CTX_LIMIT
    header  = f"(showing {VAULT_CTX_LIMIT} of {len(lines)} notes — {dropped} older notes omitted)\n"
    return header + "\n".join(trimmed)


def _trim_tags(tags_str: str) -> str:
    """
    Keep only TAG_CTX_LIMIT tags in the prompt.
    Tags are already sorted alphabetically — we keep all of them up to the limit.
    """
    if not tags_str or tags_str.startswith("(no"):
        return tags_str
    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
    if len(tags) <= TAG_CTX_LIMIT:
        return tags_str
    trimmed = tags[:TAG_CTX_LIMIT]
    return ", ".join(trimmed) + f"  (+ {len(tags) - TAG_CTX_LIMIT} more)"


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(
    raw_concept: dict,
    existing_titles: set[str],
    existing_tags: str,
    source_label: str,
    fallback_section: str,
) -> Optional[dict]:

    try:
        if int(raw_concept.get("significance", 0)) < 7:
            return None
    except (ValueError, TypeError):
        return None

    title   = str(raw_concept.get("title", "")).strip()
    summary = str(raw_concept.get("summary", "")).strip()

    if not title or len(title) < 3:
        return None
    if len(summary) < MIN_CONCEPT_LEN:
        return None

    examples = raw_concept.get("examples", [])
    if not isinstance(examples, list):
        examples = []
    examples = [str(e).strip() for e in examples if str(e).strip()]

    tags  = _clean_tags(raw_concept.get("tags", []), existing_tags)
    links = _clean_links(raw_concept.get("links", []), existing_titles)

    return {
        "title":          title,
        "summary":        summary,
        "examples":       examples,
        "tags":           tags,
        "links":          links,
        "source_section": str(raw_concept.get("source_section", fallback_section)).strip(),
        "source_label":   source_label,
    }


def _clean_tags(raw_tags: list, existing_tags: str) -> list[str]:
    cleaned = []
    for tag in raw_tags:
        tag = str(tag).lower().strip().strip("#").replace(" ", "-")
        tag = "-".join(tag.split("-")[:2])
        if len(tag) > 1:
            cleaned.append(tag)
    cleaned = list(dict.fromkeys(cleaned))

    needs_llm = len(cleaned) > MAX_TAGS or any(len(t) > 20 for t in cleaned)
    if needs_llm and existing_tags:
        try:
            system, user = tag_normalisation_prompt(cleaned, existing_tags)
            result = llm_client.call_json(user, system=system)
            normalised = result.get("normalised_tags", cleaned)
            if isinstance(normalised, list) and normalised:
                cleaned = [str(t).lower().strip() for t in normalised[:MAX_TAGS]]
        except Exception:
            pass

    return cleaned[:MAX_TAGS]


def _clean_links(raw_links: list, existing_titles: set[str]) -> list[dict]:
    if not isinstance(raw_links, list):
        return []
    valid = []
    for link in raw_links:
        if not isinstance(link, dict):
            continue
        target = str(link.get("to", "")).strip()
        if target and target.lower() in existing_titles:
            valid.append({
                "to":            target,
                "bidirectional": bool(link.get("bidirectional", False)),
            })
    return valid


# ── Debug ─────────────────────────────────────────────────────────────────────

def describe(concepts: list[dict]) -> None:
    print(f"\n{'═'*55}")
    print(f"  {len(concepts)} concept(s) extracted")
    print(f"{'═'*55}")
    for c in concepts:
        print(f"\n  ● {c['title']}")
        print(f"    Tags : {', '.join(c['tags']) or '—'}")
        print(f"    Links: {', '.join(lk['to'] for lk in c['links']) or '—'}")
        print(f"    {c['summary'][:110]}...")
    print(f"\n{'═'*55}\n")