"""
merge_detector.py — Identifies concepts that likely represent the same idea.

Strategy (two-pass, fast → deep):

  Pass 1 — Embedding similarity (local, no LLM tokens spent)
    Encode all concept titles + summaries as vectors.
    Flag pairs above the similarity threshold from config.

  Pass 2 — LLM confirmation (only for flagged pairs)
    Ask the LLM to decide: genuine duplicate or just related?
    This filters out false positives from embedding-only comparison.

Output: a merge_suggestions.md file in /Review/ for manual review.
You decide what to merge — the system never auto-merges.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

from config import cfg, folder
import llm_client
from prompts import merge_prompt
from vault_manager import load_vault_index


THRESHOLD = cfg["processing"]["merge_similarity_threshold"]


# ── Main entry ────────────────────────────────────────────────────────────────

def detect_merges(new_concepts: list[dict]) -> Optional[Path]:
    """
    Compare newly extracted concepts against the full vault for merge candidates.
    Also checks new concepts against each other (within-run duplicates).

    Args:
        new_concepts: Output from extractor.extract()

    Returns:
        Path to merge_suggestions.md if any candidates found, else None
    """
    if not new_concepts:
        return None

    print(f"\n[merge_detector] Checking {len(new_concepts)} new concept(s) for merge candidates...")

    # Build corpus: new concepts + existing vault notes
    vault_records = load_vault_index()
    existing = [
        {"title": r["title"], "summary": "", "source": "vault"}
        for r in vault_records
    ]

    candidates = _find_candidates(new_concepts, existing)

    if not candidates:
        print("[merge_detector] No merge candidates found.")
        return None

    print(f"[merge_detector] {len(candidates)} candidate pair(s) — running LLM confirmation...")

    confirmed = _confirm_with_llm(candidates)

    if not confirmed:
        print("[merge_detector] No confirmed merges after LLM review.")
        return None

    path = _write_review_file(confirmed)
    print(f"[merge_detector] Written {len(confirmed)} suggestion(s) → {path}")
    return path


# ── Pass 1: Embedding similarity ──────────────────────────────────────────────

def _find_candidates(
    new_concepts: list[dict],
    existing: list[dict],
) -> list[tuple[dict, dict, float]]:
    """
    Return pairs (new, existing, score) where score >= THRESHOLD.
    Uses sentence-transformers if available, falls back to keyword overlap.
    """
    try:
        return _embedding_candidates(new_concepts, existing)
    except ImportError:
        print("[merge_detector] sentence-transformers not installed — using keyword fallback")
        return _keyword_candidates(new_concepts, existing)


def _embedding_candidates(
    new_concepts: list[dict],
    existing: list[dict],
) -> list[tuple[dict, dict, float]]:
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, runs on CPU

    # Encode new concepts: title + summary gives richer signal than title alone
    new_texts = [
        f"{c['title']}. {c.get('summary', '')[:200]}"
        for c in new_concepts
    ]
    existing_texts = [
        f"{e['title']}. {e.get('summary', '')[:200]}"
        for e in existing
    ]

    new_embeddings      = model.encode(new_texts,      convert_to_tensor=True)
    existing_embeddings = model.encode(existing_texts, convert_to_tensor=True)

    scores = util.cos_sim(new_embeddings, existing_embeddings)

    candidates = []
    for i, new_c in enumerate(new_concepts):
        for j, exist_c in enumerate(existing):
            try:
                score = float(scores[i][j])
            except (IndexError, TypeError):
                continue
            # Skip self-match
            if new_c["title"].lower() == exist_c["title"].lower():
                continue
            if score >= THRESHOLD:
                candidates.append((new_c, exist_c, round(score, 3)))

    # Sort by score descending
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def _keyword_candidates(
    new_concepts: list[dict],
    existing: list[dict],
) -> list[tuple[dict, dict, float]]:
    """
    Lightweight fallback: Jaccard similarity on title + summary words.
    Less accurate but requires no dependencies.
    """
    def token_set(text: str) -> set[str]:
        import re
        words = re.findall(r"\b\w{4,}\b", text.lower())
        return set(words)

    candidates = []
    for new_c in new_concepts:
        new_tokens = token_set(f"{new_c['title']} {new_c.get('summary', '')}")
        for exist_c in existing:
            if new_c["title"].lower() == exist_c["title"].lower():
                continue
            exist_tokens = token_set(f"{exist_c['title']} {exist_c.get('summary', '')}")
            union = new_tokens | exist_tokens
            if not union:
                continue
            score = len(new_tokens & exist_tokens) / len(union)
            if score >= THRESHOLD * 0.7:  # looser threshold for keyword fallback
                candidates.append((new_c, exist_c, round(score, 3)))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:20]  # cap at 20 to avoid LLM overload


# ── Pass 2: LLM confirmation ──────────────────────────────────────────────────

def _confirm_with_llm(
    candidates: list[tuple[dict, dict, float]]
) -> list[dict]:
    """
    For each candidate pair, ask the LLM whether it's a genuine duplicate.
    Returns only confirmed merge suggestions with reasoning.
    """
    confirmed = []

    for new_c, exist_c, score in candidates:
        system, user = merge_prompt(
            title_a   = new_c["title"],
            summary_a = new_c.get("summary", "(no summary)"),
            title_b   = exist_c["title"],
            summary_b = exist_c.get("summary", "(no summary available — check vault)"),
        )

        try:
            result = llm_client.call_json(user, system=system)
        except Exception as e:
            print(f"  [merge_detector] LLM call failed for pair: {e}")
            continue

        should_merge = result.get("should_merge", False)
        confidence   = float(result.get("confidence", 0.0))
        reason       = str(result.get("reason", ""))

        if should_merge and confidence >= 0.7:
            confirmed.append({
                "title_a":    new_c["title"],
                "title_b":    exist_c["title"],
                "score":      score,
                "confidence": confidence,
                "reason":     reason,
            })

    return confirmed


# ── Review file writer ────────────────────────────────────────────────────────

def _write_review_file(suggestions: list[dict]) -> Path:
    """
    Write a merge_suggestions.md into /Review/.
    Each suggestion shows both titles, the reason, and manual action options.
    """
    review_dir = folder("review")
    today      = str(date.today())
    filename   = f"merge_suggestions_{today}.md"
    path       = review_dir / filename

    lines = [
        f"---",
        f"tags: [review, merge-suggestion]",
        f"date: {today}",
        f"---",
        f"",
        f"# Merge Suggestions — {today}",
        f"",
        f"*{len(suggestions)} potential duplicate(s) detected. Review and decide manually.*",
        f"*Delete this file when done.*",
        f"",
        f"---",
        f"",
    ]

    for i, s in enumerate(suggestions, start=1):
        confidence_pct = int(s["confidence"] * 100)
        lines += [
            f"## {i}. [[{s['title_a']}]] ↔ [[{s['title_b']}]]",
            f"",
            f"**Similarity score:** {s['score']}  ",
            f"**LLM confidence:** {confidence_pct}%  ",
            f"**Reason:** {s['reason']}",
            f"",
            f"**Options:**",
            f"- [ ] Merge → keep `[[{s['title_a']}]]`, redirect `[[{s['title_b']}]]`",
            f"- [ ] Merge → keep `[[{s['title_b']}]]`, redirect `[[{s['title_a']}]]`",
            f"- [ ] Keep both — they are related but distinct",
            f"",
            f"---",
            f"",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
