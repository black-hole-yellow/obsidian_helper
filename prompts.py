"""
prompts.py — All LLM prompt templates in one place.

Keeping prompts separate makes them easy to tune without
touching extraction logic. Each function returns a
(system, user) tuple ready to pass to llm_client.
"""

# ── Batch extraction prompt ───────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a knowledge extraction engine for a personal knowledge base.
Your job is to read source material and extract only the most valuable, high-level concepts.

RULES:
- QUALITY OVER QUANTITY: Extract a MAXIMUM of 3 concepts per batch. Only the best ideas.
- THE SIGNIFICANCE FILTER: Assign a "significance" score 1-10 per concept. Only extract 7+.
- Each concept must be self-contained and understandable on its own.
- Titles: 2-6 words, noun-phrase style (e.g. "Attention Residue Effect").
- Summaries: 3-5 sentences, high-level but precise.
- Return ONLY valid JSON — no explanation, no markdown, no preamble."""


def extraction_prompt(
    chunks: list[dict],       # batch of chunk dicts
    source_title: str,
    existing_tags: str,
    vault_summary: str,
) -> tuple[str, str]:
    """
    Build a single prompt covering a batch of chunks.
    Sending 2-3 chunks per call cuts round-trip overhead significantly.
    """
    system = EXTRACTION_SYSTEM

    # Build the text sections block
    sections = ""
    for c in chunks:
        sections += f"\n--- SECTION: {c['heading']} (chunk {c['chunk_index']}/{c['total_chunks']}) ---\n"
        sections += c["text"] + "\n"

    user = f"""SOURCE: {source_title}

EXISTING TAGS (reuse when possible):
{existing_tags}

EXISTING NOTES (link by exact title only):
{vault_summary}

TEXT:
{sections.strip()}

Extract the most valuable concepts across all sections above.
Return this exact JSON:
{{
  "concepts": [
    {{
      "title": "Concept Title",
      "summary": "3-5 sentence summary.",
      "significance": 8,
      "examples": ["example from source"],
      "tags": ["tag-one", "tag-two"],
      "links": [{{"to": "Exact Note Title", "bidirectional": true}}],
      "source_section": "section name where concept was found"
    }}
  ]
}}

If nothing scores 7+, return: {{"concepts": []}}"""

    return system, user


# ── Merge suggestion prompt ──────────────────────────────────────────────────

MERGE_SYSTEM = """You are a knowledge base curator.
Your job is to decide whether two notes represent the same concept and should be merged.
Be conservative — only suggest merging if the concepts are genuinely equivalent, not just related.
Return ONLY valid JSON."""


def merge_prompt(title_a: str, summary_a: str, title_b: str, summary_b: str) -> tuple[str, str]:
    """Prompt to evaluate whether two concepts should be merged."""
    system = MERGE_SYSTEM

    user = f"""Compare these two knowledge base notes:

NOTE A: "{title_a}"
{summary_a}

NOTE B: "{title_b}"
{summary_b}

Should these be merged into a single note?

Return:
{{
  "should_merge": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "one sentence explanation"
}}"""

    return system, user


# ── Tag normalisation prompt ─────────────────────────────────────────────────

def tag_normalisation_prompt(raw_tags: list[str], existing_tags: str) -> tuple[str, str]:
    """
    Ask the LLM to clean and normalise a set of raw tags against the existing tag list.
    Used as a post-processing step to prevent tag drift.
    """
    system = (
        "You are a tag normalisation engine for a personal knowledge base. "
        "Your job is to clean, deduplicate, and align tags with an existing tag vocabulary. "
        "Return ONLY valid JSON."
    )

    user = f"""EXISTING TAGS (use these whenever possible):
{existing_tags}

RAW TAGS TO NORMALISE:
{', '.join(raw_tags)}

Rules:
- Maximum 2 words per tag, lowercase, hyphen-separated if two words
- Prefer existing tags over new ones
- Drop tags that are too vague (e.g. "thing", "concept", "idea")
- Drop tags that duplicate another tag in the list
- Return at most 8 tags total

Return:
{{
  "normalised_tags": ["tag-one", "tag-two", "tag-three"]
}}"""

    return system, user