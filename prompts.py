"""
prompts.py — All LLM prompt templates in one place.

Keeping prompts separate makes them easy to tune without
touching extraction logic. Each function returns a
(system, user) tuple ready to pass to llm_client.
"""

# ── Extraction prompt ────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a knowledge extraction engine for a personal knowledge base.
Your job is to read a chunk of source material and extract every meaningful atomic concept,
idea, event, action, or principle it contains.

RULES:
- Each extracted concept/event/action/principle must be self-contained and understandable on its own
- Prefer specific, concrete ideas over vague generalities
- Extract only what is genuinely present in the text — do not invent or infer beyond it
- Titles must be 2-6 words, noun-phrase style (e.g. "Attention Residue Effect")
- Summaries must be 3-5 sentences, high-level but precise
- Examples must come directly from the source text — do not fabricate them
- Tags: 1-2 words max, describe an event or action, reuse existing tags when possible
- Links: only reference concepts that have a clear, meaningful relationship
- Return ONLY valid JSON — no explanation, no markdown, no preamble"""


def extraction_prompt(
    chunk_text: str,
    section_heading: str,
    source_title: str,
    chunk_index: int,
    total_chunks: int,
    existing_tags: str,
    vault_summary: str,
) -> tuple[str, str]:
    """
    Build the extraction prompt for a single chunk.

    Returns (system_prompt, user_prompt) tuple.
    """
    system = EXTRACTION_SYSTEM

    user = f"""SOURCE: {source_title}
SECTION: {section_heading}
POSITION: Chunk {chunk_index} of {total_chunks}

EXISTING TAGS IN MY KNOWLEDGE BASE (reuse these when possible):
{existing_tags}

EXISTING NOTES IN MY KNOWLEDGE BASE (use for linking — reference by exact title):
{vault_summary}

TEXT TO EXTRACT FROM:
\"\"\"
{chunk_text}
\"\"\"

Extract all atomic concepts from this text. For each concept, determine:
1. Which existing notes it should link to (use exact titles from the list above)
2. Whether each link is bidirectional (the other concept is equally about this one)
   or one-directional (this concept references the other, but not vice versa)
3. Which existing tags apply, and what new tags are needed

Return this exact JSON structure:
{{
  "concepts": [
    {{
      "title": "Concept Title Here",
      "summary": "3-5 sentence high-level summary of the concept.",
      "examples": ["example 1 from source", "example 2 from source"],
      "tags": ["tag-one", "tag-two", "tag-three"],
      "links": [
        {{"to": "Exact Title Of Related Note", "bidirectional": true}},
        {{"to": "Another Related Note", "bidirectional": false}}
      ],
      "source_section": "{section_heading}"
    }}
  ]
}}

If no meaningful concepts exist in this chunk, return: {{"concepts": []}}"""

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
