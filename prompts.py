"""
prompts.py — All LLM prompt templates in one place.

Keeping prompts separate makes them easy to tune without
touching extraction logic. Each function returns a
(system, user) tuple ready to pass to llm_client.
"""

# ── Extraction prompt ────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a knowledge extraction engine for a personal knowledge base.
Your job is to read a chunk of source material and extract only the most valuable, high-level concepts.

RULES:
- QUALITY OVER QUANTITY: Extract a MAXIMUM of 3 concepts per chunk. If there is only 1 good idea, only extract 1.
- THE SIGNIFICANCE FILTER: Assign a "significance" score from 1 to 10 for each concept (10 being the core thesis of the document). Only extract concepts that score 7 or higher. Ignore minor details, introductory text, and boilerplate.
- Each extracted concept must be self-contained and understandable on its own.
- Titles must be 2-6 words, noun-phrase style (e.g. "Attention Residue Effect").
- Summaries must be 3-5 sentences, high-level but precise.
- Return ONLY valid JSON — no explanation, no markdown, no preamble."""


def extraction_prompt(
    chunk_text: str,
    section_heading: str,
    source_title: str,
    chunk_index: int,
    total_chunks: int,
    existing_tags: str,
    vault_summary: str,
) -> tuple[str, str]:
    
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

Extract high-value concepts. For each concept, determine links and tags.
Return this exact JSON structure:
{{
  "concepts": [
    {{
      "title": "Concept Title Here",
      "summary": "3-5 sentence high-level summary of the concept.",
      "significance": 8,
      "examples": ["example 1 from source", "example 2 from source"],
      "tags": ["tag-one", "tag-two"],
      "links": [
        {{"to": "Exact Title Of Related Note", "bidirectional": true}}
      ],
      "source_section": "{section_heading}"
    }}
  ]
}}

If no concepts score a 7 or higher in this chunk, return: {{"concepts": []}}"""

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
