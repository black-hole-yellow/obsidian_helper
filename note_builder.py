"""
note_builder.py — Converts extracted concept dicts into Obsidian .md files.

Responsibilities:
  - Build the .md content (frontmatter + body)
  - Handle bidirectional links (patch the target note to add back-link)
  - Write files into the correct vault folders
  - Write a source summary note into /Sources/
  - Update vault_index and tag_index after writing
  - Never overwrite an existing note — append new content instead
"""

from datetime import date
from pathlib import Path
from typing import Optional
import re

from config import folder
from vault_manager import (
    add_note_to_index,
    add_tags_to_index,
    load_vault_index,
)


# ── Main entry ────────────────────────────────────────────────────────────────

def build_notes(concepts: list[dict], source_label: str = "") -> list[Path]:
    """
    Write one .md file per concept into the Concepts folder.

    Args:
        concepts:     Output from extractor.extract()
        source_label: Human-readable source name for the source summary note

    Returns:
        List of paths to all written .md files
    """
    written: list[Path] = []
    concepts_dir = folder("concepts")

    # Build a title→path map for this batch (for bidirectional linking)
    title_to_path: dict[str, Path] = {}
    for c in concepts:
        safe   = _safe_filename(c["title"])
        title_to_path[c["title"]] = concepts_dir / f"{safe}.md"

    # Also load existing vault notes for bidirectional patching
    vault_index = load_vault_index()
    for record in vault_index:
        existing_path = concepts_dir / record["file"]
        if existing_path.exists():
            title_to_path.setdefault(record["title"], existing_path)

    # Write each concept
    for concept in concepts:
        path = _write_concept(concept, concepts_dir, title_to_path)
        if path:
            written.append(path)
            add_note_to_index(concept["title"], concept["tags"], path.name)
            add_tags_to_index(concept["tags"])

    # Apply bidirectional links (patch target notes)
    _apply_backlinks(concepts, title_to_path)

    # Write source summary note
    if source_label and concepts:
        summary_path = _write_source_summary(concepts, source_label)
        if summary_path:
            written.append(summary_path)

    print(f"[note_builder] Wrote {len([p for p in written if 'Concepts' in str(p)])} concept note(s)")
    return written


# ── Concept note writer ───────────────────────────────────────────────────────

def _write_concept(
    concept: dict,
    concepts_dir: Path,
    title_to_path: dict[str, Path],
) -> Optional[Path]:
    """
    Write a single concept .md file.
    If the file already exists, append new content under a separator
    rather than overwriting — preserving manual edits.
    """
    filename = _safe_filename(concept["title"]) + ".md"
    path     = concepts_dir / filename

    content  = _render_concept(concept)

    if path.exists():
        # Append new version under a dated separator
        existing = path.read_text(encoding="utf-8")
        separator = f"\n\n---\n*Updated: {date.today()}*\n\n"
        # Only append the body (skip frontmatter duplicate)
        body_only = _strip_frontmatter(content)
        path.write_text(existing + separator + body_only, encoding="utf-8")
    else:
        path.write_text(content, encoding="utf-8")

    return path


def _render_concept(concept: dict) -> str:
    """Render a concept dict as Obsidian markdown."""
    today       = str(date.today())
    tags        = concept.get("tags", [])
    links       = concept.get("links", [])
    examples    = concept.get("examples", [])
    source      = concept.get("source_label", "")
    section     = concept.get("source_section", "")

    # ── YAML Frontmatter ─────────────────────────────────────────────────────
    tag_list = ", ".join(tags)
    frontmatter = f"---\ntags: [{tag_list}]\ndate: {today}\nsource: \"{source}\"\n---\n"

    # ── Title ─────────────────────────────────────────────────────────────────
    lines = [frontmatter, f"# {concept['title']}", ""]

    # ── Summary ───────────────────────────────────────────────────────────────
    lines.append(concept["summary"])
    lines.append("")

    # ── Examples ──────────────────────────────────────────────────────────────
    if examples:
        lines.append("## Examples")
        for ex in examples:
            lines.append(f"- {ex}")
        lines.append("")

    # ── Links ─────────────────────────────────────────────────────────────────
    if links:
        lines.append("## Links")
        for link in links:
            arrow = "↔" if link["bidirectional"] else "→"
            lines.append(f"- {arrow} [[{link['to']}]]")
        lines.append("")

    # ── Source reference ──────────────────────────────────────────────────────
    if source or section:
        lines.append("## Source")
        if source:
            lines.append(f"- **From:** {source}")
        if section:
            lines.append(f"- **Section:** {section}")
        lines.append("")

    return "\n".join(lines)


# ── Bidirectional back-links ──────────────────────────────────────────────────

def _apply_backlinks(concepts: list[dict], title_to_path: dict[str, Path]) -> None:
    """
    For every bidirectional link, open the target note and add a
    back-link to the source concept if it isn't already there.
    """
    for concept in concepts:
        source_title = concept["title"]
        for link in concept.get("links", []):
            if not link.get("bidirectional"):
                continue

            target_title = link["to"]
            target_path  = title_to_path.get(target_title)

            if not target_path or not target_path.exists():
                continue

            content = target_path.read_text(encoding="utf-8")
            backlink = f"[[{source_title}]]"

            if backlink in content:
                continue  # already linked

            # Find the Links section and append, or add a new one
            if "## Links" in content:
                content = content.replace(
                    "## Links",
                    f"## Links\n- ↔ {backlink}",
                    1,
                )
            else:
                content += f"\n## Links\n- ↔ {backlink}\n"

            target_path.write_text(content, encoding="utf-8")


# ── Source summary note ───────────────────────────────────────────────────────

def _write_source_summary(concepts: list[dict], source_label: str) -> Optional[Path]:
    """
    Write a single summary note into /Sources/ listing all concepts
    extracted from this source. Acts as an index for the source.
    """
    sources_dir = folder("sources")
    filename    = _safe_filename(source_label) + ".md"
    path        = sources_dir / filename
    today       = str(date.today())

    lines = [
        f"---",
        f"tags: [source-note]",
        f"date: {today}",
        f"---",
        f"",
        f"# {source_label}",
        f"",
        f"*Processed: {today} — {len(concepts)} concept(s) extracted*",
        f"",
        f"## Concepts",
        f"",
    ]

    # Group by section
    by_section: dict[str, list[str]] = {}
    for c in concepts:
        section = c.get("source_section", "General")
        by_section.setdefault(section, []).append(c["title"])

    for section, titles in by_section.items():
        lines.append(f"### {section}")
        for title in titles:
            lines.append(f"- [[{title}]]")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_filename(title: str) -> str:
    """Convert a concept title to a safe filename."""
    safe = re.sub(r'[<>:"/\\|?*]', "", title)   # remove illegal chars
    safe = re.sub(r"\s+", " ", safe).strip()      # collapse spaces
    return safe[:100]                              # max 100 chars


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter block from content."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            return content[end + 3:].lstrip("\n")
    return content