"""
main.py — CLI entry point for the PKM Automation System.

Commands:
  process   Process a source file and populate the vault
  sync      Rebuild vault index from existing Obsidian notes
  status    Show vault statistics

Usage:
  python main.py process --input "file.pdf" --type pdf --source "Book: Deep Work"
  python main.py process --input "https://youtube.com/..." --type youtube
  python main.py sync
  python main.py status
"""

import sys
import time
from pathlib import Path

import click

from config import cfg, folder, vault_path


# ── CLI root ──────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """PKM Automation — turns raw sources into structured Obsidian notes."""
    pass


# ── process ───────────────────────────────────────────────────────────────────

@click.command()
@click.option("--input",  "-i", "source", required=True,  help="File path or YouTube URL")
@click.option("--type",   "-t", "source_type", required=True,
              type=click.Choice(["text", "pdf", "youtube"], case_sensitive=False),
              help="Source type")
@click.option("--source", "-s", "source_label", default="",
              help='Human-readable label, e.g. "Book: Deep Work by Cal Newport"')
@click.option("--no-merge", is_flag=True, default=False,
              help="Skip merge detection (faster, useful for first runs)")
def process(source: str, source_type: str, source_label: str, no_merge: bool):
    """Process a source and write Obsidian notes to the vault."""

    _print_header()

    # ── Preflight ──────────────────────────────────────────────────────────
    _ensure_vault_exists()

    from llm_client import check_ollama_running
    if not check_ollama_running():
        click.echo("\n  ✗ Cannot connect to Ollama. Start it with: ollama serve\n")
        sys.exit(1)

    # Auto-generate source label if not provided
    if not source_label:
        if source_type == "youtube":
            source_label = f"Video: {source}"
        else:
            source_label = Path(source).stem.replace("-", " ").replace("_", " ").title()

    click.echo(f"  Source : {source}")
    click.echo(f"  Type   : {source_type}")
    click.echo(f"  Label  : {source_label}")
    click.echo(f"  Vault  : {vault_path()}")
    click.echo()

    start = time.time()

    # ── Step 1: Preprocess ─────────────────────────────────────────────────
    _step("1/4", "Preprocessing source")
    from __init__ import load
    content = load(source, source_type)
    meta    = content["metadata"]
    click.echo(f"       {meta.get('word_count', 0):,} words  |  "
               f"{meta.get('section_count', 0)} section(s) detected")

    # ── Step 2: Chunk ──────────────────────────────────────────────────────
    _step("2/4", "Chunking")
    from chunker import chunk
    chunks = chunk(content)
    click.echo(f"       {len(chunks)} chunk(s) ready for extraction")

    # ── Step 3: Extract ────────────────────────────────────────────────────
    _step("3/4", "Extracting concepts (LLM)")
    from extractor import extract
    concepts = extract(chunks, source_label=source_label)

    if not concepts:
        click.echo("\n  ⚠  No concepts extracted. Check your source content or LLM output.\n")
        sys.exit(0)

    # ── Step 4: Build notes ────────────────────────────────────────────────
    _step("4/4", "Writing notes to vault")
    from note_builder import build_notes
    written = build_notes(concepts, source_label=source_label)

    concept_files = [p for p in written if "Concepts" in str(p)]
    source_files  = [p for p in written if "Sources" in str(p)]

    # ── Merge detection ────────────────────────────────────────────────────
    merge_path = None
    if not no_merge:
        from merge_detector import detect_merges
        merge_path = detect_merges(concepts)

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    _print_summary(
        concepts      = concepts,
        concept_files = concept_files,
        source_files  = source_files,
        merge_path    = merge_path,
        elapsed       = elapsed,
    )


# ── sync ──────────────────────────────────────────────────────────────────────

@click.command()
def sync():
    """Rebuild vault index from existing Obsidian notes. Run after manual edits."""
    _print_header()
    _ensure_vault_exists()

    click.echo("  Scanning vault for existing notes...\n")

    from vault_manager import rebuild_index_from_vault
    rebuild_index_from_vault()

    from vault_manager import load_vault_index, load_tag_index
    notes = load_vault_index()
    tags  = load_tag_index()

    click.echo(f"\n  ✓ Index rebuilt")
    click.echo(f"    {len(notes)} note(s)  |  {len(tags)} unique tag(s)")
    click.echo(f"    Index location: {vault_path() / cfg['vault']['folders']['index']}\n")


# ── status ────────────────────────────────────────────────────────────────────

@click.command()
def status():
    """Show vault statistics."""
    _print_header()
    _ensure_vault_exists()

    from vault_manager import load_vault_index, load_tag_index

    notes    = load_vault_index()
    tags     = load_tag_index()
    concepts = list(folder("concepts").glob("*.md"))
    sources  = list(folder("sources").glob("*.md"))
    reviews  = list(folder("review").glob("*.md"))

    click.echo(f"  Vault path  : {vault_path()}")
    click.echo(f"  Notes       : {len(concepts)} concept(s)  |  {len(sources)} source(s)")
    click.echo(f"  Tags        : {len(tags)} unique tag(s)")
    click.echo(f"  Pending review : {len(reviews)} file(s)")
    click.echo()

    if tags:
        click.echo("  Top tags:")
        for tag in sorted(tags)[:15]:
            click.echo(f"    #{tag}")

    click.echo()

    # LLM status
    from llm_client import check_ollama_running
    ollama_ok = check_ollama_running()
    status_icon = "✓" if ollama_ok else "✗"
    click.echo(f"  Ollama ({cfg['llm']['model']}) : {status_icon} {'running' if ollama_ok else 'not running'}\n")


# ── Register commands ─────────────────────────────────────────────────────────

cli.add_command(process)
cli.add_command(sync)
cli.add_command(status)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_header():
    click.echo()
    click.echo("  ┌─────────────────────────────────┐")
    click.echo("  │   PKM Automation System         │")
    click.echo("  └─────────────────────────────────┘")
    click.echo()


def _step(label: str, description: str):
    click.echo(f"  [{label}] {description}...")


def _ensure_vault_exists():
    vp = vault_path()
    if not vp.exists():
        click.echo(f"\n  ✗ Vault path not found: {vp}")
        click.echo("  Update 'vault.path' in config.yaml\n")
        sys.exit(1)


def _print_summary(
    concepts:      list[dict],
    concept_files: list[Path],
    source_files:  list[Path],
    merge_path:    Path | None,
    elapsed:       float,
):
    click.echo()
    click.echo("  ┌─────────────────────────────────┐")
    click.echo("  │   Done                          │")
    click.echo("  └─────────────────────────────────┘")
    click.echo()
    click.echo(f"  Concepts written  : {len(concept_files)}")
    click.echo(f"  Source note       : {len(source_files)}")

    # Count links generated
    total_links = sum(len(c.get("links", [])) for c in concepts)
    click.echo(f"  Links created     : {total_links}")

    # Count unique tags used
    all_tags = set(tag for c in concepts for tag in c.get("tags", []))
    click.echo(f"  Tags applied      : {len(all_tags)}")

    if merge_path:
        click.echo(f"  Merge suggestions : {merge_path.name}  (check /Review/)")
    else:
        click.echo(f"  Merge suggestions : none")

    click.echo(f"  Time elapsed      : {elapsed:.1f}s")
    click.echo()
    click.echo(f"  Open Obsidian → your notes are ready.")
    click.echo()


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
