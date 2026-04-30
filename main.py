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


@click.group()
def cli():
    """PKM Automation — turns raw sources into structured Obsidian notes."""
    pass


# ── process ───────────────────────────────────────────────────────────────────

@click.command()
@click.option("--input",  "-i", "source",       required=True)
@click.option("--type",   "-t", "source_type",  required=True,
              type=click.Choice(["text", "pdf", "youtube"], case_sensitive=False))
@click.option("--source", "-s", "source_label", default="",
              help='Label e.g. "Book: Deep Work by Cal Newport"')
@click.option("--no-merge", is_flag=True, default=False,
              help="Skip merge detection (faster)")
def process(source: str, source_type: str, source_label: str, no_merge: bool):
    """Process a source and write Obsidian notes to the vault."""
    _print_header()
    _ensure_vault_exists()

    from llm_client import check_llm_ready
    if not check_llm_ready():
        click.echo(f"\n  ✗ LLM not ready. Check config.yaml and provider setup.\n")
        sys.exit(1)

    if not source_label:
        if source_type == "youtube":
            source_label = f"Video: {source}"
        else:
            source_label = Path(source).stem.replace("-", " ").replace("_", " ").title()

    click.echo(f"  Source   : {source}")
    click.echo(f"  Type     : {source_type}")
    click.echo(f"  Label    : {source_label}")
    click.echo(f"  Vault    : {vault_path()}")
    click.echo(f"  Provider : {cfg['llm']['provider']} / {cfg['llm']['model']}")
    click.echo()

    start = time.time()

    # Step 1 — Preprocess
    _step("1/4", "Preprocessing source")
    from __init__ import load
    content = load(source, source_type)
    meta    = content["metadata"]
    click.echo(f"       {meta.get('word_count', 0):,} words  |  "
               f"{meta.get('section_count', 0)} section(s)")

    # Step 2 — Chunk
    _step("2/4", "Chunking")
    from chunker import chunk
    chunks = chunk(content["text"], source_title=source_label)
    click.echo(f"       {len(chunks)} chunk(s)")

    # Step 3 — Extract
    _step("3/4", "Extracting concepts (LLM)")
    from extractor import extract
    concepts = extract(chunks, source_label=source_label)

    if not concepts:
        click.echo("\n  ⚠  No concepts extracted. Try a different model or check your source.\n")
        sys.exit(0)

    # Step 4 — Build notes
    _step("4/4", "Writing notes to vault")
    from note_builder import build_notes
    written = build_notes(concepts, source_label=source_label)

    concept_files = [p for p in written if "Concepts" in str(p)]
    source_files  = [p for p in written if "Sources"  in str(p)]

    # Merge detection
    merge_path = None
    if not no_merge:
        from merge_detector import detect_merges
        merge_path = detect_merges(concepts)

    _print_summary(concepts, concept_files, source_files, merge_path, time.time() - start)


# ── sync ──────────────────────────────────────────────────────────────────────

@click.command()
def sync():
    """Rebuild vault index from existing Obsidian notes. Run after manual edits."""
    _print_header()
    _ensure_vault_exists()
    click.echo("  Scanning vault...\n")

    from vault_manager import rebuild_index_from_vault, load_vault_index, load_tag_index
    rebuild_index_from_vault()

    notes = load_vault_index()
    tags  = load_tag_index()
    click.echo(f"\n  ✓ Index rebuilt — {len(notes)} note(s)  |  {len(tags)} tag(s)")
    click.echo(f"    {vault_path() / cfg['vault']['folders']['index']}\n")


# ── status ────────────────────────────────────────────────────────────────────

@click.command()
def status():
    """Show vault and LLM status."""
    _print_header()
    _ensure_vault_exists()

    from vault_manager import load_vault_index, load_tag_index
    notes    = load_vault_index()
    tags     = load_tag_index()
    concepts = list(folder("concepts").glob("*.md"))
    sources  = list(folder("sources").glob("*.md"))
    reviews  = list(folder("review").glob("*.md"))

    click.echo(f"  Vault      : {vault_path()}")
    click.echo(f"  Concepts   : {len(concepts)}")
    click.echo(f"  Sources    : {len(sources)}")
    click.echo(f"  Tags       : {len(tags)}")
    click.echo(f"  Review     : {len(reviews)} pending")
    click.echo(f"  Provider   : {cfg['llm']['provider']} / {cfg['llm']['model']}")
    click.echo()

    if tags:
        click.echo("  Tags in use:")
        for tag in sorted(tags)[:15]:
            click.echo(f"    #{tag}")
    click.echo()

    from llm_client import check_llm_ready
    ok = check_llm_ready()
    click.echo(f"  LLM status : {'✓ ready' if ok else '✗ not ready'}\n")


# ── Register ──────────────────────────────────────────────────────────────────

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
        click.echo(f"\n  ✗ Vault not found: {vp}")
        click.echo("  Update 'vault.path' in config.yaml\n")
        sys.exit(1)


def _print_summary(concepts, concept_files, source_files, merge_path, elapsed):
    click.echo()
    click.echo("  ┌─────────────────────────────────┐")
    click.echo("  │   Done                          │")
    click.echo("  └─────────────────────────────────┘")
    click.echo()
    click.echo(f"  Concepts  : {len(concept_files)}")
    click.echo(f"  Source    : {len(source_files)}")
    click.echo(f"  Links     : {sum(len(c.get('links', [])) for c in concepts)}")
    click.echo(f"  Tags      : {len(set(t for c in concepts for t in c.get('tags', [])))}")
    if merge_path:
        click.echo(f"  Review    : {merge_path.name}")
    click.echo(f"  Time      : {elapsed:.1f}s")
    click.echo()
    click.echo("  Open Obsidian — your notes are ready.")
    click.echo()


if __name__ == "__main__":
    cli()