"""
config.py — Loads and validates config.yaml.
All other modules import `cfg` from here.
"""

from pathlib import Path
import yaml


_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)

    # Resolve vault path (supports ~)
    raw["vault"]["path"] = str(Path(raw["vault"]["path"]).expanduser().resolve())

    return raw


cfg = _load()


# ── Convenience helpers ─────────────────────────────────────────────────────

def vault_path() -> Path:
    return Path(cfg["vault"]["path"])


def folder(name: str) -> Path:
    """Return absolute path to a named vault folder (concepts, sources, etc.)"""
    base = vault_path()
    sub  = cfg["vault"]["folders"][name]
    p    = base / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def index_path(filename: str) -> Path:
    """Return absolute path to a file inside the _index folder."""
    return folder("index") / filename