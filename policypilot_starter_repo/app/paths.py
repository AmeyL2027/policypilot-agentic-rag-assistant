from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Canonical data layout
DATA_DIR = BASE_DIR / "data"
PUBLIC_DOCS_DIR = DATA_DIR / "public_docs"
SYNTHETIC_DOCS_DIR = DATA_DIR / "synthetic_docs"
PROCESSED_DIR = DATA_DIR / "processed"
QDRANT_LOCAL_DIR = DATA_DIR / "qdrant_local"

# Backward-compatible legacy location
LEGACY_PUBLIC_DOCS_DIR = DATA_DIR / "raw" / "public"


def resolve_public_docs_dir() -> Path:
    """
    Prefer the canonical public-doc folder. If it does not exist yet but the
    legacy folder has files, use the legacy folder to avoid breaking old setups.
    """
    if PUBLIC_DOCS_DIR.exists():
        return PUBLIC_DOCS_DIR
    if LEGACY_PUBLIC_DOCS_DIR.exists():
        return LEGACY_PUBLIC_DOCS_DIR
    return PUBLIC_DOCS_DIR
