"""Central configuration and token-budget constants."""

import re
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "cache"
REPORTS_DIR = BASE_DIR / "reports"
PROMPTS_DIR = BASE_DIR / "prompts"
CODEBOOK_PATH = BASE_DIR / "codebook.yaml"

DEFAULT_PROMPT = PROMPTS_DIR / "file_inspector.md"
EXPLAINER_PROMPT = PROMPTS_DIR / "explainer.md"
OVERVIEW_PROMPT = PROMPTS_DIR / "overview.md"

# Token / budget limits
MAX_CHARS_PER_FILE = 20000
MAX_FILES_PER_RUN = 0          # 0 = kein Limit
MAX_FINDINGS_PER_FILE = 10

# GPU batching — pause every BATCH_SIZE files to cool down
BATCH_SIZE = 15
BATCH_DELAY_SECONDS = 10

# Project context budget (Phase 1: overview cache → prompt prefix)
MAX_CONTEXT_CHARS = 6000  # ~500 tokens

# RAG context (Phase 2: ChromaDB similarity search)
RAG_TOP_K = 5                          # Number of similar context entries to inject
EMBEDDING_MODEL = "nomic-embed-text"   # Ollama embedding model (offline)

# Ollama
OLLAMA_TIMEOUT = 240

# gitingest content separator (48 "=" characters, per gitingest source)
INGEST_SEPARATOR = "=" * 48

# Risk categories — prioritized in Top-K file selection
RISK_CATEGORIES: frozenset[str] = frozenset({"api", "data_access", "domain"})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def repo_slug(repo_url: str) -> str:
    """Normalize a repo URL to a safe filesystem slug, e.g. ``owner-repo``."""
    parsed = urlparse(repo_url)
    slug = parsed.path.strip("/").lower()
    slug = re.sub(r"\.git(?:/)?$", "", slug)
    slug = slug.replace("/", "-")
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    return slug or "repo"


# ---------------------------------------------------------------------------
# Runtime-overridable limits (set via CLI flags; read by analyzer)
# ---------------------------------------------------------------------------

_active_max_chars_per_file: int = MAX_CHARS_PER_FILE
_active_batch_size: int = BATCH_SIZE
_active_batch_delay: float = BATCH_DELAY_SECONDS


def set_max_chars_per_file(limit: int) -> None:
    """Override per-file truncation limit at runtime (called once from checker.py)."""
    global _active_max_chars_per_file
    _active_max_chars_per_file = max(1, int(limit))


def get_max_chars_per_file() -> int:
    """Return the active per-file char limit (default: MAX_CHARS_PER_FILE)."""
    return _active_max_chars_per_file


def set_batch_size(n: int) -> None:
    """Override GPU batch size at runtime (0 = no pause)."""
    global _active_batch_size
    _active_batch_size = max(0, int(n))


def get_batch_size() -> int:
    return _active_batch_size


def set_batch_delay(seconds: float) -> None:
    """Override GPU batch delay at runtime."""
    global _active_batch_delay
    _active_batch_delay = max(0.0, float(seconds))


def get_batch_delay() -> float:
    return _active_batch_delay
