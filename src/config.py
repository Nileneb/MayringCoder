"""Central configuration and token-budget constants."""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "cache"
REPORTS_DIR = BASE_DIR / "reports"
PROMPTS_DIR = BASE_DIR / "prompts"
CODEBOOK_PATH = BASE_DIR / "codebook.yaml"

DEFAULT_PROMPT = PROMPTS_DIR / "file_inspector.md"
EXPLAINER_PROMPT = PROMPTS_DIR / "explainer.md"
OVERVIEW_PROMPT = PROMPTS_DIR / "overview.md"

# Token / budget limits
MAX_CHARS_PER_FILE = 3000
MAX_FILES_PER_RUN = 20          # 0 = kein Limit
MAX_FINDINGS_PER_FILE = 10

# GPU batching — pause every BATCH_SIZE files to cool down
BATCH_SIZE = 25
BATCH_DELAY_SECONDS = 5

# Ollama
OLLAMA_TIMEOUT = 120

# gitingest content separator (48 "=" characters, per gitingest source)
INGEST_SEPARATOR = "=" * 48

# Risk categories — prioritized in Top-K file selection
RISK_CATEGORIES: frozenset[str] = frozenset({"api", "data_access", "domain"})
