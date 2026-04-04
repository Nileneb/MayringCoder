# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Run the full 3-stage pipeline (overview → analyze → turbulence):**
```bash
bash run.sh
```

**Run individual stages:**
```bash
# Stage 1: Overview map (what each file does)
.venv/bin/python checker.py --mode overview --no-limit --max-chars 190000

# Stage 2: Full analysis (default mode)
.venv/bin/python checker.py --repo <URL> [--full] [--dry-run] [--adversarial]

# Stage 3: Turbulence analysis (mixed responsibilities / hot zones)
.venv/bin/python turbulence_run.py [--llm]
```

**Tests:**
```bash
.venv/bin/python -m pytest               # all tests
.venv/bin/python -m pytest tests/test_cache.py   # single file
.venv/bin/python -m pytest -k "test_name"        # single test
.venv/bin/python -m pytest --cov=src            # with coverage
```

**Environment (`.env`):**
```
GITHUB_REPO=https://github.com/owner/repo
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=<model-name>       # optional; prompted at runtime if unset
GITHUB_TOKEN=<token>            # optional; needed for private repos
TURB_MODEL=mistral:7b-instruct  # optional; for turbulence LLM mode
```

## Architecture

MayringCoder applies Mayring's qualitative content analysis methodology to GitHub repositories. The pipeline has three stages:

### Stage 1 — Overview (`--mode overview`)
`checker.py` → `src/fetcher.py` (gitingest) → `src/splitter.py` → `src/categorizer.py` → `src/analyzer.py:overview_files()` → `src/context.py` (saves JSON cache + ChromaDB index)

Fetches the repo via gitingest, splits the concatenated content into individual files, applies exclude patterns from `codebook.yaml` + `.mayringignore`, categorizes files into Mayring categories (api, data_access, domain, …), then runs a summarization LLM call per file. Results are saved to `cache/<repo-slug>_overview_context.json` and optionally indexed into ChromaDB for RAG.

### Stage 2 — Analyze (`--mode analyze`, default)
`checker.py` → `src/cache.py` (SQLite diff) → `src/analyzer.py:analyze_files()` → `src/extractor.py` → `src/aggregator.py` → `src/report.py`

Only changed/new files are analyzed (SQLite snapshot diff in `cache/<repo-slug>.db`). Each file is sent to the LLM with a prompt from `prompts/`. The LLM response is parsed as JSON; if that fails, `src/extractor.py` runs a regex fallback, then a second LLM extraction call (`extract_findings.md`). Test files are automatically routed to `prompts/test_inspector.md` if it exists. Optional `--adversarial` flag runs a second LLM pass (Advocatus Diaboli) to reject false positives. Findings are aggregated and a Markdown report is written to `reports/`.

### Stage 3 — Turbulence (`turbulence_run.py`)
`src/turbulence_analyzer.py:analyze_repo()` — Heuristic (fast, default) or LLM mode (`--llm`). Writes files to a temp dir, detects hot zones (lines with mixed responsibilities), measures per-file turbulence score (0–1), identifies potential redundancies between files. Output: `reports/turbulence-<ts>.json` + `.md`.

### Key modules

| Module | Role |
|---|---|
| `src/config.py` | All constants, `repo_slug()`, `set/get_max_chars_per_file()` |
| `src/categorizer.py` | Exclude-pattern matching, Mayring category assignment |
| `src/analyzer.py` | `_ollama_generate()` (streaming), `analyze_file()`, `overview_file()` |
| `src/extractor.py` | Stage-2 extraction + `validate_findings()` (adversarial) |
| `src/context.py` | Overview JSON cache + ChromaDB RAG (nomic-embed-text) |
| `src/cache.py` | SQLite snapshot diff, run-key based caching |
| `src/history.py` | Run persistence, `compare_runs()`, `cleanup_runs()` |
| `src/model_selector.py` | Resolves Ollama model (interactive prompt if unset) |

### Shared utilities in `src/config.py`

`config.py` is the single source of truth for constants **and** two runtime helpers that all modules must use — do not duplicate them elsewhere:

- `repo_slug(url)` — normalizes a GitHub URL to a safe filesystem slug (strips `.git`, replaces `/` with `-`). Used by `cache.py`, `context.py`, and `history.py` to derive cache paths.
- `set_max_chars_per_file(n)` / `get_max_chars_per_file()` — runtime override for the per-file char limit. `checker.py` calls the setter once; `analyzer.py` and `report.py` read via the getter at call time.

### Prompts
`prompts/` contains Markdown prompt files. Default: `file_inspector.md`. Alternatives: `overview.md`, `smell_inspector.md`, `explainer.md`, `test_inspector.md`, `extract_findings.md`. `codebook.yaml` defines which codebook/prompt combinations are compatible.

Two codebooks exist:
- `codebook.yaml` — Code quality analysis (pairs with `file_inspector.md`, `smell_inspector.md`, `explainer.md`)
- `codebook_sozialforschung.yaml` — Social science research (pairs with `mayring_deduktiv.md`, `mayring_induktiv.md`)

### Caching model
The SQLite DB (`cache/<repo-slug>.db`) stores snapshots and per-file hashes. Files are only re-analyzed when their hash changes. `--run-id` or `--cache-by-model` creates isolated cache namespaces so different models can be compared. `MAX_FILES_PER_RUN = 20` limits throughput per run (override with `--budget N` or `--no-limit`).

### Ollama integration
All LLM calls go through `src/analyzer.py:_ollama_generate()` with `stream=True` to avoid read timeouts on long responses. `OLLAMA_TIMEOUT = 240` seconds. Embedding calls use `/api/embed` (batch) with an in-process `_EMBEDDING_CACHE` to avoid redundant calls.
