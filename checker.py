#!/usr/bin/env python3
"""RepoChecker — Einstiegspunkt. Orchestriert die Mayring-Analyse-Pipeline."""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

import tempfile

from src.aggregator import aggregate_findings
from src.analyzer import analyze_files, configure_training_log, overview_files
from src.cache import find_changed_files, init_db, mark_files_analyzed, reset_repo
from src.categorizer import (
    categorize_files,
    filter_excluded_files,
    load_codebook,
    load_exclude_patterns,
    load_mayringignore,
)
from src.config import CACHE_DIR, CODEBOOK_PATH, DEFAULT_PROMPT, EMBEDDING_MODEL, MAX_FILES_PER_RUN, OVERVIEW_PROMPT, PROMPTS_DIR, set_batch_delay, set_batch_size, set_max_chars_per_file
from src.context import (
    index_overview_to_vectordb,
    load_overview_context,
    load_overview_cache_raw,
    query_similar_context,
    save_overview_context,
    set_max_context_chars,
)
from src.exporter import export_results
from src.fetcher import fetch_repo
from src.history import cleanup_runs, compare_runs, generate_run_id, list_runs, save_run
from src.model_selector import resolve_model
from src.report import generate_report, generate_overview_report
from src.splitter import split_into_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RepoChecker — lokale Code-Analyse mit Ollama")
    p.add_argument("--repo", help="GitHub-Repo URL (überschreibt .env)")
    p.add_argument("--model", help="Ollama-Modell (überschreibt .env)")
    p.add_argument("--full", action="store_true",
                   help="Full-Scan: Cache ignorieren, kein Datei-Limit (impliziert --no-limit)")
    p.add_argument("--dry-run", action="store_true", help="Nur Diff + Selektion zeigen, keine Analyse")
    p.add_argument("--show-selection", action="store_true", help="Zeigt ausgewählte Dateien inkl. Kategorie")
    p.add_argument("--prompt", help="Pfad zu einem alternativen Prompt")
    p.add_argument("--debug", action="store_true", help="Speichert Raw-Snapshot lokal unter cache/")
    p.add_argument("--reset", action="store_true", help="Cache-DB für das Repo löschen (alle Analysen zurücksetzen)")
    p.add_argument("--mode", choices=["analyze", "overview", "turbulence"], default="analyze",
                   help="Modus: 'overview' = Funktions-Übersicht, 'analyze' = Fehlersuche (Standard), "
                        "'turbulence' = Hot-Zone-Analyse (vermischte Verantwortlichkeiten)")
    p.add_argument("--llm", action="store_true",
                   help="Turbulenz-Modus: LLM für Chunk-Kategorisierung nutzen (langsamer, genauer). "
                        "Standard: Heuristik (kein Ollama nötig).")
    p.add_argument("--no-limit", action="store_true", help="Kein Datei-Limit pro Lauf (alle Dateien verarbeiten)")
    p.add_argument("--max-chars", type=int, metavar="N", help="Zeichenlimit pro Datei überschreiben (Kontextlimit wird automatisch angepasst)")
    p.add_argument("--budget", type=int, metavar="N", help="Datei-Limit pro Lauf überschreiben (Standard: 20)")
    p.add_argument("--log-training-data", action="store_true",
                   help="Jeden LLM-Call (Prompt + Antwort) in ein JSONL-Logfile schreiben. "
                        "Speicherort: cache/<slug>_training_log.jsonl. "
                        "Basis für Fine-Tuning-Datensätze.")
    p.add_argument("--time-budget", type=float, metavar="SECONDS",
                   help="Maximale Laufzeit in Sekunden. Nach Ablauf wird graceful gestoppt "
                        "und ein Report mit den bisherigen Ergebnissen geschrieben. "
                        "Ideal für Benchmark-Läufe mit vergleichbarem Zeitfenster pro Modell.")
    p.add_argument("--batch-size", type=int, metavar="N",
                   help="GPU-Pause alle N Dateien (0 = keine Pause, Standard: BATCH_SIZE aus config.py)")
    p.add_argument("--batch-delay", type=float, metavar="S",
                   help="Pausendauer in Sekunden (Standard: BATCH_DELAY_SECONDS aus config.py)")
    p.add_argument("--run-id", help="Logischer Run-Key für Cache + Report (ermöglicht Modell-/Run-Vergleiche)")
    p.add_argument("--cache-by-model", action="store_true", help="Modellnamen als Cache-Key verwenden (wenn kein --run-id gesetzt ist)")
    p.add_argument("--codebook", help="Pfad zu einem alternativen Codebook (YAML)")
    p.add_argument("--codebook-profile", metavar="PROFILE",
                   help="Codebook-Profil aus codebooks/profiles/ laden (z.B. laravel, python). "
                        "Überschreibt --codebook wenn gesetzt. Auto-Detection wenn nicht angegeben.")
    p.add_argument("--export", metavar="DATEI", help="Ergebnisse exportieren (.csv oder .json)")
    p.add_argument("--history", action="store_true", help="Vergangene Runs anzeigen")
    p.add_argument("--compare", nargs=2, metavar="RUN_ID", help="Zwei Runs vergleichen (alt neu)")
    p.add_argument("--cleanup", type=int, metavar="N", help="Nur die N neuesten Runs behalten, Rest löschen")
    p.add_argument("--resolve-model-only", action="store_true",
                   help="Gibt nur den aufgelösten Modellnamen aus und beendet (für Shell-Skripting)")
    p.add_argument("--min-confidence", choices=["high", "medium", "low"], default="low",
                   help="Minimale Confidence-Schwelle für Findings (Standard: low). "
                        "'high' zeigt nur Findings mit confidence=high (reduziert Falsch-Positive). "
                        "'medium' zeigt high+medium.")
    p.add_argument("--adversarial", action="store_true",
                   help="Jedes Finding wird durch einen zweiten LLM-Call (Advocatus Diaboli) "
                        "geprüft. ABGELEHNT-Findings werden verworfen. "
                        "Erhöht Token-Kosten, reduziert Falsch-Positive drastisch.")
    p.add_argument("--adversarial-cost-report", action="store_true",
                   help="Zeigt nach der Analyse: wie viele Findings BESTÄTIGT vs. ABGELEHNT wurden.")
    p.add_argument("--second-opinion", metavar="MODEL", default=None,
                   help="Zweites Modell für unabhängige Validierung (z. B. deepseek-coder:6.7b-instruct). "
                        "Jedes Finding wird mit reduziertem Kontext durch ein anderes Modell geprüft. "
                        "Verdikt: BESTÄTIGT / ABGELEHNT / PRÄZISIERT. "
                        "Überschreibt die Umgebungsvariable SECOND_OPINION_MODEL.")
    # Embedding prefilter (Issue #11)
    p.add_argument("--embedding-prefilter", action="store_true",
                   help="Aktiviert den Embedding-Vorfilter: Dateien werden anhand semantischer "
                        "Ähnlichkeit zur Forschungsfrage vorselektiert. "
                        "Reduziert LLM-Aufrufe bei großen Korpora.")
    p.add_argument("--embedding-model", default=None, metavar="MODEL",
                   help=f"Ollama-Embedding-Modell für den Vorfilter (Standard: {EMBEDDING_MODEL})")
    p.add_argument("--embedding-top-k", type=int, default=20, metavar="N",
                   help="Maximale Anzahl Dateien nach Embedding-Vorfilter (Standard: 20, 0 = kein Limit)")
    p.add_argument("--embedding-threshold", type=float, default=None, metavar="F",
                   help="Minimale Kosinus-Ähnlichkeit für Embedding-Vorfilter (z. B. 0.3)")
    p.add_argument("--embedding-query", default=None, metavar="TEXT",
                   help="Forschungsfrage / Suchbegriffe für den Embedding-Vorfilter. "
                        "Standard: wird aus Prompt-Name und Codebook abgeleitet.")
    # Feed-forward pipeline (Issue #17)
    p.add_argument("--use-overview-cache", action="store_true",
                   help="Turbulence-Modus: Kategorien aus Overview-Cache übernehmen "
                        "statt per LLM/Heuristik neu zu kategorisieren.")
    p.add_argument("--use-turbulence-cache", action="store_true",
                   help="Analyze-Modus: Hot-Zone-Kontext aus Turbulence-Cache laden "
                        "und in den Analyse-Prompt injizieren. Dateien mit tier=stable "
                        "werden übersprungen.")
    # Finding-reactive RAG (Issue #18)
    p.add_argument("--rag-enrichment", action="store_true",
                   help="Finding-reaktive RAG-Queries: Jedes Finding bekommt "
                        "einen semantisch passenden Projektkontext aus der "
                        "Vektor-DB (benötigt vorherigen --mode overview Lauf).")
    # Memory pipeline batch ingestion
    p.add_argument("--populate-memory", action="store_true",
                   help="Repo laden und alle Dateien in die Memory-Pipeline ingesten.")
    p.add_argument("--memory-categorize", action="store_true",
                   help="Mayring-Kategorisierung während Memory-Ingestion aktivieren.")
    p.add_argument("--ingest-issues", metavar="REPO",
                   help="GitHub Issues von REPO (owner/name) in Memory laden (benötigt gh CLI). "
                        "z. B. --ingest-issues Nileneb/MayringCoder")
    p.add_argument("--issues-state", choices=["open", "closed", "all"], default="open",
                   help="Welche Issues laden (Standard: open)")
    p.add_argument("--issues-limit", type=int, default=100, metavar="N",
                   help="Maximale Anzahl Issues (Standard: 100)")
    p.add_argument("--multiview", action="store_true",
                   help="Multi-view Indexing für Issues: LLM extrahiert Fact/Impl/Decision/Entities-Sichten")
    p.add_argument("--force-reingest", action="store_true",
                   help="Bestehende Chunks invalidieren und neu ingesten (ignoriert Dedup-Schutz)")
    p.add_argument("--ingest-images", metavar="REPO_URL",
                   help="Repo-Bilder (PNG/JPG/SVG) captionieren und in Memory ingesten. "
                        "z. B. --ingest-images https://github.com/Nileneb/app.linn.games")
    p.add_argument("--vision-model", default="qwen2.5vl:3b", metavar="MODEL",
                   help="Ollama Vision-Modell für Bild-Captioning (Standard: qwen2.5vl:3b)")
    p.add_argument("--max-images", type=int, default=50, metavar="N",
                   help="Maximale Anzahl Bilder pro Ingest-Lauf (Standard: 50)")
    p.add_argument("--gpu-metrics", action="store_true",
                   help="GPU-Metriken via nvidia-smi erfassen (VRAM, Auslastung, Watt, Temp). "
                        "Ergebnis unter cache/gpu_metrics_<ts>.csv.")
    p.add_argument("--pi", action="store_true",
                   help="Pi-Agent aktivieren: Kleines Modell nutzt Memory-Tool-Calling um Projektkontext "
                        "abzufragen, bevor Findings erzeugt werden. Reduziert false positives bei "
                        "schwachen Modellen (z.B. qwen3.5:2b). Benötigt befüllte Memory-DB "
                        "(--populate-memory, --ingest-issues oder --ingest-images).")
    p.add_argument("--pi-task", metavar="TASK",
                   help="Freier Auftrag an Pi mit Memory-Zugriff (z.B. 'Entwickle PICO-Suchterms für X'). "
                        "Gibt die Antwort als Freitext aus — kein JSON-Zwang. "
                        "Optional: --repo für Memory-Scope-Filter.")
    return p.parse_args()


def _repo_slug_for_debug(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    slug = parsed.path.strip("/").replace("/", "-").lower()
    return re.sub(r"[^a-z0-9\-]", "", slug) or "repo"


def _is_test_file(filename: str) -> bool:
    """Return True if *filename* looks like a test file (heuristic)."""
    import re as _re
    patterns = [
        _re.compile(r"(?:^|/)(?:test[s]?[_\-].*|tests?|spec|__tests?__)", _re.IGNORECASE),
        _re.compile(r"_test\.(?:py|php|js|ts|go|java)$", _re.IGNORECASE),
        _re.compile(r"(?:^|/)(?:test)\.\w+$", _re.IGNORECASE),
    ]
    return any(p.search(filename) for p in patterns)


def _load_prompt(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _load_turbulence_cache(repo_url: str) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    """Load turbulence cache and build hot-zone context map + tier map.

    Returns:
        (hot_zone_context_map, tier_map) — both None if cache doesn't exist.
        hot_zone_context_map: {filename: context_string} for prompt injection.
        tier_map: {filename: tier} for filtering stable files.
    """
    from src.config import CACHE_DIR, repo_slug as _repo_slug
    cache_path = CACHE_DIR / f"{_repo_slug(repo_url)}_turbulence.json"
    if not cache_path.exists():
        return None, None
    try:
        import json as _json
        report = _json.loads(cache_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None, None

    hot_zone_map: dict[str, str] = {}
    tier_map: dict[str, str] = {}

    for cf in report.get("all_files", report.get("critical_files", [])):
        path = cf.get("path", "")
        tier = cf.get("tier", "")
        tier_map[path] = tier

        zones = cf.get("hot_zones", [])
        if not zones:
            continue

        lines = ["## Hot-Zone-Kontext (aus Turbulenz-Analyse)"]
        for zone in zones:
            start = zone.get("start_line", "?")
            end = zone.get("end_line", "?")
            cats = zone.get("categories", [])
            peak = zone.get("peak_score", 0)
            cats_str = " × ".join(cats) if isinstance(cats, list) else str(cats)
            lines.append(
                f"ACHTUNG: Hot-Zone bei Zeile {start}-{end} "
                f"({cats_str}, Peak-Score: {peak:.0%})"
            )
            # Include affected functions from overview if available
            affected = zone.get("affected_functions", [])
            for fn_info in affected[:5]:
                if isinstance(fn_info, dict):
                    name = fn_info.get("name", "")
                    inputs = ", ".join(fn_info.get("inputs", []))
                    calls = ", ".join(fn_info.get("calls", []))
                    lines.append(f"  Betroffene Funktion: {name}({inputs}) → calls: {calls}")

        hot_zone_map[path] = "\n".join(lines)

    return hot_zone_map, tier_map


def main() -> None:
    load_dotenv()
    args = parse_args()

    repo_url = args.repo or os.getenv("GITHUB_REPO", "")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    _env_model = (os.getenv("OLLAMA_MODEL") or "").strip() or None
    _needs_llm = (
        args.mode in ("analyze", "overview")
        or (args.mode == "turbulence" and args.llm)
        or args.resolve_model_only
    )
    model = resolve_model(ollama_url, args.model, _env_model) if _needs_llm else (args.model or _env_model or "")

    # Shell-helper: print model name and exit (used by run.sh to avoid double-prompting).
    if args.resolve_model_only:
        print(model)
        sys.exit(0)

    token = os.getenv("GITHUB_TOKEN") or None
    prompt_path = Path(args.prompt) if args.prompt else DEFAULT_PROMPT
    codebook_path = Path(args.codebook) if args.codebook else CODEBOOK_PATH
    cache_run_key = args.run_id or (model if args.cache_by_model else "default")

    # --codebook-profile: profile-based modular codebook (overrides --codebook)
    _modular_codebook: list[dict] | None = None
    _modular_exclude_pats: list[str] | None = None
    if getattr(args, "codebook_profile", None):
        from src.categorizer import load_codebook_modular
        _modular_exclude_pats, _modular_codebook = load_codebook_modular(args.codebook_profile)

    # Training-data logger (opt-in)
    if args.log_training_data:
        from src.config import repo_slug as _rslug
        _log_path = CACHE_DIR / f"{_rslug(os.getenv('GITHUB_REPO', 'unknown'))}_training_log.jsonl"
        configure_training_log(_log_path, run_id=cache_run_key)
        print(f"Training-Log: {_log_path}")

    if args.max_chars is not None:
        if args.max_chars < 500:
            print("Fehler: --max-chars muss mindestens 500 sein.")
            sys.exit(1)

        # Keep one control variable: context budget follows per-file budget (2/3 ratio).
        max_chars_per_file = args.max_chars
        max_context_chars = max(500, (max_chars_per_file * 2) // 3)
        set_max_chars_per_file(max_chars_per_file)
        set_max_context_chars(max_context_chars)
        print(
            f"Limits aktiv: Datei={max_chars_per_file} Zeichen, "
            f"Kontext={max_context_chars} Zeichen"
        )

    if args.batch_size is not None:
        set_batch_size(args.batch_size)
    if args.batch_delay is not None:
        set_batch_delay(args.batch_delay)

    if not repo_url:
        print("Fehler: Kein Repository angegeben. Nutze --repo oder setze GITHUB_REPO in .env")
        sys.exit(1)

    # --reset: delete cache DB and exit
    if args.reset:
        removed = reset_repo(repo_url, run_key=(cache_run_key if args.run_id or args.cache_by_model else None))
        if removed:
            print(f"Cache gelöscht: {removed}")
            print("Nächster Lauf analysiert alle Dateien von vorn.")
        else:
            print("Kein Cache vorhanden — nichts zu löschen.")
        sys.exit(0)

    # --history: show past runs and exit
    if args.history:
        runs = list_runs(repo_url)
        if not runs:
            print("Keine Run-History vorhanden.")
        else:
            print(f"{'Run-ID':<20} {'Modus':<10} {'Modell':<18} {'Dateien':>7} {'Zeit (s)':>8}  Zeitstempel")
            print("-" * 90)
            for r in runs:
                print(
                    f"{r['run_id']:<20} {r['mode']:<10} {r['model']:<18} "
                    f"{r['files_checked']:>7} {r['timing_seconds']:>8.1f}  {r['timestamp']}"
                )
        sys.exit(0)

    # --compare: diff two runs and exit
    if args.compare:
        try:
            cmp = compare_runs(args.compare[0], args.compare[1], repo_url)
        except FileNotFoundError as exc:
            print(f"Fehler: {exc}")
            sys.exit(1)
        s = cmp["summary"]
        print(f"Vergleich: {cmp['run_a']} → {cmp['run_b']}")
        print(f"  Dateien: {s['files_a']} → {s['files_b']}")
        print(f"  Neu:      {s['new_count']}")
        print(f"  Behoben:  {s['resolved_count']}")
        print(f"  Severity: {s['severity_changed_count']} geändert")
        if cmp["new"]:
            print("\nNeue Findings:")
            for f in cmp["new"]:
                print(f"  + [{f.get('severity','?')}] {f['_filename']}: {f.get('type','')}")
        if cmp["resolved"]:
            print("\nBehobene Findings:")
            for f in cmp["resolved"]:
                print(f"  - [{f.get('severity','?')}] {f['_filename']}: {f.get('type','')}")
        if cmp["severity_changed"]:
            print("\nSeverity geändert:")
            for f in cmp["severity_changed"]:
                print(f"  ~ {f['_filename']}: {f.get('type','')} {f['_old_severity']} → {f['_new_severity']}")
        sys.exit(0)

    # --cleanup: keep only N newest runs and exit
    if args.cleanup is not None:
        deleted = cleanup_runs(repo_url, keep=args.cleanup)
        print(f"{deleted} alte Runs gelöscht, {args.cleanup} behalten.")
        sys.exit(0)

    if not prompt_path.exists():
        print(f"Fehler: Prompt-Datei nicht gefunden: {prompt_path}")
        sys.exit(1)

    # Codebook/Prompt-Kompatibilitätscheck
    _COMPAT_MAP = {
        "codebook_sozialforschung.yaml": {"mayring_deduktiv.md", "mayring_induktiv.md"},
        "codebook.yaml": {"file_inspector.md", "smell_inspector.md", "explainer.md"},
    }
    cb_name = codebook_path.name
    pr_name = prompt_path.name
    for cb_pattern, prompt_set in _COMPAT_MAP.items():
        if cb_name == cb_pattern and pr_name not in prompt_set:
            print(
                f"⚠ Warnung: Codebook '{cb_name}' passt üblicherweise zu"
                f" {', '.join(sorted(prompt_set))} — nicht zu '{pr_name}'."
                f" Ergebnisse könnten unbrauchbar sein."
            )

    start = time.perf_counter()

    if cache_run_key != "default":
        print(f"Cache-Run-Key aktiv: {cache_run_key}")

    if args.full:
        print(f"\n{'='*60}")
        print("  FULL SCAN — Cache wird ignoriert, kein Datei-Limit")
        print(f"{'='*60}\n")

    # 1. Fetch
    print(f"Repository laden: {repo_url} ...")
    summary, tree, content = fetch_repo(repo_url, token)

    if args.debug:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = CACHE_DIR / f"{_repo_slug_for_debug(repo_url)}_raw_latest.txt"
        raw_path.write_text(content, encoding="utf-8")
        print(f"  [debug] Raw-Snapshot: {raw_path}")

    # 2. Split
    files = split_into_files(content)
    print(f"{len(files)} Dateien gefunden")

    if not files:
        print("Keine analysierbaren Dateien im Repository.")
        sys.exit(0)

    # 2b. Exclude-Patterns anwenden (vor Kategorisierung)
    if _modular_exclude_pats is not None:
        exclude_pats = _modular_exclude_pats + load_mayringignore()
    else:
        exclude_pats = load_exclude_patterns(codebook_path) + load_mayringignore()
    files, excluded = filter_excluded_files(files, exclude_pats)
    if excluded:
        print(f"  ✗ {len(excluded)} Dateien ausgeschlossen (exclude patterns)")
    print(f"  → {len(files)} Dateien nach Filter")

    if not files:
        print("Alle Dateien wurden durch Exclude-Patterns herausgefiltert.")
        sys.exit(0)

    # 3. Categorize (Mayring Stufe 1: Strukturierung)
    codebook = _modular_codebook if _modular_codebook is not None else load_codebook(codebook_path)
    files = categorize_files(files, codebook)
    categories = {f["filename"]: f["category"] for f in files}

    # Determine effective file limit
    # --full implies --no-limit (bypass budget cap)
    if args.full or args.no_limit:
        max_files = 0
    elif args.budget is not None:
        max_files = args.budget
    else:
        max_files = MAX_FILES_PER_RUN

    # 3b. Embedding prefilter (Issue #11) — optional, applied before cache diff
    embedding_prefilter_meta: dict | None = None
    if args.embedding_prefilter:
        from src.embedder import filter_by_embedding

        embed_model = args.embedding_model or EMBEDDING_MODEL

        # Derive a default query from the prompt name and codebook name
        if args.embedding_query:
            embed_query = args.embedding_query
        else:
            prompt_label = prompt_path.stem.replace("_", " ")
            cb_label = codebook_path.stem.replace("_", " ")
            embed_query = f"{prompt_label} {cb_label} code quality analysis"

        print(
            f"\nEmbedding-Vorfilter aktiv"
            f" (Modell: {embed_model}, Top-K: {args.embedding_top_k}"
            + (f", Threshold: {args.embedding_threshold}" if args.embedding_threshold is not None else "")
            + f")"
        )
        print(f"  Query: {embed_query!r}")

        selected_by_embed, filtered_out_by_embed = filter_by_embedding(
            files=files,
            query=embed_query,
            ollama_url=ollama_url,
            top_k=args.embedding_top_k,
            threshold=args.embedding_threshold,
            embedding_model=embed_model,
            repo_url=repo_url,
        )

        n_before = len(files)
        files = [f for f in files if f["filename"] in set(selected_by_embed)]
        categories = {fn: cat for fn, cat in categories.items() if fn in set(selected_by_embed)}
        print(
            f"  → {len(files)} Dateien nach Embedding-Vorfilter"
            f" ({n_before - len(files)} herausgefiltert)"
        )

        embedding_prefilter_meta = {
            "model": embed_model,
            "top_k": args.embedding_top_k,
            "threshold": args.embedding_threshold,
            "query": embed_query,
            "files_before": n_before,
            "files_after": len(files),
            "filtered_out": filtered_out_by_embed,
        }

        if not files:
            print("Embedding-Vorfilter hat alle Dateien herausgefiltert. Abbruch.")
            sys.exit(0)

    # 3b. Memory batch ingestion mode
    if args.populate_memory:
        _run_populate_memory(args, repo_url, ollama_url, model)
        sys.exit(0)

    # 3c. GitHub Issues ingestion mode
    if args.ingest_issues:
        _run_ingest_issues(args, ollama_url, model)
        sys.exit(0)

    if args.ingest_images:
        _run_ingest_images(args, ollama_url, model)
        sys.exit(0)

    if args.pi_task:
        _run_pi_task(args, ollama_url, model)
        sys.exit(0)

    # 4. Turbulenz-Modus: eigene Pipeline, kein Cache-Diff nötig
    if args.mode == "turbulence":
        if args.full:
            print("--full: Turbulenz-Analyse scannt standardmäßig alle Dateien.")
        # Priority: --model CLI flag → TURB_MODEL env → interactive (if --llm) → fallback
        turb_model = resolve_model(
            ollama_url,
            cli_model=args.model,
            env_model=os.getenv("TURB_MODEL"),
        ) if args.llm else (args.model or os.getenv("TURB_MODEL", "mistral:7b-instruct"))
        os.environ["OLLAMA_URL"] = ollama_url
        os.environ["TURB_MODEL"] = turb_model
        _run_turbulence(args, repo_url, ollama_url, turb_model)
        sys.exit(0)

    # 5. Diff
    if args.full:
        filenames_to_check = [f["filename"] for f in files]
        diff: dict = {
            "changed": [], "added": filenames_to_check, "removed": [],
            "unchanged": [], "unanalyzed": filenames_to_check,
            "selected": filenames_to_check, "skipped": [],
            "snapshot_id": None,
        }
        conn = None
        mode_label = "overview" if args.mode == "overview" else "analyze"
        print(f"--full ({mode_label}): {len(filenames_to_check)} Dateien werden analysiert")
    elif args.mode == "overview":
        # Overview processes ALL files, no cache interaction
        filenames_to_check = [f["filename"] for f in files]
        diff = {
            "changed": [], "added": filenames_to_check, "removed": [],
            "unchanged": [], "unanalyzed": filenames_to_check,
            "selected": filenames_to_check, "skipped": [],
            "snapshot_id": None,
        }
        conn = None
        print(f"--mode overview: {len(filenames_to_check)} Dateien kartieren (keine Fehlersuche)")
    else:
        conn = init_db(repo_url)
        diff = find_changed_files(conn, repo_url, files, categories, max_files, run_key=cache_run_key)
        # conn stays open — needed for mark_files_analyzed after analysis

        n_c, n_a, n_r, n_u = (
            len(diff["changed"]), len(diff["added"]),
            len(diff["removed"]), len(diff["unchanged"]),
        )
        n_queue = len(diff["unanalyzed"])
        print(f"{n_c} geändert, {n_a} neu, {n_r} gelöscht, {n_u} unverändert | {n_queue} in Analyse-Queue")

        filenames_to_check = diff["selected"]
        if diff.get("skipped"):
            print(
                f"  → {len(filenames_to_check)} ausgewählt"
                f" (Budget-Limit: {max_files}),"
                f" {len(diff['skipped'])} verbleiben auf nächste Runs"
            )

    if not filenames_to_check:
        elapsed = time.perf_counter() - start
        print(f"Keine Änderungen seit dem letzten Run. Fertig in {elapsed:.0f}s")
        if conn:
            conn.close()
        sys.exit(0)

    if args.show_selection or args.dry_run:
        print("\nDateien zur Analyse:")
        for fn in filenames_to_check:
            print(f"  • [{categories.get(fn, 'uncategorized')}] {fn}")
        if diff.get("skipped"):
            print(f"\nÜbersprungen ({len(diff['skipped'])}):")
            for fn in diff["skipped"]:
                print(f"  ○ {fn}")

    if args.dry_run:
        elapsed = time.perf_counter() - start
        print(f"\nFertig in {elapsed:.0f}s")
        sys.exit(0)

    # 5. Run the selected mode
    if args.mode == "overview":
        overview_prompt = OVERVIEW_PROMPT
        print(f"\nErstelle Übersicht für {len(filenames_to_check)} Dateien mit {model} ...")
        results, _time_budget_hit = overview_files(
            files, filenames_to_check, overview_prompt, ollama_url, model,
            time_budget=args.time_budget,
        )

        # Save overview context for later analyze runs
        ctx_path = save_overview_context(results, repo_url)
        print(f"  Kontext gespeichert: {ctx_path}")

        # Index into vector DB for RAG (Phase 2) — non-fatal; analyze falls back to flat context.
        try:
            n_indexed = index_overview_to_vectordb(repo_url, ollama_url)
            if n_indexed:
                print(f"  Vektor-DB indiziert: {n_indexed} Einträge ({EMBEDDING_MODEL})")
        except Exception as exc:
            print(f"  ⚠ Vektor-DB Indexierung fehlgeschlagen (RAG deaktiviert): {exc}")

        # 7. Overview Report (kein Aggregate nötig)
        elapsed = time.perf_counter() - start
        report_path = generate_overview_report(repo_url, model, results, diff, elapsed, run_id=cache_run_key, full_scan=args.full)
        print(f"\nReport: {report_path}")
        if args.export:
            ep = export_results(results, args.export, codebook_path.name, "overview")
            print(f"Export: {ep}")

        # Save run history
        rid = args.run_id or generate_run_id()
        run_path = save_run(rid, repo_url, model, "overview", results, diff, elapsed)
        print(f"Run-History: {run_path.name}")
        print(f"Fertig in {elapsed:.0f}s")
    else:
        # 5. Analyze (Mayring Stufe 2: Reduktion + Stufe 3: Explikations-Markierung)
        # Try RAG context (Phase 2), fall back to flat overview context (Phase 1)
        rag_context_fn = None
        project_context = None

        # Check if vector DB exists for this repo
        from src.context import _chroma_dir
        if _chroma_dir(repo_url).exists():
            def _rag_ctx(file: dict) -> str | None:
                query = f"[{file.get('category', '?')}] {file['filename']}"
                return query_similar_context(query, repo_url, ollama_url)
            rag_context_fn = _rag_ctx
            print("  RAG-Kontext aktiv (ChromaDB → Similarity Search)")
        else:
            project_context = load_overview_context(repo_url)
            if project_context:
                print("  Projektkontext aus Overview-Cache geladen (Phase 1 Fallback)")

        # ── Feed-forward: load turbulence cache for hot-zone context (Issue #17) ──
        hot_zone_context_map: dict[str, str] | None = None
        if args.use_turbulence_cache:
            hot_zone_context_map, _turb_tiers = _load_turbulence_cache(repo_url)
            if hot_zone_context_map is not None:
                n_hz = sum(1 for v in hot_zone_context_map.values() if v)
                print(f"  Hot-Zone-Kontext geladen: {n_hz} Dateien mit Hot-Zones")
                # Tier-based filtering: skip stable files
                if _turb_tiers:
                    stable_files = {fn for fn, tier in _turb_tiers.items() if tier == "stable"}
                    before = len(filenames_to_check)
                    filenames_to_check = [fn for fn in filenames_to_check if fn not in stable_files]
                    skipped = before - len(filenames_to_check)
                    if skipped:
                        print(f"  → {skipped} stabile Dateien übersprungen (tier=stable)")
            else:
                print("  Turbulence-Cache nicht gefunden — kein Hot-Zone-Kontext")

        # ── P6: Separate test-prompt routing ─────────────────────────────────
        # Split files into test vs non-test. Test files get test_inspector.md.
        test_prompt_path = PROMPTS_DIR / "test_inspector.md"
        use_test_prompt = test_prompt_path.exists()

        test_files, non_test_files = [], []
        for fn in filenames_to_check:
            if _is_test_file(fn):
                test_files.append(fn)
            else:
                non_test_files.append(fn)

        if test_files and use_test_prompt:
            print(f"  → {len(test_files)} Test-Datei(en) werden mit test_inspector.md analysiert")

        results: list[dict] = []

        # ── Pi-Agent: Memory-DB check ────────────────────────────────────────
        _use_pi = getattr(args, "pi", False)
        from src.config import repo_slug as _repo_slug_fn
        _pi_repo_slug = _repo_slug_fn(repo_url)
        if _use_pi:
            try:
                from src.memory_store import init_memory_db as _init_mem_db
                _mem_conn = _init_mem_db()
                _chunk_count = _mem_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                _mem_conn.close()
                if _chunk_count == 0:
                    print(
                        "  Warnung: Pi-Agent aktiv, aber Memory-DB enthält keine Chunks. "
                        "Starte zuvor --populate-memory, --ingest-issues oder --ingest-images."
                    )
                else:
                    print(f"  Pi-Agent aktiv: {_chunk_count} Memory-Chunks verfügbar")
            except Exception as _pi_check_exc:
                print(f"  Warnung: Pi-Agent Memory-Check fehlgeschlagen: {_pi_check_exc}")

        # ── Main analysis (non-test files) ───────────────────────────────────
        _time_budget_hit = False
        if non_test_files:
            print(f"\nAnalysiere {len(non_test_files)} Nicht-Test-Dateien mit {model} ...")
            batch_results, _time_budget_hit = analyze_files(
                files, non_test_files, prompt_path, ollama_url, model,
                project_context=project_context,
                context_fn=rag_context_fn,
                hot_zone_context_map=hot_zone_context_map,
                time_budget=args.time_budget,
                use_pi=_use_pi,
                pi_repo_slug=_pi_repo_slug,
            )
            results.extend(batch_results)

        # ── Test analysis (test_inspector.md) ───────────────────────────────
        if test_files and use_test_prompt and not _time_budget_hit:
            print(f"\nAnalysiere {len(test_files)} Test-Dateien mit {model} (test_inspector.md) ...")
            batch_results, _tbh = analyze_files(
                files, test_files, test_prompt_path, ollama_url, model,
                project_context=None,
                context_fn=None,
                time_budget=args.time_budget,
                use_pi=_use_pi,
                pi_repo_slug=_pi_repo_slug,
            )
            results.extend(batch_results)
            _time_budget_hit = _time_budget_hit or _tbh

        # Mark successfully analyzed files so they leave the backlog queue
        if conn is not None and diff.get("snapshot_id") is not None:
            analyzed_ok = [r["filename"] for r in results if "error" not in r]
            mark_files_analyzed(conn, diff["snapshot_id"], analyzed_ok, run_key=cache_run_key)
            remaining = len(diff["unanalyzed"]) - len(analyzed_ok)
            print(f"  → {len(analyzed_ok)} analysiert, {max(0, remaining)} verbleiben in Queue")
        if conn:
            conn.close()

        # ── Finding-reactive RAG enrichment (Issue #18) ────────────────────
        if args.rag_enrichment:
            from src.context import enrich_findings_with_rag, _chroma_dir
            if _chroma_dir(repo_url).exists():
                n_findings = sum(
                    len(r.get("potential_smells", []) + r.get("codierungen", []))
                    for r in results if "error" not in r
                )
                print(f"\nRAG-Enrichment: bereichere {n_findings} Findings ...", flush=True)
                results = enrich_findings_with_rag(results, repo_url, ollama_url)
                n_enriched = sum(
                    1 for r in results if "error" not in r
                    for s in (r.get("potential_smells", []) + r.get("codierungen", []))
                    if s.get("_rag_context")
                )
                print(f"  → {n_enriched}/{n_findings} Findings mit RAG-Kontext angereichert")
            else:
                print("  RAG-Enrichment: Vektor-DB nicht gefunden (--mode overview zuerst ausführen)")

        # ── P5: Adversarial validation (Advocatus Diaboli) ───────────────────
        adversarial_stats: dict | None = None
        if args.adversarial:
            print(f"\nAdvocatus Diaboli: prüfe {len(results)} Findings ...", flush=True)
            from src.extractor import validate_findings
            all_findings: list[dict] = []
            for r in results:
                if "error" in r:
                    continue
                for smell in r.get("potential_smells", []):
                    all_findings.append({**smell, "_filename": r["filename"]})
            if all_findings:
                validated, adversarial_stats = validate_findings(
                    all_findings, results, ollama_url, model,
                    min_confidence=args.min_confidence,
                )
                n_total = len(all_findings)
                n_rejected = adversarial_stats["rejected"]
                print(
                    f"  Adversarial: {adversarial_stats['validated']}/{n_total} BESTÄTIGT, "
                    f"{n_rejected} ABGELEHNT, "
                    f"{adversarial_stats.get('below_confidence', 0)} unter Confidence-Schwelle"
                )
                # Remove ABGELEHNT findings from results
                rejected_filenames: set[str] = {f["_filename"] for f in validated}
                for r in results:
                    if r["filename"] not in rejected_filenames:
                        r["potential_smells"] = [
                            s for s in r.get("potential_smells", [])
                            if s.get("_adversarial_verdict") != "ABGELEHNT"
                        ]

        # ── P5b: Second-opinion validation (different model) ────────────────
        second_opinion_model_name = (
            args.second_opinion
            or os.getenv("SECOND_OPINION_MODEL")
        )
        second_opinion_stats: dict | None = None
        if second_opinion_model_name:
            from src.extractor import second_opinion_validate
            all_findings_so: list[dict] = []
            for r in results:
                if "error" in r:
                    continue
                for smell in r.get("potential_smells", []):
                    all_findings_so.append({**smell, "_filename": r["filename"]})
            if all_findings_so:
                print(
                    f"\nSecond Opinion ({second_opinion_model_name}):"
                    f" prüfe {len(all_findings_so)} Findings ...",
                    flush=True,
                )
                validated_so, second_opinion_stats = second_opinion_validate(
                    all_findings_so, results, ollama_url, second_opinion_model_name
                )
                second_opinion_stats["model"] = second_opinion_model_name
                print(
                    f"  Second Opinion: "
                    f"{second_opinion_stats['confirmed']} BESTÄTIGT, "
                    f"{second_opinion_stats['rejected']} ABGELEHNT, "
                    f"{second_opinion_stats['refined']} PRÄZISIERT, "
                    f"{second_opinion_stats['errors']} Fehler"
                )
                # Remove ABGELEHNT findings from results
                for r in results:
                    r["potential_smells"] = [
                        s for s in r.get("potential_smells", [])
                        if s.get("_second_opinion_verdict") != "ABGELEHNT"
                    ]

        # ── 6. Aggregate (Mayring Stufe 4: Zusammenführung) ─────────────────
        min_conf = args.min_confidence
        aggregation = aggregate_findings(
            results,
            min_confidence=min_conf,
            adversarial_stats=adversarial_stats if args.adversarial else None,
            second_opinion_stats=second_opinion_stats,
        )

        # ── 7. Report ───────────────────────────────────────────────────────
        elapsed = time.perf_counter() - start
        report_path = generate_report(
            repo_url, model, results, aggregation, diff, elapsed,
            run_id=cache_run_key,
            embedding_prefilter_meta=embedding_prefilter_meta,
            full_scan=args.full,
            time_budget_hit=_time_budget_hit,
        )
        n_filtered = aggregation.get("_below_confidence_filtered", 0)
        print(f"\nReport: {report_path}")
        if n_filtered > 0:
            print(f"  → {n_filtered} Findings unter Confidence-Schwelle '{min_conf}' herausgefiltert")
        if args.adversarial_cost_report or args.adversarial:
            stats = aggregation.get("adversarial_stats", {})
            if stats:
                print(
                    f"  Adversarial: {stats.get('validated', 0)} BESTÄTIGT, "
                    f"{stats.get('rejected', 0)} ABGELEHNT, "
                    f"{stats.get('errors', 0)} Fehler"
                )
        if second_opinion_stats:
            print(
                f"  Second Opinion: "
                f"{second_opinion_stats.get('confirmed', 0)} BESTÄTIGT, "
                f"{second_opinion_stats.get('rejected', 0)} ABGELEHNT, "
                f"{second_opinion_stats.get('refined', 0)} PRÄZISIERT"
            )
        if args.export:
            ep = export_results(results, args.export, codebook_path.name, "analyze")
            print(f"Export: {ep}")

        # Save run history
        rid = args.run_id or generate_run_id()
        run_path = save_run(rid, repo_url, model, "analyze", results, diff, elapsed, aggregation,
                            extra={"time_budget_hit": _time_budget_hit})
        print(f"Run-History: {run_path.name}")
        print(f"Fertig in {elapsed:.0f}s")


def _run_turbulence(args, repo_url: str, ollama_url: str, turb_model: str) -> None:
    """Turbulenz-Analyse (Stufe 2 in Feed-Forward): Hot-Zone-Erkennung über alle Repo-Dateien."""
    import json
    from datetime import datetime
    from src.config import CACHE_DIR, REPORTS_DIR, repo_slug as _repo_slug
    from src.turbulence_analyzer import analyze_repo
    from src.turbulence_report import build_markdown

    print(f"\n{'='*60}")
    print("  Stufe 2: Turbulenz-Analyse")
    print(f"  Repo:    {repo_url}")
    print(f"  Modus:   {'LLM (' + turb_model + ')' if args.llm else 'Heuristik (schnell)'}")
    if args.use_overview_cache:
        print("  Overview-Cache: aktiv")
    print(f"{'='*60}")

    start = time.perf_counter()

    # Feed-forward: load overview cache if requested (Issue #17)
    overview_cache = None
    if args.use_overview_cache:
        overview_cache = load_overview_cache_raw(repo_url)
        if overview_cache:
            print(f"  Overview-Cache geladen: {len(overview_cache)} Dateien")
        else:
            print("  Overview-Cache nicht gefunden — Fallback auf Standard-Kategorisierung")

    print(f"\nRepository laden: {repo_url} ...")
    _, _, content = fetch_repo(repo_url, os.getenv("GITHUB_TOKEN") or None)

    files = split_into_files(content)
    print(f"{len(files)} Dateien gefunden")

    if not files:
        print("Keine analysierbaren Dateien — Turbulenz-Analyse übersprungen.")
        return

    def _write_files(file_list: list[dict], target_dir: str) -> int:
        base = Path(target_dir)
        written = 0
        for f in file_list:
            p = base / f["filename"]
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                p.write_text(f["content"], encoding="utf-8")
                written += 1
            except OSError:
                pass
        return written

    with tempfile.TemporaryDirectory(prefix="turb_") as tmpdir:
        written = _write_files(files, tmpdir)
        print(f"{written} Dateien ins Arbeitsverzeichnis geschrieben\n")
        report = analyze_repo(
            tmpdir,
            use_llm=args.llm,
            model=turb_model if args.llm else None,
            overview_cache=overview_cache,
        )

    report["full_scan"] = args.full
    elapsed = time.perf_counter() - start

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")

    json_path = REPORTS_DIR / f"turbulence-{ts}.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Stable cache path for feed-forward pipeline (Issue #17)
    cache_path = CACHE_DIR / f"{_repo_slug(repo_url)}_turbulence.json"
    cache_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    md_path = REPORTS_DIR / f"turbulence-{ts}.md"
    md_path.write_text(
        build_markdown(report, repo_url, turb_model, elapsed, full_scan=args.full),
        encoding="utf-8",
    )

    print(f"\nReport (JSON): {json_path}")
    print(f"Report (MD):   {md_path}")
    print(f"Cache:         {cache_path}")
    print(f"Fertig in {elapsed:.0f}s")


def _run_populate_memory(args, repo_url: str, ollama_url: str, model: str) -> None:
    """Batch-ingest all repository files into the memory pipeline."""
    from src.memory_ingest import get_or_create_chroma_collection, ingest
    from src.memory_store import init_memory_db
    from src.memory_schema import Source, source_fingerprint
    from src.gpu_metrics import start_monitoring, stop_monitoring, parse_metrics, format_summary

    import hashlib as _hashlib

    def _content_hash(text: str) -> str:
        return "sha256:" + _hashlib.sha256(text.encode("utf-8")).hexdigest()

    token = os.getenv("GITHUB_TOKEN") or None
    codebook_path = Path(args.codebook) if getattr(args, "codebook", None) else CODEBOOK_PATH

    # GPU-Monitoring starten (optional)
    _gpu_proc = None
    _gpu_csv = None
    if getattr(args, "gpu_metrics", False):
        from datetime import datetime as _dt
        _gpu_csv = CACHE_DIR / f"gpu_metrics_{_dt.now().strftime('%Y%m%d_%H%M%S')}.csv"
        _gpu_proc = start_monitoring(_gpu_csv)
        if _gpu_proc:
            print(f"[populate-memory] GPU-Monitoring aktiv → {_gpu_csv}")

    print(f"[populate-memory] Repository laden: {repo_url} ...")
    _summary, _tree, _repo_content = fetch_repo(repo_url, token)

    files = split_into_files(_repo_content)
    print(f"[populate-memory] {len(files)} Dateien gefunden")

    # Auto-detect profile from gitingest tree if no explicit profile set
    _profile = getattr(args, "codebook_profile", None)
    if not _profile and _tree:
        from src.categorizer import detect_profile_from_tree
        _profile = detect_profile_from_tree(_tree)
        print(f"[populate-memory] Profil auto-detected: {_profile}")

    # Use modular codebook if profile available, else fallback
    if _profile:
        from src.categorizer import load_codebook_modular
        exclude_pats, codebook = load_codebook_modular(_profile)
        exclude_pats = list(exclude_pats) + load_mayringignore()
        print(f"[populate-memory] Codebook-Profil: {_profile} ({len(codebook)} Kategorien, {len(exclude_pats)} Excludes)")
    else:
        codebook = load_codebook(codebook_path)
        exclude_pats = load_exclude_patterns(codebook_path) + load_mayringignore()

    files, excluded = filter_excluded_files(files, exclude_pats)
    print(f"[populate-memory] {len(excluded)} Dateien ausgeschlossen, {len(files)} verbleiben")

    if not files:
        print("[populate-memory] Keine Dateien nach Filter — Abbruch.")
        return

    conn = init_memory_db()
    chroma = get_or_create_chroma_collection()

    total = len(files)
    ok_count = 0
    error_count = 0
    dedup_count = 0

    try:
        for f in files:
            content_hash = _content_hash(f["content"])
            source = Source(
                source_id=f"repo:{repo_url}:{f['filename']}",
                source_type="repo_file",
                repo=repo_url,
                path=f["filename"],
                content_hash=content_hash,
                branch="",
                commit="",
            )
            try:
                result = ingest(
                    source,
                    f["content"],
                    conn,
                    chroma,
                    ollama_url,
                    model,
                    opts={
                        "categorize": args.memory_categorize,
                        "mode": "hybrid",
                        "codebook": getattr(args, "codebook_profile", None) or "auto",
                    },
                )
                dedup_count += result.get("deduped", 0)
                ok_count += 1
            except Exception as exc:
                print(f"[populate-memory] FEHLER bei {f['filename']!r}: {exc}")
                error_count += 1
    finally:
        conn.close()

    if _gpu_proc:
        stop_monitoring(_gpu_proc)
        if _gpu_csv:
            gpu_summary = parse_metrics(_gpu_csv)
            print(f"[populate-memory] {format_summary(gpu_summary)}")

    print(
        f"\n[populate-memory] Fertig: {total} Dateien total, "
        f"{ok_count} OK, {error_count} Fehler, {dedup_count} Dedup."
    )


def _run_ingest_issues(args, ollama_url: str, model: str) -> None:
    """GitHub Issues in die Memory-Pipeline ingesten."""
    from src.ingest_github_issues import fetch_issues, issues_to_sources
    from src.memory_ingest import get_or_create_chroma_collection, ingest
    from src.memory_store import init_memory_db
    from src.gpu_metrics import start_monitoring, stop_monitoring, parse_metrics, format_summary

    issues_repo = args.ingest_issues
    state = getattr(args, "issues_state", "open")
    limit = getattr(args, "issues_limit", 100)
    do_multiview = getattr(args, "multiview", False)

    # GPU-Monitoring starten (optional)
    _gpu_proc = None
    _gpu_csv = None
    if getattr(args, "gpu_metrics", False):
        from datetime import datetime as _dt
        _gpu_csv = CACHE_DIR / f"gpu_metrics_issues_{_dt.now().strftime('%Y%m%d_%H%M%S')}.csv"
        _gpu_proc = start_monitoring(_gpu_csv)
        if _gpu_proc:
            print(f"[ingest-issues] GPU-Monitoring aktiv → {_gpu_csv}")

    print(f"[ingest-issues] Issues laden von {issues_repo!r} (state={state}, limit={limit}) ...")
    issues = fetch_issues(issues_repo, state=state, limit=limit)
    if not issues:
        print("[ingest-issues] Keine Issues gefunden oder gh CLI nicht verfügbar.")
        return

    print(f"[ingest-issues] {len(issues)} Issues gefunden")
    sources = issues_to_sources(issues, issues_repo)

    do_force = getattr(args, "force_reingest", False)
    conn = init_memory_db()
    chroma = get_or_create_chroma_collection()
    ok_count = 0
    error_count = 0
    dedup_count = 0

    if do_force:
        from src.memory_store import deactivate_chunks_by_source, get_chunks_by_source
        from src.memory_retrieval import invalidate_query_cache
        print("[ingest-issues] --force-reingest: Bestehende Chunks werden invalidiert ...")
        old_chunk_ids: list[str] = []
        for source, _ in sources:
            old_chunks = get_chunks_by_source(conn, source.source_id, active_only=False)
            old_chunk_ids.extend(c.chunk_id for c in old_chunks)
            deactivate_chunks_by_source(conn, source.source_id)
        if old_chunk_ids:
            try:
                chroma.delete(ids=old_chunk_ids)
            except Exception:
                pass
        invalidate_query_cache()

    try:
        for source, content in sources:
            try:
                result = ingest(
                    source,
                    content,
                    conn,
                    chroma,
                    ollama_url,
                    model,
                    opts={"categorize": False, "mode": "hybrid", "codebook": "social", "multiview": do_multiview},
                )
                dedup_count += result.get("deduped", 0)
                ok_count += 1
            except Exception as exc:
                print(f"[ingest-issues] FEHLER bei Issue {source.path!r}: {exc}")
                error_count += 1
    finally:
        conn.close()

    if _gpu_proc:
        stop_monitoring(_gpu_proc)
        if _gpu_csv:
            gpu_summary = parse_metrics(_gpu_csv)
            print(f"[ingest-issues] {format_summary(gpu_summary)}")

    print(
        f"\n[ingest-issues] Fertig: {len(sources)} Issues total, "
        f"{ok_count} OK, {error_count} Fehler, {dedup_count} Dedup."
    )


def _run_ingest_images(args, ollama_url: str, model: str) -> None:
    """Repo-Bilder captionieren und in Memory ingesten."""
    from src.image_ingest import run_image_ingest

    repo_url = args.ingest_images
    vision_model = getattr(args, "vision_model", "qwen2.5vl:3b")
    max_images = getattr(args, "max_images", 50)
    do_force = getattr(args, "force_reingest", False)

    print(f"[ingest-images] Starte Bild-Ingest für: {repo_url}")
    print(f"[ingest-images] Vision-Modell: {vision_model}, Max-Bilder: {max_images}")

    result = run_image_ingest(
        repo_url=repo_url,
        ollama_url=ollama_url,
        vision_model=vision_model,
        embed_model=model,
        max_images=max_images,
        force_reingest=do_force,
    )

    print(
        f"\n[ingest-images] Fertig: {result['images_found']} Bilder total, "
        f"{result['images_captioned']} captioniert, "
        f"{result['images_skipped']} Dedup, "
        f"{result['images_failed']} Fehler."
    )


def _run_pi_task(args, ollama_url: str, model: str) -> None:
    """Freien Auftrag an Pi delegieren (mit Memory-Zugriff)."""
    from src.pi_agent import run_task_with_memory

    task = args.pi_task
    repo_url = getattr(args, "repo", None)
    repo_slug = None
    if repo_url:
        from src.config import repo_slug as _slug_fn
        repo_slug = _slug_fn(repo_url)

    print(f"[pi-task] Auftrag: {task[:80]}{'...' if len(task) > 80 else ''}")
    if repo_slug:
        print(f"[pi-task] Memory-Scope: {repo_slug}")
    print()

    result = run_task_with_memory(
        task=task,
        ollama_url=ollama_url,
        model=model,
        repo_slug=repo_slug,
    )

    print(result)


if __name__ == "__main__":
    main()
