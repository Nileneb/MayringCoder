"""CLI entry point und Argument-Parsing für MayringCoder-Pipeline."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.analysis.cache import reset_repo
from src.analysis.history import cleanup_runs, compare_runs, list_runs
from src.config import (
    CACHE_DIR,
    CODEBOOK_PATH,
    DEFAULT_PROMPT,
    EMBEDDING_MODEL,
    REPORTS_DIR,
    repo_slug as _repo_slug,
    set_batch_delay,
    set_batch_size,
    set_max_chars_per_file,
)
from src.analysis.context import set_max_context_chars
from src.model_selector import resolve_model
from src.pipeline import (
    run_analysis,
    run_ingest_images,
    run_ingest_issues,
    run_pi_task,
    run_populate_memory,
    run_turbulence,
)


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
    p.add_argument("--use-overview-cache", action="store_true",
                   help="Turbulence-Modus: Kategorien aus Overview-Cache übernehmen "
                        "statt per LLM/Heuristik neu zu kategorisieren.")
    p.add_argument("--use-turbulence-cache", action="store_true",
                   help="Analyze-Modus: Hot-Zone-Kontext aus Turbulence-Cache laden "
                        "und in den Analyse-Prompt injizieren. Dateien mit tier=stable "
                        "werden übersprungen.")
    p.add_argument("--rag-enrichment", action="store_true",
                   help="Finding-reaktive RAG-Queries: Jedes Finding bekommt "
                        "einen semantisch passenden Projektkontext aus der "
                        "Vektor-DB (benötigt vorherigen --mode overview Lauf).")
    p.add_argument("--populate-memory", action="store_true",
                   help="Repo laden und alle Dateien in die Memory-Pipeline ingesten.")
    p.add_argument("--memory-categorize", action="store_true",
                   help="Mayring-Kategorisierung während Memory-Ingestion aktivieren.")
    p.add_argument("--generate-wiki", action="store_true",
                   help="Verknüpfungswiki aus Overview-Cache + Memory erzeugen (cache/<slug>_wiki.md)")
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
    p.add_argument("--no-pi", action="store_true",
                   help="Pi-Agent deaktivieren (Standard: an). Pi nutzt Memory-Tool-Calling um "
                        "Projektkontext abzufragen und false positives zu reduzieren. "
                        "Deaktivieren wenn keine Memory-DB vorhanden oder Ressourcen knapp.")
    p.add_argument("--pi-task", metavar="TASK",
                   help="Freier Auftrag an Pi mit Memory-Zugriff (z.B. 'Entwickle PICO-Suchterms für X'). "
                        "Gibt die Antwort als Freitext aus — kein JSON-Zwang. "
                        "Optional: --repo für Memory-Scope-Filter.")
    p.add_argument("--workspace-id", default="default", metavar="ID",
                   help="Tenant workspace für Multi-Tenancy (Standard: 'default'). "
                        "Isoliert Memory-Daten und Reports per Nutzer. "
                        "Erstelle Workspaces mit: tools/manage_workspaces.py create <id>")
    return p.parse_args()


def main() -> None:
    import time
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

    if args.resolve_model_only:
        print(model)
        sys.exit(0)

    cache_run_key = args.run_id or (model if args.cache_by_model else "default")

    if args.log_training_data:
        from src.analysis.analyzer import configure_training_log
        _log_path = CACHE_DIR / f"{_repo_slug(os.getenv('GITHUB_REPO', 'unknown'))}_training_log.jsonl"
        configure_training_log(_log_path, run_id=cache_run_key)
        print(f"Training-Log: {_log_path}")

    if args.max_chars is not None:
        if args.max_chars < 500:
            print("Fehler: --max-chars muss mindestens 500 sein.")
            sys.exit(1)
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

    if not repo_url and not (args.ingest_issues or args.ingest_images or args.pi_task):
        print("Fehler: Kein Repository angegeben. Nutze --repo oder setze GITHUB_REPO in .env")
        sys.exit(1)

    if args.reset:
        removed = reset_repo(repo_url, run_key=(cache_run_key if args.run_id or args.cache_by_model else None))
        if removed:
            print(f"Cache gelöscht: {removed}")
            print("Nächster Lauf analysiert alle Dateien von vorn.")
        else:
            print("Kein Cache vorhanden — nichts zu löschen.")
        sys.exit(0)

    if args.history:
        runs = list_runs(repo_url, workspace_id=getattr(args, "workspace_id", "default"))
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

    if args.compare:
        try:
            cmp = compare_runs(args.compare[0], args.compare[1], repo_url, workspace_id=getattr(args, "workspace_id", "default"))
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

    if args.cleanup is not None:
        deleted = cleanup_runs(repo_url, keep=args.cleanup)
        print(f"{deleted} alte Runs gelöscht, {args.cleanup} behalten.")
        sys.exit(0)

    codebook_path = Path(args.codebook) if args.codebook else CODEBOOK_PATH
    prompt_path = Path(args.prompt) if args.prompt else DEFAULT_PROMPT

    if not prompt_path.exists():
        print(f"Fehler: Prompt-Datei nicht gefunden: {prompt_path}")
        sys.exit(1)

    _COMPAT_MAP = {
        "social.yaml": {"mayring_deduktiv.md", "mayring_induktiv.md"},
        "code.yaml": {"file_inspector.md", "smell_inspector.md", "explainer.md"},
    }
    cb_name = codebook_path.name
    pr_name = prompt_path.name
    for cb_pattern, prompt_set in _COMPAT_MAP.items():
        if cb_name == cb_pattern and pr_name not in prompt_set:
            print(
                f"Warnung: Codebook '{cb_name}' passt üblicherweise zu"
                f" {', '.join(sorted(prompt_set))} — nicht zu '{pr_name}'."
                f" Ergebnisse könnten unbrauchbar sein."
            )

    if cache_run_key != "default":
        print(f"Cache-Run-Key aktiv: {cache_run_key}")

    if args.full:
        print(f"\n{'='*60}")
        print("  FULL SCAN — Cache wird ignoriert, kein Datei-Limit")
        print(f"{'='*60}\n")

    if args.populate_memory:
        run_populate_memory(args, repo_url, ollama_url, model)
        sys.exit(0)

    if args.generate_wiki:
        from src.api.dependencies import get_conn, get_chroma
        from src.memory.wiki import generate_wiki
        generate_wiki(get_conn(), get_chroma(), repo_url, ollama_url, model, args.workspace_id)
        sys.exit(0)

    if args.ingest_issues:
        run_ingest_issues(args, ollama_url, model)
        sys.exit(0)

    if args.ingest_images:
        run_ingest_images(args, ollama_url, model)
        sys.exit(0)

    if args.pi_task:
        run_pi_task(args, ollama_url, model)
        sys.exit(0)

    if args.mode == "turbulence":
        turb_model = resolve_model(
            ollama_url,
            cli_model=args.model,
            env_model=os.getenv("TURB_MODEL"),
        ) if args.llm else (args.model or os.getenv("TURB_MODEL", "mistral:7b-instruct"))
        os.environ["OLLAMA_URL"] = ollama_url
        os.environ["TURB_MODEL"] = turb_model
        run_turbulence(args, repo_url, ollama_url, turb_model)
        sys.exit(0)

    start = time.perf_counter()
    run_analysis(args, repo_url, ollama_url, model, start)


if __name__ == "__main__":
    main()
