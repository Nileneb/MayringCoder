"""MayringCoder Pipeline — **Direkt-Executor** (kein HTTP).

Rollen-Abgrenzung (siehe Issue #66 Phase 5):
  - `python -m src.cli`  — DIESE DATEI. Wird vom API-Server als
                           subprocess gestartet (src.api.job_queue.
                           run_checker_job). Ruft src/workflows/*
                           direkt auf, greift lokal auf memory.db,
                           chromadb und ollama zu. Keine Remote-Auth.
  - `src/pipeline.py`    — 40-LOC BC-Shim, re-exportiert dieses Modul
                           unter altem Namen für bestehende Aufrufer.
  - `checker.py`         — HTTP-Client (remote), ruft den API-Server,
                           nicht dieses Modul.

Wer ingesten oder analysieren will ohne Server: direkt hier entlang.
Wer gegen den Prod-Server arbeitet: nutz stattdessen `checker.py`.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.analysis.cache import reset_repo
from src.analysis.history import cleanup_runs, compare_runs, list_runs
from src.cli_args import parse_args
from src.config import (
    CACHE_DIR,
    CODEBOOK_PATH,
    DEFAULT_PROMPT,
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

def _cmd_history(args: argparse.Namespace, repo_url: str) -> None:
    runs = list_runs(repo_url, workspace_id=getattr(args, "workspace_id", "default"))
    if not runs:
        print("Keine Run-History vorhanden.")
        return
    print(f"{'Run-ID':<20} {'Modus':<10} {'Modell':<18} {'Dateien':>7} {'Zeit (s)':>8}  Zeitstempel")
    print("-" * 90)
    for r in runs:
        print(
            f"{r['run_id']:<20} {r['mode']:<10} {r['model']:<18} "
            f"{r['files_checked']:>7} {r['timing_seconds']:>8.1f}  {r['timestamp']}"
        )


def _cmd_compare(args: argparse.Namespace, repo_url: str) -> None:
    try:
        cmp = compare_runs(
            args.compare[0], args.compare[1], repo_url,
            workspace_id=getattr(args, "workspace_id", "default"),
        )
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


def _cmd_cleanup(args: argparse.Namespace, repo_url: str) -> None:
    deleted = cleanup_runs(repo_url, keep=args.cleanup)
    print(f"{deleted} alte Runs gelöscht, {args.cleanup} behalten.")


def _cmd_generate_wiki(args: argparse.Namespace, repo_url: str, ollama_url: str, model: str) -> None:
    wid = args.workspace_id or "default"
    from src.api.dependencies import get_conn, get_chroma
    from src.memory.wiki import generate_wiki
    generate_wiki(get_conn(), get_chroma(), repo_url, ollama_url, model, wid, doc_type=args.wiki_type)
    try:
        from src.analysis.context import load_overview_cache_raw
        from src.wiki_v2.clustering import ClusterEngine
        from src.wiki_v2.edge_detector import EdgeDetector
        from src.wiki_v2.graph import WikiGraph
        from src.wiki_v2.history import WikiHistory
        from src.wiki_v2.models import WikiNode
        from src.config import repo_slug as _repo_slug_fn
        slug = _repo_slug_fn(repo_url) if repo_url else wid
        oc = load_overview_cache_raw(repo_url) or {} if repo_url else {}
        db = WikiGraph(wid, slug, CACHE_DIR / "wiki_v2.db")
        detector = EdgeDetector()
        edges = detector.detect_from_overview(oc, get_conn(), wid, slug)
        node_ids = set(oc.keys()) | {e.source for e in edges} | {e.target for e in edges}
        for nid in sorted(node_ids):
            db.upsert_node(WikiNode(id=nid, repo_slug=slug, workspace_id=wid))
        for e in edges:
            db.add_edge(e)
        clusters = ClusterEngine().cluster(
            db, strategy=getattr(args, "wiki_cluster_strategy", "louvain"),
            ollama_url=ollama_url, model=model,
        )
        so_model = getattr(args, "wiki_second_opinion", None)
        if so_model and clusters:
            from src.wiki_v2.second_opinion import WikiSecondOpinion
            so = WikiSecondOpinion()
            c_verdicts = so.validate_clusters(clusters, db, so_model, ollama_url)
            clusters = so.apply_cluster_verdicts(clusters, c_verdicts, db)
            e_verdicts = so.validate_edges(db.get_edges(), db, so_model, ollama_url)
            print(so.second_opinion_report(c_verdicts, e_verdicts))
        db.to_json()
        WikiHistory().create_snapshot(db, trigger="rebuild")
        db.close()
        print(f"Wiki 2.0 graph.json geschrieben für workspace '{wid}'")
    except Exception as _e:
        print(f"Wiki 2.0 update skipped: {_e}")


def _cmd_wiki_history(args: argparse.Namespace) -> None:
    import sqlite3 as _sq
    from src.wiki_v2.history import WikiHistory
    wid = args.workspace_id or "default"
    conn = _sq.connect(str(CACHE_DIR / "wiki_v2.db"))
    conn.row_factory = _sq.Row
    snaps = WikiHistory().timeline(conn, wid)
    conn.close()
    if not snaps:
        print(f"Keine Snapshots für workspace '{wid}'.")
        return
    print(f"{'ID':>5}  {'Trigger':<20} {'Nodes':>5} {'Edges':>5} {'Cluster':>7}  Zeitstempel")
    print("-" * 65)
    for s in snaps:
        print(f"{s.snapshot_id:>5}  {s.trigger:<20} {s.node_count:>5} {s.edge_count:>5} {s.cluster_count:>7}  {s.created_at}")


def _cmd_wiki_team_activity(args: argparse.Namespace) -> None:
    import sqlite3 as _sq
    from src.wiki_v2.history import team_activity
    wid = args.workspace_id or "default"
    conn = _sq.connect(str(CACHE_DIR / "wiki_v2.db"))
    conn.row_factory = _sq.Row
    activity = team_activity(conn, wid)
    conn.close()
    if not activity:
        print(f"Keine Contributions für workspace '{wid}' (letzte 30 Tage).")
        return
    print(f"Team-Aktivität workspace '{wid}' (letzte 30 Tage):")
    for user, count in sorted(activity.items(), key=lambda x: -x[1]):
        print(f"  {user}: {count} Aktionen")


def _cmd_wiki_history_cleanup(args: argparse.Namespace) -> None:
    import sqlite3 as _sq
    from src.wiki_v2.history import WikiHistory
    wid = args.workspace_id or "default"
    keep = args.wiki_history_cleanup
    conn = _sq.connect(str(CACHE_DIR / "wiki_v2.db"))
    conn.row_factory = _sq.Row
    deleted = WikiHistory().cleanup(conn, wid, keep=keep)
    conn.close()
    print(f"{deleted} Snapshot(s) gelöscht, {keep} neueste behalten.")


def _cmd_rebuild_transitions(args: argparse.Namespace, repo_url: str) -> None:
    from src.api.dependencies import get_conn
    from src.memory.predictive import build_transition_matrix, persist_transitions
    conn = get_conn()
    matrix = build_transition_matrix(conn, repo_slug=(_repo_slug(repo_url) if repo_url else ""))
    persist_transitions(matrix, conn)
    print(f"[transitions] {len(matrix)} from-topics, {sum(len(v) for v in matrix.values())} edges → topic_transitions")


def _cmd_generate_ambient(args: argparse.Namespace, repo_url: str, ollama_url: str, model: str) -> None:
    from src.api.dependencies import get_conn
    from src.memory.ambient import generate_ambient_snapshot
    result = generate_ambient_snapshot(get_conn(), ollama_url, model, _repo_slug(repo_url), args.workspace_id)
    if result:
        print(f"[ambient] Snapshot generiert ({len(result)} Zeichen)")
    else:
        print("[ambient] Kein Snapshot generiert (kein Model oder Fehler)", file=sys.stderr)


def _cmd_turbulence(args: argparse.Namespace, repo_url: str, ollama_url: str) -> None:
    turb_model = (
        resolve_model(ollama_url, cli_model=args.model, env_model=os.getenv("TURB_MODEL"))
        if args.llm
        else (args.model or os.getenv("TURB_MODEL", "mistral:7b-instruct"))
    )
    os.environ["OLLAMA_URL"] = ollama_url
    os.environ["TURB_MODEL"] = turb_model
    run_turbulence(args, repo_url, ollama_url, turb_model)


def _cmd_generate_training_data(args: argparse.Namespace) -> None:
    from src.training.memory_context_generator import run as run_memory_gen, DEFAULT_OUTPUT
    wid = getattr(args, "workspace_id", "default") or "default"
    result = run_memory_gen(
        workspace_id=wid,
        output_path=DEFAULT_OUTPUT,
        skip_feedback=getattr(args, "skip_auto_feedback", False),
        limit=getattr(args, "training_limit", 500),
    )
    print(
        f"[training] workspace={result['workspace_id']} | "
        f"feedback_written={result['feedback_written']} | "
        f"pairs_written={result['pairs_written']} | "
        f"output={result['output']}"
    )


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

    cache_run_key = args.run_id or (model if (args.cache_by_model and model) else "default")

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
        print(f"Limits aktiv: Datei={max_chars_per_file} Zeichen, Kontext={max_context_chars} Zeichen")

    if args.batch_size is not None:
        set_batch_size(args.batch_size)
    if args.batch_delay is not None:
        set_batch_delay(args.batch_delay)

    if not repo_url and not (args.ingest_issues or args.ingest_images or args.pi_task or args.generate_wiki):
        print("Fehler: Kein Repository angegeben. Nutze --repo oder setze GITHUB_REPO in .env")
        sys.exit(1)

    if args.reset:
        removed = reset_repo(repo_url, run_key=(cache_run_key if args.run_id or args.cache_by_model else None))
        print(f"Cache gelöscht: {removed}" if removed else "Kein Cache vorhanden — nichts zu löschen.")
        if removed:
            print("Nächster Lauf analysiert alle Dateien von vorn.")
        sys.exit(0)

    # Read-only / utility commands
    if args.history:              _cmd_history(args, repo_url);             sys.exit(0)
    if args.compare:              _cmd_compare(args, repo_url);              sys.exit(0)
    if args.cleanup is not None:  _cmd_cleanup(args, repo_url);             sys.exit(0)

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
        print(f"\n{'='*60}\n  FULL SCAN — Cache wird ignoriert, kein Datei-Limit\n{'='*60}\n")

    _wid_for_ops = getattr(args, "workspace_id", "default") or "default"
    if _wid_for_ops == "default":
        print(
            "Warnung: --workspace-id nicht gesetzt, Daten landen im 'default'-Workspace. "
            "Bei Produktionsnutzung bitte explizit angeben: --workspace-id <id>"
        )

    # Pipeline commands
    if args.populate_memory:                           run_populate_memory(args, repo_url, ollama_url, model); sys.exit(0)
    if args.generate_wiki:                             _cmd_generate_wiki(args, repo_url, ollama_url, model);  sys.exit(0)
    if getattr(args, "wiki_history", False):           _cmd_wiki_history(args);                               sys.exit(0)
    if getattr(args, "wiki_team_activity", False):     _cmd_wiki_team_activity(args);                         sys.exit(0)
    if getattr(args, "wiki_history_cleanup", None) is not None: _cmd_wiki_history_cleanup(args);              sys.exit(0)
    if args.rebuild_transitions:                       _cmd_rebuild_transitions(args, repo_url);               sys.exit(0)
    if args.generate_ambient:                          _cmd_generate_ambient(args, repo_url, ollama_url, model); sys.exit(0)
    if args.ingest_issues:                             run_ingest_issues(args, ollama_url, model);             sys.exit(0)
    if args.ingest_images:                             run_ingest_images(args, ollama_url, model);             sys.exit(0)
    if args.pi_task:                                   run_pi_task(args, ollama_url, model);                   sys.exit(0)
    if getattr(args, "generate_training_data", None):  _cmd_generate_training_data(args);                      sys.exit(0)
    if args.mode == "turbulence":                      _cmd_turbulence(args, repo_url, ollama_url);            sys.exit(0)

    start = time.perf_counter()
    run_analysis(args, repo_url, ollama_url, model, start)


if __name__ == "__main__":
    main()
