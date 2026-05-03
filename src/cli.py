"""MayringCoder Pipeline — Direkt-Executor (kein HTTP).

Wird vom API-Server als subprocess gestartet (src.api.job_queue.run_checker_job).
Greift direkt auf memory.db, chromadb und ollama zu. Keine Remote-Auth.

src/pipeline.py — BC-Shim, re-exportiert dieses Modul unter altem Namen.
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
    from src.api.dependencies import get_conn
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
        detector = EdgeDetector(ollama_url=ollama_url, model=model)
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


def _cmd_classify_igio(args: argparse.Namespace, ollama_url: str, model: str) -> None:
    """Backfill IGIO axes for unclassified active chunks.

    Reads `chunks` rows where `igio_axis = ''` and `is_active = 1`, runs
    `classify_chunk()` per row, persists the verdict back to the row when the
    confidence clears `--igio-min-confidence`.

    DBAdapter wraps sqlite3 with `row_factory = sqlite3.Row`, so rows are
    Mapping-like and we always use key access.
    """
    from src.config import CACHE_DIR
    from src.memory.store import init_memory_db
    from src.wiki_v2.igio_classifier import classify_chunk, now_iso

    if not model:
        print("Fehler: --classify-igio braucht ein --model.")
        return

    limit = max(1, int(getattr(args, "igio_limit", 200)))
    threshold = float(getattr(args, "igio_min_confidence", 0.5))
    workspace = getattr(args, "workspace_id", None)

    conn = init_memory_db(CACHE_DIR / "memory.db")
    where = ["igio_axis = ''", "is_active = 1", "text != ''"]
    params: list = []
    if workspace:
        where.append("workspace_id = ?")
        params.append(workspace)
    sql = (
        f"SELECT chunk_id, text, category_labels FROM chunks "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY created_at DESC LIMIT ?"
    )
    params.append(limit)
    rows = conn.execute(sql, tuple(params)).fetchall()
    if not rows:
        print(f"[igio] Nichts zu klassifizieren (workspace={workspace or 'all'}).")
        conn.close()
        return

    print(f"[igio] {len(rows)} Chunks → Klassifikator (model={model})")
    counts: dict[str, int] = {a: 0 for a in ("issue", "goal", "intervention", "outcome", "")}
    persisted = 0
    for r in rows:
        cats = [c for c in (r["category_labels"] or "").split(",") if c]
        verdict = classify_chunk(
            r["text"], cats, ollama_url=ollama_url, model=model,
        )
        counts[verdict.axis] = counts.get(verdict.axis, 0) + 1
        if verdict.axis and verdict.confidence >= threshold:
            conn.execute(
                "UPDATE chunks SET igio_axis = ?, igio_confidence = ?, "
                "igio_classified_at = ? WHERE chunk_id = ?",
                (verdict.axis, verdict.confidence, now_iso(), r["chunk_id"]),
            )
            persisted += 1

    conn.commit()
    conn.close()
    print(
        f"[igio] persisted={persisted}/{len(rows)} | "
        f"issue={counts.get('issue',0)} goal={counts.get('goal',0)} "
        f"intervention={counts.get('intervention',0)} outcome={counts.get('outcome',0)} "
        f"unclassified={counts.get('',0)}"
    )


def _cmd_generate_recap(args: argparse.Namespace) -> None:
    """Render a markdown recap for a single issue id."""
    from src.config import CACHE_DIR, WIKI_DIR
    from src.memory.store import init_memory_db
    from src.wiki_v2.recap_indexer import build_recap
    from src.wiki_v2.recap_renderer import render_recap

    issue_id = str(getattr(args, "generate_recap", "") or "").strip().lstrip("#")
    if not issue_id:
        print("Fehler: --generate-recap erwartet eine Issue-ID.")
        return
    workspace = getattr(args, "workspace_id", None) or "default"

    conn = init_memory_db(CACHE_DIR / "memory.db")
    recap = build_recap(issue_id, conn=conn, workspace_id=workspace)
    conn.close()

    md = render_recap(recap)
    out_arg = getattr(args, "recap_out", None)
    out_path = Path(out_arg) if out_arg else (WIKI_DIR / workspace / f"recap-{issue_id}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(
        f"[recap] issue=#{issue_id} workspace={workspace} → {out_path} "
        f"(issue:{len(recap.issue_chunks)} interv:{len(recap.intervention_chunks)} "
        f"out:{len(recap.outcome_chunks)} plans:{len(recap.plans)} "
        f"commits:{len(recap.commits)})"
    )


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
    if args.model and not args.llm:
        args.llm = True
    turb_model = (
        resolve_model(ollama_url, cli_model=args.model, env_model=os.getenv("TURB_MODEL"))
        if args.llm
        else (os.getenv("TURB_MODEL", "mistral:7b-instruct"))
    )
    os.environ["OLLAMA_URL"] = ollama_url
    os.environ["TURB_MODEL"] = turb_model
    run_turbulence(args, repo_url, ollama_url, turb_model)


def _cmd_generate_training_data(args: argparse.Namespace) -> None:
    wid = getattr(args, "workspace_id", "default") or "default"
    pipeline = args.generate_training_data

    if pipeline == "kategorie":
        from src.training.kategorie_coaching import run as run_kategorie, DEFAULT_OUTPUT as KAT_OUT
        result = run_kategorie(
            workspace_id=wid,
            output_path=KAT_OUT,
            limit=getattr(args, "training_limit", 1000),
        )
        print(
            f"[training:kategorie] workspace={result['workspace_id']} | "
            f"pairs_written={result['pairs_written']} | "
            f"output={result['output']}"
        )
    else:
        from src.training.memory_context_generator import run as run_memory_gen, DEFAULT_OUTPUT
        result = run_memory_gen(
            workspace_id=wid,
            output_path=DEFAULT_OUTPUT,
            skip_feedback=getattr(args, "skip_auto_feedback", False),
            limit=getattr(args, "training_limit", 500),
        )
        print(
            f"[training:memory] workspace={result['workspace_id']} | "
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
    _is_non_llm_path = (
        getattr(args, "populate_memory", False)
        or getattr(args, "generate_training_data", None)
    )
    _needs_llm = not _is_non_llm_path and (
        args.mode in ("analyze", "overview")
        or (args.mode == "turbulence" and args.llm)
        or args.resolve_model_only
        or getattr(args, "classify_igio", False)
    )
    model = resolve_model(ollama_url, args.model, None) if _needs_llm else (args.model or "")

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

    if not repo_url and not (args.ingest_issues or args.ingest_images or args.pi_task or args.generate_wiki
                              or getattr(args, "classify_igio", False)
                              or getattr(args, "generate_recap", None)
                              or getattr(args, "generate_training_data", None)):
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
    if getattr(args, "classify_igio", False):          _cmd_classify_igio(args, ollama_url, model);            sys.exit(0)
    if getattr(args, "generate_recap", None):          _cmd_generate_recap(args);                              sys.exit(0)
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
