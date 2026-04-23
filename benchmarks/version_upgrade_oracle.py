#!/usr/bin/env python3
"""Oracle-Memory-Injection: echte Developer-Diffs in Pi-Agent Memory.

Liest bestehende context_*.json Dateien, klont old+new Tag in einem Repo,
extrahiert git diff pro GT-Datei und injiziert als Memory-Dokument in den
Pi-Agent Workspace. Beim nächsten MayringCoder-Run findet der Pi-Agent
diese Diffs per search_memory und gibt sie an qwen weiter.

Der Pi-Agent sucht workspace-agnostisch — Oracle-Chunks werden sofort
gefunden ohne Code-Änderungen am Pi-Agent.

Usage:
    python benchmarks/version_upgrade_oracle.py \\
        --context-dir benchmarks/version_upgrade_results/ \\
        --workspace-id python_ecosystem_oracle

    python benchmarks/version_upgrade_oracle.py \\
        --repo psf/requests \\
        --max-diff-chars 800
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "benchmarks" / "version_upgrade_results"


def _fetch_both_tags(repo: str, old_tag: str, new_tag: str, dest: str) -> None:
    subprocess.run(["git", "init", dest], check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", f"https://github.com/{repo}"],
        cwd=dest, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "fetch", "origin",
         f"refs/tags/{old_tag}:refs/tags/{old_tag}",
         f"refs/tags/{new_tag}:refs/tags/{new_tag}"],
        cwd=dest, check=True, capture_output=True,
    )


def get_gt_diff_for_file(
    repo_dir: str,
    old_tag: str,
    new_tag: str,
    filepath: str,
    max_chars: int = 600,
) -> tuple[str, str]:
    """Returns (stat_header, diff_body) for a single file."""
    stat_result = subprocess.run(
        ["git", "diff", "--numstat", old_tag, new_tag, "--", filepath],
        cwd=repo_dir, capture_output=True, text=True, check=False,
    )
    added, removed = "?", "?"
    if stat_result.returncode == 0 and stat_result.stdout.strip():
        parts = stat_result.stdout.strip().split("\t")
        if len(parts) >= 2:
            added, removed = parts[0], parts[1]

    diff_result = subprocess.run(
        ["git", "diff", old_tag, new_tag, "--", filepath],
        cwd=repo_dir, capture_output=True, text=True, check=False,
    )
    if diff_result.returncode != 0:
        return f"+{added} / -{removed} lines", ""
    diff_body = diff_result.stdout[:max_chars]
    if len(diff_result.stdout) > max_chars:
        diff_body += "\n... (truncated)"
    return f"+{added} / -{removed} lines", diff_body


def build_oracle_document(ctx: dict, repo_dir: str, max_diff_chars: int = 600) -> str:
    repo = ctx["repo"]
    old_tag = ctx["old_tag"]
    new_tag = ctx["new_tag"]
    gt_files = ctx.get("gt_files", [])

    lines = [
        f"# {repo} — Oracle: Tatsächliche Änderungen {old_tag} → {new_tag}",
        f"",
        f"Echte Developer-Änderungen beim Major-Upgrade. Quelle: git diff {old_tag}..{new_tag}",
        f"",
    ]
    for filepath in sorted(gt_files):
        stat, diff_body = get_gt_diff_for_file(repo_dir, old_tag, new_tag, filepath, max_diff_chars)
        lines += [
            f"### {filepath}  ({stat})",
            f"```diff",
            diff_body,
            f"```",
            f"",
        ]
    return "\n".join(lines)


def inject_oracle_doc(doc: str, repo: str, old_tag: str, workspace_id: str) -> bool:
    repo_safe = repo.replace("/", "__")
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = f"{repo_safe}_{old_tag}_oracle_diffs.md"
        (Path(tmpdir) / fname).write_text(doc, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "src.cli",
             "--repo", tmpdir,
             "--populate-memory",
             "--workspace-id", workspace_id,
             "--no-limit"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
    return result.returncode == 0


def build_oracle_for_context(ctx: dict, workspace_id: str, max_diff_chars: int = 600) -> bool:
    repo = ctx["repo"]
    old_tag = ctx["old_tag"]
    new_tag = ctx["new_tag"]
    with tempfile.TemporaryDirectory() as tmpdir:
        _fetch_both_tags(repo, old_tag, new_tag, tmpdir)
        doc = build_oracle_document(ctx, tmpdir, max_diff_chars)
    return inject_oracle_doc(doc, repo, old_tag, workspace_id)


def load_contexts(context_dir: str) -> dict[str, dict]:
    contexts = {}
    for f in Path(context_dir).glob("context_*.json"):
        try:
            ctx = json.loads(f.read_text())
            contexts[ctx["repo"]] = ctx
        except (KeyError, json.JSONDecodeError):
            pass
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(description="Version-Upgrade Oracle: Diff-Memory-Injection")
    parser.add_argument("--context-dir", default=str(RESULTS_DIR))
    parser.add_argument("--workspace-id", default="python_ecosystem_oracle")
    parser.add_argument("--max-diff-chars", type=int, default=600)
    parser.add_argument("--repo", default=None, help="Filter auf einzelnes Repo, z.B. psf/requests")
    args = parser.parse_args()

    contexts = load_contexts(args.context_dir)
    if not contexts:
        print(f"Keine context_*.json in {args.context_dir}")
        sys.exit(1)

    if args.repo:
        contexts = {k: v for k, v in contexts.items() if k == args.repo}
        if not contexts:
            print(f"Repo '{args.repo}' nicht in context_*.json gefunden")
            sys.exit(1)

    print(f"Oracle-Injection für {len(contexts)} Repo(s) → workspace '{args.workspace_id}'")
    for repo, ctx in contexts.items():
        gt_count = len(ctx.get("gt_files", []))
        print(f"  [{repo}] {ctx['old_tag']}→{ctx['new_tag']} ({gt_count} GT-Dateien)...")
        ok = build_oracle_for_context(ctx, args.workspace_id, args.max_diff_chars)
        print(f"  [{repo}] {'OK' if ok else 'FEHLER'}")

    print(f"\nNächster Schritt:")
    print(f"  python benchmarks/version_upgrade_prep.py --workspace-id {args.workspace_id}")


if __name__ == "__main__":
    main()
