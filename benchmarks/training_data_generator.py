#!/usr/bin/env python3
"""Generates training data from version-upgrade benchmark results.

Two outputs:
  1. JSONL pairs for future qwen fine-tuning (upgrade prediction task)
  2. Memory injection into python_ecosystem workspace so Pi-agent knows
     which files are high-priority in known packages

Usage:
    python benchmarks/training_data_generator.py
    python benchmarks/training_data_generator.py --context-dir benchmarks/version_upgrade_results/
    python benchmarks/training_data_generator.py --haiku-output benchmarks/.../haiku_output_*.json
    python benchmarks/training_data_generator.py --inject-memory
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.version_upgrade_utils import compute_metrics, filter_python_files

RESULTS_DIR = ROOT / "benchmarks" / "version_upgrade_results"
TRAINING_DIR = ROOT / "benchmarks" / "training_data"


def load_all_contexts(context_dir: str) -> dict[str, dict]:
    contexts = {}
    for f in Path(context_dir).glob("context_*.json"):
        try:
            ctx = json.loads(f.read_text())
            contexts[ctx["repo"]] = ctx
        except (KeyError, json.JSONDecodeError):
            pass
    return contexts


def load_haiku_outputs(context_dir: str) -> list[dict]:
    """Loads all haiku_output_*.json files, merges by repo (latest wins)."""
    merged: dict[str, dict] = {}
    for f in sorted(Path(context_dir).glob("haiku_output_*.json")):
        try:
            entries = json.loads(f.read_text())
            for e in entries:
                merged[e["repo"]] = e
        except (KeyError, json.JSONDecodeError):
            pass
    return list(merged.values())


def generate_upgrade_predict_pairs(contexts: dict[str, dict]) -> list[dict]:
    """Generates prompt→completion pairs for the upgrade-prediction task.

    The model learns: given file summaries → identify files needing improvement.
    Ground truth = actual developer changes between versions.
    """
    pairs = []
    for repo, ctx in contexts.items():
        prompt_summary = ctx.get("prompt_summary", "")
        gt_files = ctx.get("gt_files", [])
        if not prompt_summary or not gt_files:
            continue

        prompt = (
            f"Du analysierst die Python-Bibliothek `{repo}` Version `{ctx['old_tag']}`.\n\n"
            f"MayringCoder hat die Codebase analysiert:\n{prompt_summary}\n\n"
            f"Welche Quelldateien brauchen am dringendsten Verbesserung/Refactoring "
            f"für ein Major-Version-Upgrade? Berücksichtige: API-Konsistenz, "
            f"Fehlerbehandlung, Code-Qualität, Wartbarkeit, Python 2/3 Kompatibilität.\n\n"
            f"Antworte NUR mit einem JSON-Array von Dateipfaden."
        )
        completion = json.dumps(sorted(gt_files), ensure_ascii=False)

        pairs.append({
            "type": "upgrade_prediction",
            "repo": repo,
            "old_tag": ctx["old_tag"],
            "new_tag": ctx["new_tag"],
            "prompt": prompt,
            "completion": completion,
            "gt_count": len(gt_files),
        })
    return pairs


def _extract_snippet_for_file(fname: str, prompt_summary: str) -> str:
    for line in prompt_summary.split("\n"):
        if line.startswith(f"- {fname} ["):
            return line[2:]
    return fname


def generate_file_importance_pairs(contexts: dict[str, dict]) -> list[dict]:
    """Per-file training pairs: given file summary → importance verdict.

    Uses shown_files (files that actually appeared in the Haiku prompt) when available,
    so labels only cover files the model could have seen — avoiding noise from
    GT files that were never shown.
    """
    pairs = []
    for repo, ctx in contexts.items():
        gt_set = set(ctx.get("gt_files", []))
        prompt_summary = ctx.get("prompt_summary", "")
        shown_files = ctx.get("shown_files")

        if not prompt_summary:
            continue

        if shown_files is not None:
            # Accurate path: iterate exactly the files that appeared in the prompt
            for fname in shown_files:
                is_gt = fname in gt_set
                snippet = _extract_snippet_for_file(fname, prompt_summary)
                pairs.append({
                    "type": "file_importance",
                    "repo": repo,
                    "old_tag": ctx["old_tag"],
                    "filename": fname,
                    "prompt": (
                        f"Bewerte die Verbesserungsnotwendigkeit dieser Datei in `{repo}` "
                        f"Version `{ctx['old_tag']}`:\n{snippet}\n\n"
                        f"Antworte mit: high | medium | low"
                    ),
                    "completion": "high" if is_gt else "low",
                    "is_gt": is_gt,
                })
        else:
            # Legacy fallback: parse prompt_summary lines directly
            for line in prompt_summary.split("\n"):
                if not line.startswith("- "):
                    continue
                try:
                    fname = line[2:line.index(" [")]
                except ValueError:
                    continue
                is_gt = fname in gt_set
                pairs.append({
                    "type": "file_importance",
                    "repo": repo,
                    "old_tag": ctx["old_tag"],
                    "filename": fname,
                    "prompt": (
                        f"Bewerte die Verbesserungsnotwendigkeit dieser Datei in `{repo}` "
                        f"Version `{ctx['old_tag']}`:\n{line[2:]}\n\n"
                        f"Antworte mit: high | medium | low"
                    ),
                    "completion": "high" if is_gt else "low",
                    "is_gt": is_gt,
                })
    return pairs


def build_memory_document(ctx: dict) -> str:
    """Creates a markdown document for Pi-agent memory injection."""
    repo = ctx["repo"]
    old_tag = ctx["old_tag"]
    new_tag = ctx["new_tag"]
    gt_files = ctx.get("gt_files", [])
    prompt_summary = ctx.get("prompt_summary", "")

    # Build file descriptions from prompt_summary
    file_desc: dict[str, str] = {}
    for line in prompt_summary.split("\n"):
        if not line.startswith("- "):
            continue
        try:
            fname = line[2:line.index(" [")]
            desc = line[line.index("]: ") + 3:][:150]
            file_desc[fname] = desc
        except ValueError:
            continue

    lines = [
        f"# {repo} — Version Upgrade Analysis ({old_tag} → {new_tag})",
        f"",
        f"Tatsächlich geänderte Quelldateien im Major-Upgrade (Ground Truth aus git diff):",
        f"",
    ]
    for f in sorted(gt_files):
        desc = file_desc.get(f, "")
        lines.append(f"- **{f}**: {desc}" if desc else f"- **{f}**")

    lines += [
        f"",
        f"## Fazit",
        f"Beim Upgrade von `{old_tag}` auf `{new_tag}` wurden {len(gt_files)} Quelldateien "
        f"geändert. Die kritischsten Dateien sind die Kernmodule (sessions, models, adapters, "
        f"utils, auth) sowie die gebündelten urllib3-Komponenten.",
    ]
    return "\n".join(lines)


def inject_into_memory(ctx: dict, workspace_id: str = "python_ecosystem") -> bool:
    doc = build_memory_document(ctx)
    repo_safe = ctx["repo"].replace("/", "__")
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = f"{repo_safe}_{ctx['old_tag']}_upgrade_insights.md"
        doc_path = Path(tmpdir) / fname
        doc_path.write_text(doc)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Version-Upgrade Training Data Generator")
    parser.add_argument("--context-dir", default=str(RESULTS_DIR))
    parser.add_argument("--haiku-output", default=None,
                        help="Optional: haiku output JSON for TP/FN stats")
    parser.add_argument("--inject-memory", action="store_true",
                        help="Inject GT insights into python_ecosystem Pi-agent memory")
    parser.add_argument("--ecosystem-workspace", default="python_ecosystem")
    parser.add_argument("--output-dir", default=str(TRAINING_DIR))
    args = parser.parse_args()

    contexts = load_all_contexts(args.context_dir)
    if not contexts:
        print(f"No context JSONs found in {args.context_dir}")
        sys.exit(1)
    print(f"Loaded {len(contexts)} context(s): {', '.join(contexts.keys())}")

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # 1. Upgrade prediction pairs (GT as completion)
    upgrade_pairs = generate_upgrade_predict_pairs(contexts)
    upgrade_path = TRAINING_DIR / f"upgrade_predict_{ts}.jsonl"
    upgrade_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in upgrade_pairs))
    print(f"\nUpgrade-Prediction pairs: {len(upgrade_pairs)} → {upgrade_path}")

    # 2. File importance pairs
    importance_pairs = generate_file_importance_pairs(contexts)
    importance_path = TRAINING_DIR / f"file_importance_{ts}.jsonl"
    importance_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in importance_pairs))
    gt_pos = sum(1 for p in importance_pairs if p["is_gt"])
    gt_neg = len(importance_pairs) - gt_pos
    print(f"File-Importance pairs: {len(importance_pairs)} ({gt_pos} high, {gt_neg} low) → {importance_path}")

    # 3. TP/FN breakdown if haiku output available
    haiku_outputs = load_haiku_outputs(args.context_dir)
    if haiku_outputs:
        print(f"\nTP/FN Analyse:")
        haiku_by_repo = {e["repo"]: e for e in haiku_outputs}
        for repo, ctx in contexts.items():
            entry = haiku_by_repo.get(repo)
            if not entry:
                continue
            suggested = filter_python_files(entry.get("suggested_files", []))
            gt = set(ctx["gt_files"])
            m = compute_metrics(gt, suggested)
            tp_files = sorted(gt & suggested)
            fn_files = sorted(gt - suggested)
            fp_files = sorted(suggested - gt)
            print(f"\n  {repo} ({ctx['old_tag']}→{ctx['new_tag']}):")
            print(f"    Recall={m['recall']:.3f} Precision={m['precision']:.3f} F1={m['f1']:.3f}")
            print(f"    TP ({len(tp_files)}): {tp_files[:5]}{'...' if len(tp_files) > 5 else ''}")
            print(f"    FN ({len(fn_files)}): {fn_files[:5]}{'...' if len(fn_files) > 5 else ''}")
            if fp_files:
                print(f"    FP ({len(fp_files)}): {fp_files[:5]}")

    # 4. Memory injection
    if args.inject_memory:
        print(f"\nInjecting insights into workspace '{args.ecosystem_workspace}'...")
        for repo, ctx in contexts.items():
            ok = inject_into_memory(ctx, args.ecosystem_workspace)
            status = "✓" if ok else "✗"
            print(f"  {status} {repo}")

    print(f"\nTraining data saved to: {TRAINING_DIR}")
    print(f"Total JSONL pairs: {len(upgrade_pairs) + len(importance_pairs)}")


if __name__ == "__main__":
    main()
