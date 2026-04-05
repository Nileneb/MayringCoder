"""Run-History — persist each analysis run as a JSON snapshot for comparison."""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import CACHE_DIR, repo_slug as _repo_slug


def _runs_dir(repo_url: str) -> Path:
    """Return ``cache/{repo_slug}/runs/`` directory for the given repo."""
    return CACHE_DIR / _repo_slug(repo_url) / "runs"


def generate_run_id() -> str:
    """Generate a run-ID like ``20260402-143012``."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _build_run_payload(
    run_id: str,
    repo_url: str,
    model: str,
    mode: str,
    results: list[dict],
    diff: dict,
    timing: float,
    aggregation: dict | None,
    extra: dict | None = None,
) -> dict:
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo_url": repo_url,
        "model": model,
        "mode": mode,
        "timing_seconds": round(timing, 2),
        "files_checked": len(results),
        "diff_summary": {
            "changed": len(diff.get("changed", [])),
            "added": len(diff.get("added", [])),
            "removed": len(diff.get("removed", [])),
            "unchanged": len(diff.get("unchanged", [])),
        },
        "results": results,
    }
    if aggregation is not None:
        payload["aggregation"] = aggregation
    if extra:
        payload.update(extra)
    return payload


def save_run(
    run_id: str,
    repo_url: str,
    model: str,
    mode: str,
    results: list[dict],
    diff: dict,
    timing: float,
    aggregation: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Persist a single run as ``{runs_dir}/{run_id}.json``.

    Handles collisions by appending ``_1``, ``_2``, … if the ID already exists.
    """
    runs = _runs_dir(repo_url)
    runs.mkdir(parents=True, exist_ok=True)

    out = runs / f"{run_id}.json"
    counter = 1
    while out.exists():
        out = runs / f"{run_id}_{counter}.json"
        counter += 1

    payload = _build_run_payload(out.stem, repo_url, model, mode, results, diff, timing, aggregation, extra)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def list_runs(repo_url: str) -> list[dict]:
    """Return lightweight metadata for all stored runs, newest first."""
    runs = _runs_dir(repo_url)
    if not runs.exists():
        return []

    entries: list[dict] = []
    for p in sorted(runs.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        entries.append({
            "run_id": data.get("run_id", p.stem),
            "timestamp": data.get("timestamp", ""),
            "mode": data.get("mode", "?"),
            "model": data.get("model", "?"),
            "files_checked": data.get("files_checked", 0),
            "timing_seconds": data.get("timing_seconds", 0),
        })
    return entries


def load_run(run_id: str, repo_url: str) -> dict | None:
    """Load the full payload for a given run-ID. Returns *None* if not found."""
    runs = _runs_dir(repo_url)
    path = runs / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def compare_runs(run_id_a: str, run_id_b: str, repo_url: str) -> dict:
    """Compare two runs and return new, resolved, and severity-changed findings.

    ``run_id_a`` is treated as the *older* run, ``run_id_b`` as the *newer* one.
    Returns a dict with ``new``, ``resolved``, ``severity_changed``, and ``summary``.
    """
    a = load_run(run_id_a, repo_url)
    b = load_run(run_id_b, repo_url)

    if a is None:
        raise FileNotFoundError(f"Run '{run_id_a}' nicht gefunden")
    if b is None:
        raise FileNotFoundError(f"Run '{run_id_b}' nicht gefunden")

    def _key(finding: dict, filename: str) -> str:
        return f"{filename}::{finding.get('type', '')}::{finding.get('evidence_excerpt', '')[:80]}"

    def _extract(run_data: dict) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in run_data.get("results", []):
            fn = r.get("filename", "")
            for smell in r.get("potential_smells", []):
                k = _key(smell, fn)
                out[k] = {**smell, "_filename": fn}
        return out

    old_map = _extract(a)
    new_map = _extract(b)

    old_keys = set(old_map)
    new_keys = set(new_map)

    added = [new_map[k] for k in sorted(new_keys - old_keys)]
    resolved = [old_map[k] for k in sorted(old_keys - new_keys)]

    severity_changed: list[dict] = []
    for k in sorted(old_keys & new_keys):
        old_sev = old_map[k].get("severity", "info")
        new_sev = new_map[k].get("severity", "info")
        if old_sev != new_sev:
            severity_changed.append({
                **new_map[k],
                "_old_severity": old_sev,
                "_new_severity": new_sev,
            })

    return {
        "run_a": run_id_a,
        "run_b": run_id_b,
        "new": added,
        "resolved": resolved,
        "severity_changed": severity_changed,
        "summary": {
            "new_count": len(added),
            "resolved_count": len(resolved),
            "severity_changed_count": len(severity_changed),
            "files_a": a.get("files_checked", 0),
            "files_b": b.get("files_checked", 0),
        },
    }


def cleanup_runs(repo_url: str, keep: int = 10) -> int:
    """Delete old runs, keeping the *keep* newest. Returns number of deleted runs."""
    runs = _runs_dir(repo_url)
    if not runs.exists():
        return 0

    all_files = sorted(runs.glob("*.json"), reverse=True)
    to_delete = all_files[keep:]

    for p in to_delete:
        p.unlink()

    return len(to_delete)
