#!/usr/bin/env python3
"""UserPromptSubmit hook: warn-block bei CI-fail oder neuen security-alerts.

User-Auftrag (2026-05-11): "kannst du dir bei github actions einen webhook
trigger setzen, so dass wenn 1. irgendwas in der ci/cd pipeline scheitert
2. im security and quality bereich etwas aufploppt → IMMER bei dir eine
meldung erscheint, ich NICHT erst prompten muss".

100% "ohne user-prompt" geht nicht (claude-code session ist user-getrieben).
Aber: dieser hook läuft VOR jedem deiner prompts → bei rotem CI oder neuen
code-scanning-alerts injiziert er einen warning-block am promptanfang. Du
siehst es beim ersten zeichen das du tippst, kein extra-aktion nötig.

State-cache in ~/.config/mayring/ci_security_state.json damit "neu seit
letztem check" detection funktioniert (sonst spammt der hook bei jedem
prompt dieselben open-alerts).

Output: NUR wenn etwas problematisch — sonst silent skip.

Output-format (gets injected as prompt-prefix):
    ## ⚠️ CI/Security-Watch
    - **MayringCoder CI**: Post-deploy smoke FAILED (workflow_run 25640...)
    - **app.linn.games CodeQL**: 2 NEUE alerts (severity=warning)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_STATE_FILE = Path(os.path.expanduser("~/.config/mayring/ci_security_state.json"))
_REPOS = [
    "Nileneb/MayringCoder",
    "Nileneb/app.linn.games",
]
_WATCH_WORKFLOWS = {
    # repo → list of workflow-names to watch. None = ALL.
    "Nileneb/MayringCoder": None,
    "Nileneb/app.linn.games": None,
}
# WHY(2026-05-11): known-noise workflows. "Automatic Dependency Submission"
# ist GitHub's auto-dependency-graph-job (kein file im repo) — läuft auf
# jeden push inkl. ephemeral feature-branches, aber wir squash-merge +
# delete-branch sofort → die job kann die branch nicht mehr checkouten →
# "couldn't find remote ref" → fail. Harmlos (dependency-graph für master
# funktioniert), aber rauscht im hook. Hier rausfiltern — der hook soll
# nur ECHTE CI-fails (tests, deploy, smoke, build) melden.
_IGNORE_WORKFLOW_SUBSTRINGS = (
    "automatic dependency submission",
    "dependency submission",
)


def _gh(args: list[str], timeout: float = 8.0) -> dict | list | None:
    """Run gh CLI silently, return parsed JSON or None on failure."""
    try:
        out = subprocess.run(
            ["gh"] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        if out.returncode != 0:
            return None
        return json.loads(out.stdout) if out.stdout.strip() else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None


def _load_state() -> dict:
    try:
        with open(_STATE_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"failed_runs": {}, "alert_counts": {}}


def _save_state(state: dict) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except OSError:
        pass


def _check_ci(repo: str, state: dict) -> list[str]:
    """Return warning lines if CI has new failures since last check.

    WHY(2026-05-11): limit 10→30. Bei aktiven repos (deploy + smoke +
    build + linter + ingest pro merge) fielen failed runs aus dem
    limit-10-window bevor der hook sie sah → unbemerkte fails.
    """
    runs = _gh([
        "run", "list", "--repo", repo,
        "--limit", "30",
        "--json", "databaseId,name,conclusion,status,event",
    ])
    if not runs:
        return []
    warnings: list[str] = []
    seen_failed = state.setdefault("failed_runs", {}).setdefault(repo, [])
    new_failed = []
    for r in runs:
        if r.get("conclusion") == "failure" and r.get("status") == "completed":
            run_id = str(r.get("databaseId"))
            name_lc = str(r.get("name", "")).lower()
            # Skip known-noise workflows (branch-delete-race on auto-submission).
            if any(sub in name_lc for sub in _IGNORE_WORKFLOW_SUBSTRINGS):
                # Record it as seen so it doesn't keep getting re-evaluated,
                # but don't surface a warning.
                if run_id not in seen_failed:
                    new_failed.append(run_id)
                continue
            if run_id not in seen_failed:
                warnings.append(
                    f"- **{repo} CI**: `{r.get('name')}` FAILED "
                    f"(run {run_id}, trigger={r.get('event')})"
                )
                new_failed.append(run_id)
    if new_failed:
        # Append + truncate auf letzte 50 damit state-file nicht wächst.
        state["failed_runs"][repo] = (seen_failed + new_failed)[-50:]
    return warnings


def _check_security(repo: str, state: dict) -> list[str]:
    """Warn if open code-scanning alerts count went UP since last check."""
    alerts = _gh([
        "api", f"repos/{repo}/code-scanning/alerts?state=open",
        "--jq", "[.[] | {n:.number,rule:.rule.id,severity:.rule.severity}]",
    ])
    if alerts is None:
        return []
    cur_numbers = sorted(a.get("n") for a in alerts if a.get("n"))
    prev = state.setdefault("alert_counts", {}).get(repo, [])
    new_alerts = [n for n in cur_numbers if n not in prev]
    state["alert_counts"][repo] = cur_numbers
    if not new_alerts:
        return []
    severities = {n: next((a["severity"] for a in alerts if a.get("n") == n), "?")
                  for n in new_alerts}
    sev_str = ", ".join(f"#{n}({s})" for n, s in severities.items())
    return [
        f"- **{repo} Security**: {len(new_alerts)} NEUE alert(s): {sev_str}"
    ]


def main() -> int:
    state = _load_state()
    warnings: list[str] = []

    for repo in _REPOS:
        warnings.extend(_check_ci(repo, state))
        warnings.extend(_check_security(repo, state))

    _save_state(state)

    if warnings:
        # Output landet im prompt-context via UserPromptSubmit-hook-stdout.
        print("## ⚠️ CI/Security-Watch (neu seit letztem prompt)\n")
        for w in warnings:
            print(w)
        print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
