"""GitHub-Webhook → Notification-Klassifikation (#270 Phase 1).

Serverseitige Ablösung des plugin-seitigen ci_security_warner: statt bei jedem
Prompt GitHub-APIs zu pollen, empfängt der Cloud-MCP GitHub-Webhooks und
persistiert relevante Events als Notifications. Diese reine Funktion mappt einen
(event_type, payload) auf ein Notification-Dict oder None (ignorieren).

Webhook-Konfiguration auf GitHub-Seite + das Hook-Polling (UserPromptSubmit)
sind cross-repo/extern — siehe docs/v2-architecture-270.md.
"""
from __future__ import annotations

from typing import Any

# Event-Typen die als Security-relevant gelten (→ severity 'high').
_SECURITY_EVENTS = {
    "code_scanning_alert",
    "secret_scanning_alert",
    "security_advisory",
    "dependabot_alert",
    "repository_vulnerability_alert",
}


def classify_github_event(event_type: str, payload: dict[str, Any]) -> dict | None:
    """Map a GitHub webhook to a notification dict, or None to ignore.

    Returns: {kind, severity, title, detail, repo} for events worth surfacing.
    """
    repo = (payload.get("repository") or {}).get("full_name", "")

    if event_type in _SECURITY_EVENTS:
        alert = payload.get("alert") or {}
        sev = (
            alert.get("severity")
            or (alert.get("security_advisory") or {}).get("severity")
            or "high"
        )
        return {
            "kind": "security",
            "severity": sev,
            "title": f"Security alert ({event_type})",
            "detail": alert.get("html_url") or payload.get("action", ""),
            "repo": repo,
        }

    if event_type == "workflow_run":
        run = payload.get("workflow_run") or {}
        # Nur abgeschlossene, fehlgeschlagene Runs sind eine Notification wert.
        if run.get("status") == "completed" and run.get("conclusion") not in (
            "success", "skipped", "neutral", None,
        ):
            return {
                "kind": "ci_failure",
                "severity": "medium",
                "title": f"CI failed: {run.get('name', 'workflow')}",
                "detail": run.get("html_url", ""),
                "repo": repo,
            }
        return None

    if event_type == "push":
        ref = payload.get("ref", "")
        # Push auf den Default-Branch triggert downstream Auto-Ingest (v2-chain).
        default_branch = (payload.get("repository") or {}).get("default_branch", "")
        if default_branch and ref.endswith(f"/{default_branch}"):
            return {
                "kind": "push",
                "severity": "info",
                "title": f"Push to {default_branch}",
                "detail": payload.get("after", ""),
                "repo": repo,
            }
        return None

    if event_type == "pull_request" and payload.get("action") in ("opened", "reopened", "ready_for_review"):
        pr = payload.get("pull_request") or {}
        return {
            "kind": "pr",
            "severity": "info",
            "title": f"PR #{pr.get('number', '?')}: {pr.get('title', '')}",
            "detail": pr.get("html_url", ""),
            "repo": repo,
        }

    return None
