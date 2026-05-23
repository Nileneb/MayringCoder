"""Integrations + Notifications routes (#270 Phase 1).

- POST /integrations/github/webhook — empfängt GitHub-Webhooks, klassifiziert
  relevante Events und persistiert sie als Notifications (serverseitige Ablösung
  von ci_security_warner + Push-Auto-Ingest-Trigger).
- GET  /notifications?since=<iso> — liefert nicht-quittierte Events; vom
  UserPromptSubmit-Hook in EINEM Call gepollt (statt N Poll-Runden).
- POST /notifications/ack — quittiert Events.

GitHub-Webhook-Konfiguration + Hook-Polling sind cross-repo/extern, siehe
docs/v2-architecture-270.md.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request

from src.api.dependencies import get_conn as _get_conn
from src.api.integrations import github_events, notifications_store

router = APIRouter(tags=["integrations"])


@router.post("/integrations/github/webhook")
async def github_webhook(
    request: Request,
    x_github_event: str = Header(default=""),
) -> dict:
    payload = await request.json()
    notif = github_events.classify_github_event(x_github_event, payload)
    if notif is None:
        return {"recorded": False, "event": x_github_event}
    conn = _get_conn()
    nid = notifications_store.record_notification(
        conn,
        kind=notif["kind"],
        severity=notif["severity"],
        title=notif["title"],
        detail=notif["detail"],
        repo=notif["repo"],
        payload={"event": x_github_event},
    )
    return {"recorded": True, "id": nid, "kind": notif["kind"]}


@router.get("/notifications")
async def list_notifications(since: str | None = None) -> dict:
    conn = _get_conn()
    items = notifications_store.unacked_since(conn, since)
    return {"notifications": items, "count": len(items)}


@router.post("/notifications/ack")
async def ack_notifications(ids: list[int]) -> dict:
    conn = _get_conn()
    n = notifications_store.ack(conn, ids)
    return {"acked": n}
