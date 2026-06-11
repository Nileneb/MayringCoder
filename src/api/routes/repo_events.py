"""POST /repo-events — the reusable GitHub Action posts repo push/CI/security
events here. Push re-ingests the newest version; CI/security are logged in
hook_events + a lightweight searchable repo_event chunk (recall + IGIO-Lens).

WHY(repo-watching C+D): closes the gap where only MayringCoder auto-ingested and
gives every watched repo's CI/security a memory presence."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import hashlib
import hmac

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials

from src.api import watch_store
from src.api.auth import _bearer, _is_privileged, _SERVICE_TOKEN
from src.api.github_oidc import verify_github_oidc
from src.api.jwt_auth import validate_jwt_token
from src.api.routes.jobs import enqueue_populate
from src.api.routes.models import RepoEventRequest
from src.api.dependencies import get_conn as _get_conn

router = APIRouter()


def repo_event_principal(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """Authorize a repo-event POST. Accepts (1) a GitHub Actions OIDC token from the
    allowed owner (secretless, preferred) or (2) the service/admin token (back-compat).
    Returns a short principal label for logging. OIDC grants ONLY repo-events — it is
    deliberately verified here, not in the global get_token_info (no admin elsewhere)."""
    if not creds:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing Bearer token")
    token = creds.credentials
    oidc = verify_github_oidc(token)
    if oidc is not None:
        return f"oidc:{oidc.get('repository', '?')}"
    if _SERVICE_TOKEN and hmac.compare_digest(token.encode(), _SERVICE_TOKEN.encode()):
        return "service"
    info = validate_jwt_token(token)
    if info and _is_privileged(info):
        return "admin"
    raise HTTPException(
        status.HTTP_403_FORBIDDEN,
        "repo-events requires GitHub OIDC (owner-gated) or a service/admin token",
    )


def _is_smoke_repo(repo: str) -> bool:
    """Smoke-Suite-Wegwerf-Repos (github.com/smoke/repo-<ts>) — gehören NIE in einen
    User-Workspace, sonst pollutet jeder Smoke-Lauf die Projekt-Sicht. → immer 'system'."""
    return "/smoke/repo-" in (repo or "").lower()


def _default_repo_workspace(conn, repo: str = "") -> str:
    """Default-Workspace für ein unbekanntes Repo. Smoke-Test-Repos → 'system' (Wegwerf).
    Sonst pre-launch single-user: der EINZIGE kind='user'-Workspace (alle echten Repos
    gehören dem einzigen MayringCoder-User). Mehrere User → 'system' (kein Raten)."""
    if _is_smoke_repo(repo):
        return "system"
    rows = conn.execute("SELECT id FROM workspaces WHERE kind='user'").fetchall()
    return rows[0][0] if len(rows) == 1 else "system"


def _resolve_project(conn, repo: str) -> tuple[str, str | None]:
    """repo-url → (workspace_id, project_id | None).

    Reuses _resolve_workspace to get/create the workspace + project row (match-or-create),
    then returns the project_id from that same row. project_id is None only when the repo
    string is empty/unparseable and canonical_repo_ref returns nothing."""
    from src.api.routes.projects import canonical_repo_ref
    canon = canonical_repo_ref(repo)
    ws = _resolve_workspace(conn, repo)
    if not canon:
        return ws, None
    row = conn.execute(
        "SELECT id FROM projects WHERE source_type='github' AND source_ref=?",
        (canon,),
    ).fetchone()
    return ws, (row[0] if row else None)


def _resolve_workspace(conn, repo: str) -> str:
    """repo-url → projects.workspace_id; match-or-create unter dem Default-User-Workspace.

    WHY(repo-watching, blackbox-fix 2026-05-29): mirrors projects.py match-or-create.
    Vorher defaultete ein unbekanntes Repo auf 'system' → seine CI/Security-Events waren im
    user-gescopten Dashboard UNSICHTBAR (alle Repo-Events landeten in 'system'). Jetzt:
    Default = einziger kind='user'-Workspace, damit die Events dort sichtbar werden.
    """
    from src.api.routes.projects import canonical_repo_ref
    canon = canonical_repo_ref(repo)  # EINE kanonische Form wie /projects/route → keine Dubletten
    row = conn.execute(
        "SELECT workspace_id FROM projects WHERE source_type='github' AND source_ref=?",
        (canon,),
    ).fetchone()
    if row is not None:
        return row[0]
    ws = _default_repo_workspace(conn, repo)
    now = datetime.now(timezone.utc).isoformat()
    pid = str(uuid.uuid4())
    name = canon.rsplit("/", 1)[-1]
    conn.execute(
        "INSERT INTO projects (id, workspace_id, name, source_type, source_ref, created_at, updated_at) "
        "VALUES (?, ?, ?, 'github', ?, ?, ?)",
        (pid, ws, name, canon, now, now),
    )
    conn.commit()
    return ws


_AXIS = {
    ("workflow_run", "failure"): "issue",
    ("workflow_run", "success"): "outcome",
    ("security", None): "issue",
}


def _record_repo_event(conn, workspace_id: str, hook_type: str, req: RepoEventRequest) -> None:
    """Log a CI/security event into hook_events (reuse payload JSON, NO migration).

    WHY(repo-watching): idempotent on the exact serialized payload so a
    GitHub re-delivery of the same event does not create a duplicate row.
    Exact-match (not LIKE) is used deliberately: it handles None sha/workflow
    correctly (json null), which a LIKE '%"workflow": "None"%' pattern would not."""
    payload = json.dumps({
        "repo": req.repo, "sha": req.sha, "ref": req.ref,
        "conclusion": req.conclusion, "workflow": req.workflow,
        "severity": req.severity, "summary": req.summary, "url": req.url,
    }, default=str)
    existing = conn.execute(
        "SELECT 1 FROM hook_events WHERE workspace_id=? AND hook_type=? AND payload=? LIMIT 1",
        (workspace_id, hook_type, payload),
    ).fetchone()
    if existing is not None:
        return
    conn.execute(
        "INSERT INTO hook_events (workspace_id, device_id, hook_type, fired_at, payload) "
        "VALUES (?, 'github-action', ?, ?, ?)",
        (workspace_id, hook_type, datetime.now(timezone.utc).isoformat(), payload),
    )
    conn.commit()


def _repo_event_chunk(conn, workspace_id: str, req: RepoEventRequest, axis: str) -> None:
    """Insert a lightweight, searchable source+chunk for a CI/security event,
    tagged with the deterministic igio axis (NO LLM in the hot path).

    WHY(repo-watching): gives every watched repo's CI/security a memory presence
    (recall + IGIO-Lens). igio_axis is written via update_chunk_igio_axis because
    insert_chunk does not persist igio columns.
    WHY(C3 project-link): after insert, links the chunk to the matching project so
    project_match boost fires in retrieval. Fail-soft: a missing project or DB error
    is logged and skipped — the chunk is always created regardless."""
    import logging as _log_re
    from mayring_core.memory.store import upsert_source, insert_chunk, update_chunk_igio_axis
    from mayring_core.memory.schema import Source, Chunk
    from src.api.routes.projects import canonical_repo_ref
    now = datetime.now(timezone.utc).isoformat()
    if req.event_type == "workflow_run":
        text = f"CI {req.workflow or ''} {req.conclusion or ''} on {req.repo}@{(req.sha or '')[:8]}".strip()
    else:
        text = f"Security {req.severity or ''}: {req.summary or ''} in {req.repo}".strip()
    sid = f"repo_event:{req.repo}:{req.event_type}:{(req.sha or now)[:12]}"
    thash = Chunk.compute_text_hash(text)   # already 'sha256:...'
    src = Source(
        source_id=sid, source_type="repo_event", repo=req.repo,
        path=req.url or "", branch=req.ref or "main", commit=req.sha or "",
        content_hash=thash, captured_at=now,
    )
    upsert_source(conn, src, workspace_id=workspace_id)
    chunk = Chunk(
        chunk_id=Chunk.make_id(sid, 0, "event"), source_id=sid, chunk_level="event",
        ordinal=0, text=text, text_hash=thash, created_at=now, workspace_id=workspace_id,
    )
    insert_chunk(conn, chunk, workspace_id=workspace_id)
    if axis:
        update_chunk_igio_axis(conn, chunk.chunk_id, axis, confidence=0.9)

    # Link chunk → project (C3 producer A)
    try:
        canon = canonical_repo_ref(req.repo)
        proj_row = conn.execute(
            "SELECT id FROM projects WHERE source_type='github' AND source_ref=?",
            (canon,),
        ).fetchone()
        if proj_row:
            from mayring_core.memory.store import link_chunk_to_project
            link_chunk_to_project(
                conn, chunk.chunk_id, proj_row[0],
                origin_ref=canon, source="repo-analyze",
                workspace_id=workspace_id,
            )
    except Exception as _exc:
        _log_re.getLogger(__name__).warning(
            "repo_event_chunk: project linking failed for %r: %s", req.repo, _exc
        )


def _handle_event(conn, workspace_id: str, req: RepoEventRequest) -> dict:
    """Shared event handling for both the OIDC /repo-events and the HMAC webhook.
    Workspace is decided by the CALLER (OIDC: resolve-from-repo; webhook: watch-record)."""
    if req.event_type == "push":
        job_id = enqueue_populate(req.repo, workspace_id)
        return {"ok": True, "action": "populate", "job_id": job_id, "workspace_id": workspace_id}

    hook_type = "repo_ci" if req.event_type == "workflow_run" else "repo_security"
    _record_repo_event(conn, workspace_id, hook_type, req)
    axis = _AXIS.get((req.event_type, req.conclusion)) or _AXIS.get((req.event_type, None)) or ""
    _repo_event_chunk(conn, workspace_id, req, axis)
    return {"ok": True, "action": hook_type, "workspace_id": workspace_id, "igio_axis": axis}


@router.post("/repo-events")
async def repo_events(
    req: RepoEventRequest,
    principal: str = Depends(repo_event_principal),
) -> dict:
    conn = _get_conn()
    workspace_id = _resolve_workspace(conn, req.repo)
    return _handle_event(conn, workspace_id, req)


def _translate_github_webhook(event: str, p: dict) -> RepoEventRequest | None:
    """Map a raw GitHub webhook payload to the internal RepoEventRequest. Returns
    None for events we deliberately ignore (ping, non-completed workflow_run, etc.)."""
    repo = (p.get("repository") or {}).get("full_name") or ""
    if event == "push":
        return RepoEventRequest(event_type="push", repo=repo,
                                sha=p.get("after"), ref=p.get("ref"))
    if event == "workflow_run":
        wr = p.get("workflow_run") or {}
        if (p.get("action") or "") != "completed":
            return None
        return RepoEventRequest(event_type="workflow_run", repo=repo,
                                sha=wr.get("head_sha"), ref=wr.get("head_branch"),
                                conclusion=wr.get("conclusion"), workflow=wr.get("name"),
                                url=wr.get("html_url"))
    if event in ("code_scanning_alert", "dependabot_alert"):
        alert = p.get("alert") or {}
        rule = alert.get("rule") or {}
        adv = alert.get("security_advisory") or {}
        sev = rule.get("severity") or adv.get("severity")
        summ = rule.get("description") or adv.get("summary") or alert.get("state") or ""
        return RepoEventRequest(event_type="security", repo=repo, severity=sev,
                                summary=summ, url=alert.get("html_url"))
    return None


@router.post("/repo-events/webhook")
async def repo_events_webhook(request: Request) -> dict:
    """GitHub webhook intake (HMAC-authenticated). The webhook is installed server-side
    by app.linn.games per active repo; its secret lives in the watch-record. We look up
    the repo's record, verify X-Hub-Signature-256, then run the shared handler in the
    watch-record's workspace (fixes the old 'events land in system' blackbox)."""
    body = await request.body()
    event = request.headers.get("X-GitHub-Event", "")
    # GitHub sends either content_type=json (raw JSON body) or content_type=form
    # (application/x-www-form-urlencoded with the JSON in a `payload=` field). Support
    # both so a form-configured hook gets a clean response, not a 400. NOTE: the HMAC
    # is always computed over the RAW body below, independent of this parsing.
    ctype = request.headers.get("Content-Type", "")
    if ctype.startswith("application/x-www-form-urlencoded"):
        from urllib.parse import parse_qs
        raw_json = (parse_qs(body.decode("utf-8", "replace")).get("payload") or ["{}"])[0]
    else:
        raw_json = body.decode("utf-8", "replace") or "{}"
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid JSON")

    repo = (payload.get("repository") or {}).get("full_name") or ""
    rec = watch_store.find_active_by_repo(repo)
    if rec is None or not rec.get("secret"):
        # Unknown/inactive repo, or no secret on file → cannot authenticate. Fail loud.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "repo not watched")

    sig = request.headers.get("X-Hub-Signature-256", "")
    expected = "sha256=" + hmac.new(rec["secret"].encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad signature")

    if event == "ping":
        return {"ok": True, "action": "ignored", "event": "ping"}
    req = _translate_github_webhook(event, payload)
    if req is None:
        return {"ok": True, "action": "ignored", "event": event}
    conn = _get_conn()
    return _handle_event(conn, rec["workspace_id"], req)
