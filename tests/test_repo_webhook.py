import hashlib
import hmac
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _sig(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _client(monkeypatch):
    import src.api.routes.repo_events as re_mod
    monkeypatch.setattr(re_mod.watch_store, "find_active_by_repo",
                        lambda slug: {"workspace_id": "wsZ", "secret": "topsecret"}
                        if slug.lower() == "nileneb/foo" else None)
    calls = {}

    def fake_handle(conn, ws, req):
        calls["call"] = (ws, req.event_type)
        return {"ok": True, "workspace_id": ws}

    monkeypatch.setattr(re_mod, "_handle_event", fake_handle)
    monkeypatch.setattr(re_mod, "_get_conn", lambda: None)
    app = FastAPI()
    app.include_router(re_mod.router)
    return TestClient(app), calls


def test_push_webhook_valid_signature(monkeypatch):
    client, calls = _client(monkeypatch)
    payload = {"repository": {"full_name": "Nileneb/Foo"},
               "after": "abc123", "ref": "refs/heads/main"}
    body = json.dumps(payload).encode()
    r = client.post("/repo-events/webhook", content=body, headers={
        "X-GitHub-Event": "push", "Content-Type": "application/json",
        "X-Hub-Signature-256": _sig("topsecret", body)})
    assert r.status_code == 200, r.text
    assert calls["call"] == ("wsZ", "push")


def test_push_webhook_form_content_type(monkeypatch):
    from urllib.parse import urlencode
    client, calls = _client(monkeypatch)
    inner = json.dumps({"repository": {"full_name": "Nileneb/Foo"},
                        "after": "abc123", "ref": "refs/heads/main"})
    body = urlencode({"payload": inner}).encode()   # GitHub content_type=form
    r = client.post("/repo-events/webhook", content=body, headers={
        "X-GitHub-Event": "push", "Content-Type": "application/x-www-form-urlencoded",
        "X-Hub-Signature-256": _sig("topsecret", body)})   # HMAC over the RAW form body
    assert r.status_code == 200, r.text
    assert calls["call"] == ("wsZ", "push")


def test_webhook_bad_signature_401(monkeypatch):
    client, _ = _client(monkeypatch)
    body = json.dumps({"repository": {"full_name": "Nileneb/Foo"}}).encode()
    r = client.post("/repo-events/webhook", content=body, headers={
        "X-GitHub-Event": "push", "Content-Type": "application/json",
        "X-Hub-Signature-256": _sig("WRONG", body)})
    assert r.status_code == 401


def test_webhook_unwatched_repo_401(monkeypatch):
    client, _ = _client(monkeypatch)
    body = json.dumps({"repository": {"full_name": "x/unwatched"}}).encode()
    r = client.post("/repo-events/webhook", content=body, headers={
        "X-GitHub-Event": "push", "Content-Type": "application/json",
        "X-Hub-Signature-256": _sig("topsecret", body)})
    assert r.status_code == 401


def test_ping_event_ignored(monkeypatch):
    client, _ = _client(monkeypatch)
    body = json.dumps({"repository": {"full_name": "Nileneb/Foo"}, "zen": "hi"}).encode()
    r = client.post("/repo-events/webhook", content=body, headers={
        "X-GitHub-Event": "ping", "Content-Type": "application/json",
        "X-Hub-Signature-256": _sig("topsecret", body)})
    assert r.status_code == 200
    assert r.json().get("action") == "ignored"
