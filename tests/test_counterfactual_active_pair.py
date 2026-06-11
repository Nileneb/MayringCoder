"""Tests für die automatische Versionspaar-Ableitung im reranker-counterfactual-Endpoint.

Prüft ausschliesslich die Pair-Resolution-Logik (welche Versionen werden gewählt /
wann kommt HTTP 400), nicht das vollständige Re-Ranking. Das ist absichtlich:
die vollständige Ranking-Pipeline braucht echte Feedback-Daten und ist in
test_reranker_active_endpoints.py abgedeckt.
"""
from __future__ import annotations

import asyncio


def _maybe_run(c):
    return asyncio.run(c) if asyncio.iscoroutine(c) else c
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from mayring_core.memory.store import init_memory_db

import src.api.routes.retrieval_metrics as rm
from src.api.jwt_auth import TokenInfo


def _run(coro):
    return _maybe_run(coro) if asyncio.iscoroutine(coro) else coro


def _make_admin() -> TokenInfo:
    return TokenInfo(workspace_id="system", scopes=("*",))


def _seed_empty_db(tmp_path: Path):
    """Minimale DB mit allen nötigen Tabellen (kein Feedback-Inhalt).

    Leere context_feedback_log + chunk_feedback → counted=0 → Endpoint gibt
    Zero-Queries-Dict zurück, das trotzdem baseline/candidate enthält.
    """
    return init_memory_db(tmp_path / "m.db")


def test_uses_active_pair(monkeypatch, tmp_path):
    """Wenn kein baseline/candidate übergeben wird, zieht der Endpoint das Paar
    aus read_active_versions() — hier v3+v4."""
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))

    # Modell-Dateien anlegen (write_active_versions validiert deren Existenz)
    (tmp_path / "rerank_v3.json").write_text(json.dumps({"weights": {}}))
    (tmp_path / "rerank_v4.json").write_text(json.dumps({"weights": {}}))

    from mayring_core.memory.reranker_v2 import write_active_versions
    write_active_versions(["v3", "v4"])

    db = _seed_empty_db(tmp_path)
    monkeypatch.setattr(rm, "_conn", lambda: db)

    info = _make_admin()
    res = _run(rm.reranker_counterfactual(info=info))

    # Leere DB → counted=0 → Zero-Queries-Dict, enthält aber baseline+candidate
    assert res["baseline"] == "v3"
    assert res["candidate"] == "v4"
    assert res["queries"] == 0


def test_explicit_params_still_work(monkeypatch, tmp_path):
    """Explizit übergebene baseline/candidate überschreiben die aktiven Versionen
    (Back-Compat: bestehende API-Aufrufer dürfen Versions-Strings mitgeben)."""
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))

    (tmp_path / "rerank_v3.json").write_text(json.dumps({"weights": {}}))
    (tmp_path / "rerank_v4.json").write_text(json.dumps({"weights": {}}))

    from mayring_core.memory.reranker_v2 import write_active_versions
    write_active_versions(["v3", "v4"])

    db = _seed_empty_db(tmp_path)
    monkeypatch.setattr(rm, "_conn", lambda: db)

    info = _make_admin()
    # Explizit v4 als candidate statt das aktive v3-Paar
    res = _run(rm.reranker_counterfactual(info=info, baseline="v3", candidate="v4"))

    assert res["baseline"] == "v3"
    assert res["candidate"] == "v4"


def test_requires_two_active(monkeypatch, tmp_path):
    """Wenn nur eine Version aktiv ist, muss der Endpoint 400 zurückgeben."""
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))

    (tmp_path / "rerank_v4.json").write_text(json.dumps({"weights": {}}))

    from mayring_core.memory.reranker_v2 import write_active_versions
    write_active_versions(["v4"])

    db = _seed_empty_db(tmp_path)
    monkeypatch.setattr(rm, "_conn", lambda: db)

    info = _make_admin()
    with pytest.raises(HTTPException) as exc_info:
        _run(rm.reranker_counterfactual(info=info))

    assert exc_info.value.status_code == 400
    assert "2 aktive" in exc_info.value.detail


def test_requires_two_active_when_none_set(monkeypatch, tmp_path):
    """Keine aktive Version (frisches Cache-Dir) → Fallback ['v1'] → 400."""
    monkeypatch.setenv("MAYRING_CACHE_DIR", str(tmp_path))
    # kein rerank_active.json → read_active_versions() liefert ['v1']

    db = _seed_empty_db(tmp_path)
    monkeypatch.setattr(rm, "_conn", lambda: db)

    info = _make_admin()
    with pytest.raises(HTTPException) as exc_info:
        _run(rm.reranker_counterfactual(info=info))

    assert exc_info.value.status_code == 400
