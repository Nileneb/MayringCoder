"""V2 Stufe 5 — Default-disabled Cleanup.

Spec: docs/v2-master-audit.md Section 7 Stufe 5.

- LLM-Advisor Stage 3b: hook-path setzt `llm_prefilter=False`.
  Eigene leichte Stage hat sich nicht durchgesetzt → Feature DROPPEN
  (KISS, kein Bedarf bewiesen). `sl`/`pt` retrainings-features bleiben
  erhalten (Reranker-v2 nutzt sie).
- Reranker-v2 default: cache/rerank_default.txt sollte 'auto' sein
  (Auto-Rollout via training-cron) statt hard-pinned 'v1'.
- MCP_AUTH_ENABLED-Drift: Code-default + .env.example synchronisieren.
- session_compacted=False Default: in /memory/search default-on (= True
  auto-detect, kein default-off).
- Reconcile-Job Chroma↔SQLite: implementieren als POST /admin/reconcile-chroma
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# 5.1 — MCP_AUTH_ENABLED Drift
# ---------------------------------------------------------------------------


def test_env_example_mcp_auth_enabled_matches_code_default():
    """.env.example MUSS MCP_AUTH_ENABLED=true setzen.

    Code-default in src/api/mcp_auth.py ist 'true' (forgotten env var
    auf prod darf NICHT silently auth deaktivieren). .env.example
    sollte das spiegeln, sonst landen frische Deploys mit 'false' im prod.
    """
    env_example = ROOT / ".env.example"
    text = env_example.read_text(encoding="utf-8")
    # Suche nach MCP_AUTH_ENABLED=
    line = next(
        (ln for ln in text.splitlines()
         if ln.strip().startswith("MCP_AUTH_ENABLED=")),
        None,
    )
    assert line is not None, ".env.example fehlt MCP_AUTH_ENABLED-Eintrag"
    value = line.split("=", 1)[1].strip().strip('"').strip("'").lower()
    assert value == "true", (
        f".env.example MCP_AUTH_ENABLED={value!r} drifts vom Code-Default 'true'"
    )


# ---------------------------------------------------------------------------
# 5.2 — Reranker-v2 default
# ---------------------------------------------------------------------------


def test_rerank_runtime_default_is_auto_when_file_absent(tmp_path, monkeypatch):
    """Wenn cache/rerank_default.txt fehlt, MUSS _read_runtime_default 'auto'
    zurückgeben — sonst sitzt das System silent auf v1 für immer.

    Vor Stufe 5 war der Default 'v1' hardcoded; jetzt rotiert die Auto-
    Rollout-Cron auf v2 sobald Trainingsdaten reichen. cache-Datei selbst
    ist gitignored (runtime state).
    """
    monkeypatch.setattr("src.config.CACHE_DIR", tmp_path, raising=False)
    monkeypatch.delenv("RERANKER_VERSION", raising=False)
    from src.memory.reranker_v2 import _read_runtime_default
    assert _read_runtime_default() == "auto"


def test_get_active_reranker_returns_v1_or_v2_not_default_pinned(monkeypatch):
    """Default-Pfad (kein env, kein file) führt zu 'auto' → split auf v1/v2,
    NICHT zu hard-gepinntem v1.
    """
    monkeypatch.delenv("RERANKER_VERSION", raising=False)
    from src.memory import reranker_v2
    monkeypatch.setattr(
        reranker_v2, "_read_runtime_default", lambda: "auto",
    )
    version, _ = reranker_v2.get_active_reranker(query_hint="some-query")
    assert version in ("v1", "v2")


# ---------------------------------------------------------------------------
# 5.3 — Reconcile-Job Chroma↔SQLite
# ---------------------------------------------------------------------------


def test_reconcile_chroma_route_exists():
    """POST /admin/reconcile-chroma muss als Route registriert sein."""
    import importlib
    mod = importlib.import_module("src.api.routes.reconcile_admin")
    assert hasattr(mod, "router"), (
        "src/api/routes/reconcile_admin.py fehlt — Stufe 5.3"
    )
    routes = [r for r in mod.router.routes]
    paths = [getattr(r, "path", "") for r in routes]
    assert any("/admin/reconcile-chroma" in p for p in paths), (
        f"Route /admin/reconcile-chroma fehlt. Existing: {paths}"
    )


def test_reconcile_function_returns_diff_report(tmp_path, monkeypatch):
    """Die zugrundeliegende Reconcile-Funktion vergleicht SQLite chunk_ids
    gegen Chroma-collection. Returns dict mit `missing_in_chroma`,
    `missing_in_sqlite`, `total_sqlite`, `total_chroma`.
    """
    from src.api.routes.reconcile_admin import _reconcile_chroma_sqlite
    # Funktion muss callable sein + dict mit den 4 keys liefern.
    # Mit None+None args → triviale 0-counts (lazy-mode für Tests).
    out = _reconcile_chroma_sqlite(None, None)
    assert isinstance(out, dict)
    for k in ("missing_in_chroma", "missing_in_sqlite",
              "total_sqlite", "total_chroma"):
        assert k in out, f"reconcile-output fehlt key {k!r}"


# ---------------------------------------------------------------------------
# 5.4 — LLM-Advisor Stage 3b dropped (no llm_prefilter param mehr nötig)
# ---------------------------------------------------------------------------


def test_llm_prefilter_param_still_accepted_for_backward_compat():
    """`llm_prefilter`-Parameter bleibt im MemorySearchRequest-Schema
    bestehen für backward-compat (alte Hook-Versionen senden ihn noch).
    Wir DROPPEN nicht das Schema-Feld, nur das default-active-Verhalten.
    """
    from src.api.routes.models import MemorySearchRequest
    req = MemorySearchRequest(query="x", llm_prefilter=False)
    assert req.llm_prefilter is False
