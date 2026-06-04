"""Cross-tenant security tests for /conversation/micro-batch.

Code-Review found Medium #7: `request.workspace_slug` is user-controlled
in the request body (default 'default'). It flows into:
  - source_id = f"conversation:{slug}:{session}"  → cross-tenant collision
  - source_dict.repo = slug                       → cross-tenant marker
  - update_transitions_incremental(slug)          → poisons foreign Markov-chain

Fix: server MUST derive the slug from the authenticated workspace_id (JWT),
not trust the body. If body sets a mismatching slug → reject with 403.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

_INFO = SimpleNamespace(sub="test-user", org_ids=(), memberships=[])


def test_conversation_micro_batch_ignores_mismatched_slug_uses_authed_workspace():
    """A mismatched body slug must NOT cause a cross-tenant write — but it must
    also NOT drop the turn. The old 403 broke EVERY CLI capture (the hook sends
    the cwd-basename, which never equals the workspace UUID). Correct behaviour:
    ignore the attacker slug, ingest under the AUTHENTICATED workspace."""
    from src.api.routes.memory import conversation_micro_batch
    from src.api.routes.models import ConversationMicroBatchRequest, ConversationTurnModel
    import asyncio

    request = ConversationMicroBatchRequest(
        turns=[ConversationTurnModel(role="user", content="hi", timestamp="2026-05-09T10:00:00Z")],
        session_id="sess123",
        workspace_slug="alice",  # Angreifer-Wert — muss ignoriert werden
        presumarized="test summary",
    )

    captured: dict = {}

    def _fake_run_ingest(source_dict, content, conn, chroma, ollama, model, opts, ws):
        captured["source_dict"] = source_dict
        captured["ws"] = ws
        return {"status": "ok", "chunk_ids": [], "indexed": True,
                "deduped": 0, "filtered": 0, "superseded": 0}

    with patch("src.api.routes.memory._run_ingest", side_effect=_fake_run_ingest), \
         patch("src.api.routes.memory._get_conn", return_value=MagicMock()), \
         patch("src.api.routes.memory._get_chroma", return_value=MagicMock()):
        result = asyncio.run(conversation_micro_batch(request, workspace_id="bene", info=_INFO))

    # NO cross-tenant write: the attacker slug 'alice' must NOT appear anywhere
    assert "alice" not in result["source_id"]
    assert captured["source_dict"]["repo"] != "alice"
    assert captured["source_dict"]["path"] != "alice/incremental"
    # Bound to the AUTHENTICATED workspace + turn NOT dropped
    assert "bene" in result["source_id"]
    assert captured["ws"] == "bene"


def test_conversation_micro_batch_default_slug_is_overwritten_by_jwt():
    """User in 'bene'-workspace ohne body-slug: server defaultet auf
    JWT-workspace_id, NICHT auf 'default'. Plus: source_id bekommt korrekten
    slug-prefix."""
    from src.api.routes.memory import conversation_micro_batch
    from src.api.routes.models import ConversationMicroBatchRequest, ConversationTurnModel
    import asyncio

    request = ConversationMicroBatchRequest(
        turns=[ConversationTurnModel(role="user", content="hi", timestamp="2026-05-09T10:00:00Z")],
        session_id="sess123",
        # workspace_slug not set → defaults to "default"
        presumarized="test summary",
    )

    captured: dict = {}

    def _fake_run_ingest(source_dict, content, conn, chroma, ollama, model, opts, ws):
        captured["source_dict"] = source_dict
        captured["content"] = content
        captured["ws"] = ws
        return {"status": "ok", "chunk_ids": [], "indexed": True,
                "deduped": 0, "filtered": 0, "superseded": 0}

    with patch("src.api.routes.memory._run_ingest", side_effect=_fake_run_ingest), \
         patch("src.api.routes.memory._get_conn", return_value=MagicMock()), \
         patch("src.api.routes.memory._get_chroma", return_value=MagicMock()):
        result = asyncio.run(conversation_micro_batch(request, workspace_id="bene", info=_INFO))

    # source_id MUSS den JWT-workspace enthalten, nicht 'default'
    assert "bene" in result["source_id"], (
        f"source_id should contain JWT workspace 'bene', got {result['source_id']!r}"
    )
    assert "default" not in result["source_id"], (
        f"source_id should NOT contain user-default 'default', got {result['source_id']!r}"
    )
    # source_dict.repo + path also fixed
    assert captured["source_dict"]["repo"] == "bene"
    assert captured["source_dict"]["path"].startswith("bene/")


def test_conversation_micro_batch_matching_slug_passes():
    """Wenn body-slug == JWT-workspace, alles funktioniert wie immer."""
    from src.api.routes.memory import conversation_micro_batch
    from src.api.routes.models import ConversationMicroBatchRequest, ConversationTurnModel
    import asyncio

    request = ConversationMicroBatchRequest(
        turns=[ConversationTurnModel(role="user", content="hi", timestamp="2026-05-09T10:00:00Z")],
        session_id="sess123",
        workspace_slug="bene",  # matches JWT
        presumarized="test summary",
    )

    def _fake_run_ingest(*a, **kw):
        return {"status": "ok", "chunk_ids": [], "indexed": True,
                "deduped": 0, "filtered": 0, "superseded": 0}

    with patch("src.api.routes.memory._run_ingest", side_effect=_fake_run_ingest), \
         patch("src.api.routes.memory._get_conn", return_value=MagicMock()), \
         patch("src.api.routes.memory._get_chroma", return_value=MagicMock()):
        result = asyncio.run(conversation_micro_batch(request, workspace_id="bene", info=_INFO))

    assert "bene" in result["source_id"]


def test_conversation_micro_batch_stamps_owner_for_recall():
    """Conversation summaries MUST be stamped private + an explicit user_id so the
    scope_filter returns them on cross-session reads. Without the owner they were
    private+user_id=NULL → invisible to everyone (only the recency-lane masked it
    for the current session)."""
    from src.api.routes.memory import conversation_micro_batch
    from src.api.routes.models import ConversationMicroBatchRequest, ConversationTurnModel
    import asyncio

    request = ConversationMicroBatchRequest(
        turns=[ConversationTurnModel(role="user", content="hi", timestamp="2026-05-09T10:00:00Z")],
        session_id="sess123",
        presumarized="test summary",
    )

    captured: dict = {}

    def _fake_run_ingest(source_dict, content, conn, chroma, ollama, model, opts, ws):
        captured["source_dict"] = source_dict
        return {"status": "ok", "chunk_ids": [], "indexed": True,
                "deduped": 0, "filtered": 0, "superseded": 0}

    with patch("src.api.routes.memory._run_ingest", side_effect=_fake_run_ingest), \
         patch("src.api.routes.memory._get_conn", return_value=MagicMock()), \
         patch("src.api.routes.memory._get_chroma", return_value=MagicMock()):
        asyncio.run(conversation_micro_batch(request, workspace_id="bene", info=_INFO))

    assert captured["source_dict"]["visibility"] == "private"
    assert captured["source_dict"]["user_id"] == "test-user"  # info.sub, not NULL


def test_micro_batch_with_user_turn_returns_200_and_no_error():
    """Posting a micro-batch with a user turn succeeds (derive_todo runs in background thread)."""
    from src.api.routes.memory import conversation_micro_batch
    from src.api.routes.models import ConversationMicroBatchRequest, ConversationTurnModel
    import asyncio

    request = ConversationMicroBatchRequest(
        turns=[ConversationTurnModel(role="user", content="please implement the login feature",
                                     timestamp="2026-05-22T10:00:00Z")],
        session_id="sess_derive_test",
        presumarized="test summary for derive",
    )

    def _fake_run_ingest(*a, **kw):
        return {"status": "ok", "chunk_ids": [], "indexed": True,
                "deduped": 0, "filtered": 0, "superseded": 0}

    with patch("src.api.routes.memory._run_ingest", side_effect=_fake_run_ingest), \
         patch("src.api.routes.memory._get_conn", return_value=MagicMock()), \
         patch("src.api.routes.memory._get_chroma", return_value=MagicMock()):
        result = asyncio.run(conversation_micro_batch(request, workspace_id="bene", info=_INFO))

    # The response must succeed — derive_todo runs in a daemon thread and does not block
    assert "source_id" in result


def test_micro_batch_skips_derive_for_system_workspace():
    """System workspace MUST NOT trigger derive_todo (no noise from cron/service tokens)."""
    from src.api.routes.memory import conversation_micro_batch
    from src.api.routes.models import ConversationMicroBatchRequest, ConversationTurnModel
    import asyncio
    import threading

    request = ConversationMicroBatchRequest(
        turns=[ConversationTurnModel(role="user", content="please fix auth",
                                     timestamp="2026-05-22T10:00:00Z")],
        session_id="sess_system",
        presumarized="system summary",
    )

    spawned = []

    def _track_start(self, *a, **kw):
        spawned.append(self)
        # don't actually start — avoid real LLM calls
        pass

    def _fake_run_ingest(*a, **kw):
        return {"status": "ok", "chunk_ids": [], "indexed": True,
                "deduped": 0, "filtered": 0, "superseded": 0}

    with patch("src.api.routes.memory._run_ingest", side_effect=_fake_run_ingest), \
         patch("src.api.routes.memory._get_conn", return_value=MagicMock()), \
         patch("src.api.routes.memory._get_chroma", return_value=MagicMock()), \
         patch.object(threading.Thread, "start", _track_start):
        asyncio.run(conversation_micro_batch(request, workspace_id="system", info=_INFO))

    assert len(spawned) == 0, "derive_todo thread must NOT be spawned for system workspace"
