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

from unittest.mock import patch, MagicMock


def test_conversation_micro_batch_rejects_mismatched_workspace_slug():
    """User authenticated as 'bene' MUST NOT be able to ingest into
    'alice' workspace by setting body.workspace_slug='alice'."""
    from fastapi import HTTPException
    from src.api.routes.memory import conversation_micro_batch
    from src.api.routes.models import ConversationMicroBatchRequest, ConversationTurnModel
    import asyncio

    request = ConversationMicroBatchRequest(
        turns=[ConversationTurnModel(role="user", content="hi", timestamp="2026-05-09T10:00:00Z")],
        session_id="sess123",
        workspace_slug="alice",  # Angreifer-Wert
        presumarized="test summary",
    )

    try:
        asyncio.run(conversation_micro_batch(request, workspace_id="bene"))
        raise AssertionError("expected HTTPException for slug mismatch")
    except HTTPException as exc:
        assert exc.status_code == 403
        assert "workspace" in exc.detail.lower()


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
        result = asyncio.run(conversation_micro_batch(request, workspace_id="bene"))

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
        result = asyncio.run(conversation_micro_batch(request, workspace_id="bene"))

    assert "bene" in result["source_id"]
