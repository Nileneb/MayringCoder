"""Regression test against a REAL Claude Code transcript sample.

The previous Stop-hook tests passed because their fixtures embedded the
``_Injected chunks`` block inside ``message.content`` — but Claude Code
never writes hook output there. Hook output is *prompt prefix* at request
time, not persisted user message. So the in-test fixtures lied; the
parser worked perfectly against synthetic input that production never
produces.

This test loads the **actual** JSONL row Claude Code wrote during a
session (extracted into ``tests/fixtures/real_claude_transcript_sample.jsonl``
straight from ``~/.claude/projects/<…>/<session>.jsonl``) and pins down:

  1. ``extract_injected_chunks`` against the real user-turn content
     returns **0 pairs**. If a future change ever produces a non-empty
     list here, that means the upstream Claude-Code format changed
     (or someone re-introduced an in-content block).

  2. ``read_inject_state`` is therefore the only viable source of pairs
     for the Stop hook to operate on.

Update procedure: when Claude Code's transcript shape changes upstream,
re-extract the fixture by reading any session JSONL and writing the
last user/assistant pair to the fixture path.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "real_claude_transcript_sample.jsonl"


@pytest.fixture(scope="module")
def stop_hook():
    repo_root = Path(__file__).resolve().parent.parent
    hook_path = repo_root / "claude-plugin" / "hooks" / "stop_hook.py"
    spec = importlib.util.spec_from_file_location("stop_hook_real", hook_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stop_hook_real"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _flatten(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return ""


def test_real_transcript_does_not_contain_inject_block(stop_hook):
    """Production-truth: the inject block isn't in user-turn content.

    If this test ever flips green → red, the Claude-Code transcript shape
    has changed and we need to revisit the Stop-hook strategy.
    """
    assert FIXTURE.exists(), (
        f"Re-run the fixture-extraction snippet to populate {FIXTURE}"
    )
    with FIXTURE.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    user_rows = [r for r in rows if r.get("type") == "user"]
    assert user_rows, "fixture must contain at least one user row"

    for row in user_rows:
        content = _flatten(row.get("message", {}).get("content"))
        assert "Injected chunks" not in content, (
            "Real Claude-Code transcript should NOT contain the inject block "
            "in user message.content (hook output is prompt prefix, not "
            "persisted). If this fails, the upstream format changed."
        )
        pairs = stop_hook.extract_injected_chunks(content)
        assert pairs == [], (
            f"Real transcript user content yielded {len(pairs)} chunk pairs "
            f"via inline regex — should be 0."
        )


def test_state_file_is_the_only_path_for_real_transcripts(stop_hook, tmp_path, monkeypatch):
    """When the state file is missing, _auto_feedback must do nothing —
    silently relying on inline regex is exactly how the previous bug hid.
    """
    monkeypatch.setattr(stop_hook, "_INJECT_STATE_DIR", str(tmp_path / "no-state"))

    with FIXTURE.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    turns = [
        {"role": "user", "content": _flatten(rows[0]["message"]["content"]),
         "timestamp": rows[0].get("timestamp", "")},
        {"role": "assistant", "content": _flatten(rows[1]["message"]["content"]),
         "timestamp": rows[1].get("timestamp", "")},
    ]

    posted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        stop_hook, "_post_feedback",
        lambda cid, sig, tok: posted.append((cid, sig)),
    )
    stop_hook._auto_feedback(turns, "session-without-state-file", "fake-token")
    assert posted == [], (
        "With no inject-state file AND no inline block in transcript, "
        "auto_feedback must post nothing."
    )


def test_state_file_path_does_drive_feedback(stop_hook, tmp_path, monkeypatch):
    """Sanity: WITH a state file, auto_feedback uses it (and clears after)."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    session_id = "test-real-roundtrip"
    state_path = state_dir / f"{session_id}.json"
    state_path.write_text(json.dumps({"chunks": [
        {"chunk_id": "chk_aaaaaaaaaaaaaaaa", "source_id": "repo:x:src/foo.py"},
        {"chunk_id": "chk_bbbbbbbbbbbbbbbb", "source_id": "ambient:test:snapshot"},
    ]}))
    monkeypatch.setattr(stop_hook, "_INJECT_STATE_DIR", str(state_dir))

    turns = [
        {"role": "user", "content": "any user text", "timestamp": ""},
        {"role": "assistant",
         "content": "I touched src/foo.py and ran the tests.",
         "timestamp": ""},
    ]
    posted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        stop_hook, "_post_feedback",
        lambda cid, sig, tok: posted.append((cid, sig)),
    )
    stop_hook._auto_feedback(turns, session_id, "fake-token")

    assert ("chk_aaaaaaaaaaaaaaaa", "positive") in posted
    assert ("chk_bbbbbbbbbbbbbbbb", "negative") in posted
    assert not state_path.exists(), "state file must be cleared post-drain"
