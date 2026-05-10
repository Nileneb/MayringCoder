"""Stop-hook turn capture: parse JSONL transcript → user/assistant pair.

The Stop hook ingests every completed turn pair into Memory so the conversation
state is fully captured without waiting for /compact. These tests cover the
JSONL parser only — the HTTP POST to /memory/conversation/micro-batch is
exercised by integration tests against the real API.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def stop_hook():
    """Load claude-plugin/hooks/stop_hook.py as a module without registering it.

    The plugin hooks directory isn't on sys.path, so we import by file path.
    """
    repo_root = Path(__file__).resolve().parent.parent
    hook_path = repo_root / "claude-plugin" / "hooks" / "stop_hook.py"
    spec = importlib.util.spec_from_file_location("stop_hook_module", hook_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stop_hook_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_transcript(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_extract_returns_empty_for_missing_file(stop_hook, tmp_path):
    assert stop_hook.extract_last_turn_pair(str(tmp_path / "nope.jsonl")) == []


def test_extract_returns_empty_for_empty_path(stop_hook):
    assert stop_hook.extract_last_turn_pair("") == []


def test_extract_picks_last_user_and_assistant(stop_hook, tmp_path):
    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "first question"},
         "timestamp": "2026-05-08T10:00:00Z"},
        {"type": "assistant", "message": {"role": "assistant", "content": "first answer"},
         "timestamp": "2026-05-08T10:00:30Z"},
        {"type": "user", "message": {"role": "user", "content": "second question"},
         "timestamp": "2026-05-08T10:01:00Z"},
        {"type": "assistant", "message": {"role": "assistant", "content": "second answer"},
         "timestamp": "2026-05-08T10:01:30Z"},
    ])

    pair = stop_hook.extract_last_turn_pair(str(transcript))
    assert len(pair) == 2
    assert pair[0]["role"] == "user"
    assert pair[0]["content"] == "second question"
    assert pair[1]["role"] == "assistant"
    assert pair[1]["content"] == "second answer"


def test_extract_flattens_structured_content_blocks(stop_hook, tmp_path):
    """Claude Code sends content as a list of typed blocks; thinking is skipped."""
    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "go"},
         "timestamp": "2026-05-08T10:00:00Z"},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "text": "private chain of thought"},
            {"type": "text", "text": "Visible answer line 1."},
            {"type": "text", "text": "Visible answer line 2."},
        ]}, "timestamp": "2026-05-08T10:00:30Z"},
    ])

    pair = stop_hook.extract_last_turn_pair(str(transcript))
    assistant = pair[-1]
    assert assistant["role"] == "assistant"
    assert "Visible answer line 1." in assistant["content"]
    assert "Visible answer line 2." in assistant["content"]
    assert "private chain of thought" not in assistant["content"]


def test_extract_ignores_meta_rows_and_empty_content(stop_hook, tmp_path):
    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, [
        {"type": "summary", "summary": "old session"},
        {"type": "ai-title", "title": "irrelevant"},
        {"type": "user", "message": {"role": "user", "content": ""},
         "timestamp": "2026-05-08T10:00:00Z"},
        {"type": "user", "message": {"role": "user", "content": "real question"},
         "timestamp": "2026-05-08T10:00:01Z"},
        {"type": "assistant", "message": {"role": "assistant", "content": "real answer"},
         "timestamp": "2026-05-08T10:00:02Z"},
    ])

    pair = stop_hook.extract_last_turn_pair(str(transcript))
    assert [t["role"] for t in pair] == ["user", "assistant"]
    assert pair[0]["content"] == "real question"


def test_extract_skips_malformed_jsonl_lines(stop_hook, tmp_path):
    """One bad line in the middle of a transcript must not abort the whole parse."""
    transcript = tmp_path / "session.jsonl"
    with transcript.open("w") as f:
        f.write(json.dumps({"type": "user", "message": {"role": "user", "content": "q1"}}) + "\n")
        f.write("THIS IS NOT JSON\n")
        f.write(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "a1"}}) + "\n")

    pair = stop_hook.extract_last_turn_pair(str(transcript))
    assert len(pair) == 2
    assert pair[1]["content"] == "a1"


def test_extract_truncates_huge_content(stop_hook, tmp_path):
    """Per-turn content is capped so a paste-bomb can't blow up the POST payload."""
    transcript = tmp_path / "session.jsonl"
    huge = "x" * 99999
    _write_transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "q"}},
        {"type": "assistant", "message": {"role": "assistant", "content": huge}},
    ])

    pair = stop_hook.extract_last_turn_pair(str(transcript))
    assert len(pair[-1]["content"]) <= stop_hook._MAX_TURN_CHARS


def test_extract_returns_only_when_both_sides_present(stop_hook, tmp_path):
    """Without an assistant turn, downstream micro-batch would just summarise the
    user prompt — pointless. We still return what's there; the caller decides."""
    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "lone question"}},
    ])
    pair = stop_hook.extract_last_turn_pair(str(transcript))
    assert pair == [{"role": "user", "content": "lone question", "timestamp": ""}]


# ---------------------------------------------------------------------------
# Auto-feedback parser + classifier
# ---------------------------------------------------------------------------

def test_extract_injected_chunks_parses_inject_block(stop_hook):
    user_text = """## Memory-Kontext für diesen Prompt

### Code/Findings
some context here

_Injected chunks (auto-feedback by Stop hook):_
- `chk_9d411417cecd75ec` : `repo:https://github.com/x/y:src/agents/pi.py`
- `chk_2ad33a9cca1c128a` : `ambient:mayringcoder:snapshot`
"""
    pairs = stop_hook.extract_injected_chunks(user_text)
    assert pairs == [
        ("chk_9d411417cecd75ec", "repo:https://github.com/x/y:src/agents/pi.py"),
        ("chk_2ad33a9cca1c128a", "ambient:mayringcoder:snapshot"),
    ]


def test_extract_injected_chunks_dedups(stop_hook):
    text = "- `chk_aaaaaaaaaaaaaaaa` : `s1`\n- `chk_aaaaaaaaaaaaaaaa` : `s2`"
    assert stop_hook.extract_injected_chunks(text) == [("chk_aaaaaaaaaaaaaaaa", "s1")]


def test_extract_injected_chunks_empty_on_no_block(stop_hook):
    assert stop_hook.extract_injected_chunks("just a normal prompt") == []
    assert stop_hook.extract_injected_chunks("") == []


def test_classify_positive_on_full_path(stop_hook):
    src = "repo:https://github.com/x/y:src/agents/pi.py"
    answer = "I edited src/agents/pi.py to fix the loop."
    assert stop_hook.classify_chunk_relevance(src, answer) == "positive"


def test_classify_positive_on_basename(stop_hook):
    src = "repo:https://github.com/x/y:src/agents/pi.py"
    answer = "Look at pi.py for context."
    assert stop_hook.classify_chunk_relevance(src, answer) == "positive"


def test_classify_negative_when_path_unmentioned_long_answer(stop_hook):
    """Substantial answer (>=200 chars) without path-match -> negative."""
    src = "repo:https://github.com/x/y:src/agents/pi.py"
    answer = "Ich habe keine Ahnung was du willst. " + "x" * 200
    assert stop_hook.classify_chunk_relevance(src, answer) == "negative"


def test_classify_skip_when_path_unmentioned_short_answer(stop_hook):
    """Short answer (<200 chars) without path-match -> skip (kein false-negative
    fuer generic files wie web.php deren name selten erwaehnt wird)."""
    src = "repo:https://github.com/x/y:src/agents/pi.py"
    answer = "Ich habe keine Ahnung was du willst."
    assert stop_hook.classify_chunk_relevance(src, answer) == "skip"


def test_classify_skip_on_too_short_basename(stop_hook):
    """Avoid matching `a.py` against `a` — short answer also -> skip."""
    src = "repo:x:a.py"
    answer = "wir tun a und b"
    assert stop_hook.classify_chunk_relevance(src, answer) == "skip"


def test_classify_handles_non_repo_source_ids(stop_hook):
    """ambient/conversation_summary/note source_ids — same logic on tail."""
    assert stop_hook.classify_chunk_relevance(
        "conversation:mayringcoder:abc123",
        "context recap: refer back to mayringcoder session abc123",
    ) == "positive"
    assert stop_hook.classify_chunk_relevance(
        "ambient:foo:snapshot", "completely unrelated"
    ) == "skip"


def test_classify_skip_on_empty_inputs(stop_hook):
    assert stop_hook.classify_chunk_relevance("", "anything") == "skip"
    assert stop_hook.classify_chunk_relevance("repo:x:y.py", "") == "skip"
