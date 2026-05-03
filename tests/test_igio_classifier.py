"""Tests for IGIO axis classifier (no Ollama dependency).

Covers:
- Fast keyword shortcuts (no LLM call)
- LLM JSON parsing happy path
- LLM error / malformed output → empty verdict
- Schema migration is idempotent on existing memory.db rows
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.wiki_v2.igio_classifier import (
    VALID_AXES,
    IgioVerdict,
    _fast_classify,
    _parse_verdict,
    classify_chunk,
)


# ----- Fast hints -----------------------------------------------------------


@pytest.mark.parametrize(
    "text,axis",
    [
        ("============ 142 passed in 3.41s ============", "outcome"),
        ("Tests grün auf master", "outcome"),
        ('Traceback (most recent call last):\n  File "x.py"', "issue"),
        ("BUG: token expires before refresh runs", "issue"),
        ("## Plan\n\n- Step 1: ...", "intervention"),
    ],
)
def test_fast_classify_known_phrases(text: str, axis: str) -> None:
    v = _fast_classify(text)
    assert v is not None, f"expected fast hit for {text!r}"
    assert v.axis == axis
    assert v.confidence >= 0.7


def test_fast_classify_misses_neutral_text() -> None:
    assert _fast_classify("The user clicked the button to log in.") is None


# ----- JSON parser ----------------------------------------------------------


def test_parse_verdict_strict_json() -> None:
    raw = json.dumps({"axis": "outcome", "confidence": 0.83, "rationale": "tests passed"})
    v = _parse_verdict(raw)
    assert v == IgioVerdict(axis="outcome", confidence=0.83, rationale="tests passed")


def test_parse_verdict_extracts_embedded_json() -> None:
    raw = (
        'Sure, here you go:\n'
        '{"axis": "intervention", "confidence": 0.7, "rationale": "refactor"}'
        '\nBest regards.'
    )
    v = _parse_verdict(raw)
    assert v is not None and v.axis == "intervention"
    assert v.confidence == pytest.approx(0.7)


def test_parse_verdict_rejects_invalid_axis() -> None:
    raw = json.dumps({"axis": "improvement", "confidence": 0.9, "rationale": "x"})
    assert _parse_verdict(raw) is None


def test_parse_verdict_clamps_confidence() -> None:
    raw = json.dumps({"axis": "issue", "confidence": 1.5, "rationale": "x"})
    v = _parse_verdict(raw)
    assert v is not None and v.confidence == 1.0


def test_parse_verdict_missing_confidence_defaults_zero() -> None:
    raw = json.dumps({"axis": "issue", "rationale": "x"})
    v = _parse_verdict(raw)
    assert v is not None and v.confidence == 0.0


def test_parse_verdict_non_numeric_confidence_defaults_zero() -> None:
    raw = json.dumps({"axis": "issue", "confidence": "high", "rationale": "x"})
    v = _parse_verdict(raw)
    assert v is not None and v.confidence == 0.0


def test_parse_verdict_missing_rationale_yields_empty_string() -> None:
    raw = json.dumps({"axis": "outcome", "confidence": 0.6})
    v = _parse_verdict(raw)
    assert v is not None and v.rationale == ""


def test_parse_verdict_truncates_long_rationale() -> None:
    long = "y" * 500
    raw = json.dumps({"axis": "outcome", "confidence": 0.6, "rationale": long})
    v = _parse_verdict(raw)
    assert v is not None and len(v.rationale) == 200


def test_parse_verdict_handles_nested_objects_in_rationale() -> None:
    """Slice-based extraction must survive nested JSON inside the value."""
    raw = (
        '{"axis": "intervention", "confidence": 0.7, '
        '"rationale": "patched {auth, session} flow", "extra": [1, 2, 3]}'
    )
    v = _parse_verdict(raw)
    assert v is not None
    assert v.axis == "intervention"
    assert "auth" in v.rationale


def test_parse_verdict_strips_markdown_fence() -> None:
    raw = '```json\n{"axis": "goal", "confidence": 0.9, "rationale": "x"}\n```'
    v = _parse_verdict(raw)
    assert v is not None and v.axis == "goal"


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("tests", "outcome"),
        ("testing", "outcome"),
        ("test", "outcome"),
        ("fix", "intervention"),
        ("refactor", "intervention"),
        ("bug", "issue"),
    ],
)
def test_parse_verdict_remaps_common_aliases(alias: str, canonical: str) -> None:
    """qwen models keep emitting 'tests' for test fns even when the prompt
    forbids it. Re-map well-known near-misses to the canonical axis."""
    raw = json.dumps({"axis": alias, "confidence": 1.0, "rationale": "x"})
    v = _parse_verdict(raw)
    assert v is not None
    assert v.axis == canonical
    assert v.confidence < 1.0  # penalised for non-canonical token


def test_parse_verdict_alias_low_confidence_clamps_to_zero() -> None:
    """Alias confidence below the penalty must clamp to 0.0, not go negative."""
    raw = json.dumps({"axis": "tests", "confidence": 0.05, "rationale": "x"})
    v = _parse_verdict(raw)
    assert v is not None
    assert v.axis == "outcome"
    assert v.confidence == 0.0


def test_parse_verdict_alias_missing_confidence_stays_zero() -> None:
    """Missing confidence + alias penalty must not produce a negative value."""
    raw = json.dumps({"axis": "tests", "rationale": "x"})
    v = _parse_verdict(raw)
    assert v is not None
    assert v.axis == "outcome"
    assert v.confidence == 0.0


def test_parse_verdict_alias_confidence_equal_to_penalty_clamps_to_zero() -> None:
    """Boundary case: confidence exactly equal to _ALIAS_PENALTY (0.1)
    must produce 0.0 — not a near-zero positive — so the clamp semantics
    are unambiguous if the penalty constant is ever tuned."""
    raw = json.dumps({"axis": "tests", "confidence": 0.1, "rationale": "x"})
    v = _parse_verdict(raw)
    assert v is not None
    assert v.axis == "outcome"
    assert v.confidence == 0.0


def test_parse_verdict_strips_fence_with_trailing_whitespace() -> None:
    """Closing fence followed by whitespace must still parse cleanly."""
    raw = '```json\n{"axis": "outcome", "confidence": 0.7, "rationale": "x"}\n```   \n'
    v = _parse_verdict(raw)
    assert v is not None and v.axis == "outcome"


def test_parse_verdict_strips_fence_with_trailing_text() -> None:
    """Trailing prose after the closing fence is also tolerated."""
    raw = (
        '```json\n{"axis": "issue", "confidence": 0.8, "rationale": "x"}\n'
        '```\n\nLet me know if you need more.'
    )
    v = _parse_verdict(raw)
    assert v is not None and v.axis == "issue"


def test_parse_verdict_strips_fence_with_leading_whitespace() -> None:
    """Leading spaces/newlines before the opening fence must still parse."""
    raw = '   \n```json\n{"axis": "goal", "confidence": 0.6, "rationale": "x"}\n```'
    v = _parse_verdict(raw)
    assert v is not None and v.axis == "goal"


def test_parse_verdict_keeps_backticks_inside_rationale() -> None:
    """A stray triple-backtick inside the rationale (e.g. a markdown example)
    must NOT be treated as the closing fence — the closer is line-anchored."""
    raw = (
        '```json\n'
        '{"axis": "intervention", "confidence": 0.7, '
        '"rationale": "Example shown as ``` snippet ``` inside text"}\n'
        '```'
    )
    v = _parse_verdict(raw)
    assert v is not None
    assert v.axis == "intervention"
    assert "snippet" in v.rationale


def test_parse_verdict_unknown_axis_still_rejected() -> None:
    """Aliases are an allow-list, not a free-for-all."""
    raw = json.dumps({"axis": "improvement", "confidence": 0.9, "rationale": "x"})
    assert _parse_verdict(raw) is None


def test_parse_verdict_handles_empty_axis() -> None:
    raw = json.dumps({"axis": "", "confidence": 0.1, "rationale": "uncertain"})
    v = _parse_verdict(raw)
    assert v is not None and v.axis == ""


def test_parse_verdict_garbage_returns_none() -> None:
    assert _parse_verdict("not json at all") is None


# ----- Full classify_chunk path with mocked LLM -----------------------------


def test_classify_chunk_uses_fast_hint_without_llm() -> None:
    with patch("src.wiki_v2.igio_classifier.generate") as gen:
        v = classify_chunk(
            "BUG: login flow drops session after refresh",
            [], ollama_url="http://x", model="m",
        )
    assert v.axis == "issue"
    gen.assert_not_called()


def test_classify_chunk_uses_llm_when_no_fast_hint() -> None:
    with patch("src.wiki_v2.igio_classifier.generate") as gen:
        gen.return_value = json.dumps({"axis": "goal", "confidence": 0.78, "rationale": "ok"})
        v = classify_chunk(
            "We aim to keep p95 latency under 200ms for the search endpoint.",
            ["api", "performance"], ollama_url="http://x", model="m",
        )
    assert v.axis == "goal"
    assert v.confidence == pytest.approx(0.78)
    gen.assert_called_once()


def test_classify_chunk_swallows_llm_error() -> None:
    with patch("src.wiki_v2.igio_classifier.generate", side_effect=RuntimeError("net down")):
        v = classify_chunk("Some neutral text.", [], ollama_url="http://x", model="m")
    assert v.axis == ""
    assert v.confidence == 0.0


def test_classify_chunk_handles_malformed_llm_output() -> None:
    with patch("src.wiki_v2.igio_classifier.generate", return_value="???"):
        v = classify_chunk("Some neutral text.", [], ollama_url="http://x", model="m")
    assert v.axis == ""
    assert v.rationale == "parse_error"


def test_classify_chunk_truncates_long_text() -> None:
    long = "x" * 5000
    captured: dict = {}

    def _gen(_url: str, _model: str, prompt: str, **_kwargs: object) -> str:
        captured["prompt"] = prompt
        return json.dumps({"axis": "outcome", "confidence": 0.6, "rationale": "ok"})

    with patch("src.wiki_v2.igio_classifier.generate", side_effect=_gen):
        v = classify_chunk(long, [], ollama_url="http://x", model="m")
    assert v.axis == "outcome"
    # Snippet was truncated with the elision marker
    assert "…" in captured["prompt"]
    assert len(captured["prompt"]) < 2500


def test_valid_axes_constant() -> None:
    assert set(VALID_AXES) == {"issue", "goal", "intervention", "outcome"}


# ----- Schema migration is idempotent --------------------------------------


def test_migration_adds_igio_columns_idempotently(tmp_path: Path) -> None:
    from src.memory.store import init_memory_db

    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    init_memory_db(db).close()  # second run must be a no-op

    cols = {r[1] for r in sqlite3.connect(db).execute(
        "PRAGMA table_info(chunks)"
    ).fetchall()}
    assert {"igio_axis", "igio_confidence", "igio_classified_at"} <= cols


def test_migration_preserves_existing_rows_hermetic(tmp_path: Path) -> None:
    """Hermetic: build a pre-IGIO chunks table, migrate, assert rows survive."""
    from src.memory.store import init_memory_db

    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE sources (
            source_id     TEXT PRIMARY KEY,
            source_type   TEXT NOT NULL DEFAULT 'repo_file',
            repo          TEXT NOT NULL DEFAULT '',
            path          TEXT NOT NULL DEFAULT '',
            branch        TEXT NOT NULL DEFAULT 'main',
            "commit"      TEXT NOT NULL DEFAULT '',
            content_hash  TEXT NOT NULL DEFAULT '',
            captured_at   TEXT NOT NULL
        );
        CREATE TABLE chunks (
            chunk_id          TEXT PRIMARY KEY,
            source_id         TEXT NOT NULL,
            chunk_level       TEXT NOT NULL DEFAULT 'file',
            ordinal           INTEGER NOT NULL DEFAULT 0,
            start_offset      INTEGER NOT NULL DEFAULT 0,
            end_offset        INTEGER NOT NULL DEFAULT 0,
            text              TEXT NOT NULL DEFAULT '',
            text_hash         TEXT NOT NULL DEFAULT '',
            summary           TEXT NOT NULL DEFAULT '',
            category_labels   TEXT NOT NULL DEFAULT '',
            category_version  TEXT NOT NULL DEFAULT 'mayring-inductive-v1',
            embedding_model   TEXT NOT NULL DEFAULT 'nomic-embed-text',
            embedding_id      TEXT NOT NULL DEFAULT '',
            quality_score     REAL NOT NULL DEFAULT 0.0,
            dedup_key         TEXT NOT NULL DEFAULT '',
            created_at        TEXT NOT NULL,
            is_active         INTEGER NOT NULL DEFAULT 1
        );
    """)
    conn.executemany(
        "INSERT INTO sources (source_id, captured_at) VALUES (?, ?)",
        [(f"src_{i}", "2026-01-01") for i in range(3)],
    )
    conn.executemany(
        "INSERT INTO chunks (chunk_id, source_id, text, created_at) VALUES (?, ?, ?, ?)",
        [(f"chk_{i}", f"src_{i}", f"text {i}", "2026-01-01") for i in range(3)],
    )
    conn.commit()
    conn.close()

    init_memory_db(db).close()

    conn = sqlite3.connect(db)
    after = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    assert after == 3
    assert {"igio_axis", "igio_confidence", "igio_classified_at"} <= cols


def test_migration_preserves_existing_data_realdb(tmp_path: Path) -> None:
    """Optional integration check against the real cache/memory.db."""
    from src.memory.store import init_memory_db

    real_db = Path(__file__).resolve().parent.parent / "cache" / "memory.db"
    if not real_db.exists():
        pytest.skip("no real memory.db present in this checkout")

    test_db = tmp_path / "memory.db"
    shutil.copy(real_db, test_db)

    before = sqlite3.connect(test_db).execute(
        "SELECT COUNT(*) FROM chunks"
    ).fetchone()[0]

    init_memory_db(test_db).close()

    conn = sqlite3.connect(test_db)
    after = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    assert before == after
    assert {"igio_axis", "igio_confidence", "igio_classified_at"} <= cols
