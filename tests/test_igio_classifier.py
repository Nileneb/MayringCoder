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


def test_migration_preserves_existing_data(tmp_path: Path) -> None:
    """Running the migration on an existing memory.db must not drop rows."""
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
