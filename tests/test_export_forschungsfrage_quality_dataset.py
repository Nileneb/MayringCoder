"""Tests for tools/export_forschungsfrage_quality_dataset.py (#260)."""
import json
import sqlite3
from pathlib import Path

import pytest

from export_forschungsfrage_quality_dataset import export


def _build_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE forschungsfrage_quality (
            forschungsfrage TEXT,
            score INTEGER,
            warnings TEXT,
            pico TEXT,
            workspace_id TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO forschungsfrage_quality VALUES (?,?,?,?,?)",
        (
            "Wie beeinflusst X das Y bei Z?",
            78,
            json.dumps([{"text": "zu breit", "suggestion": "Population eingrenzen"}]),
            json.dumps({"population": "Z", "intervention": "X", "outcome": "Y"}),
            "ws1",
        ),
    )
    # row with empty forschungsfrage → skipped
    conn.execute(
        "INSERT INTO forschungsfrage_quality VALUES (?,?,?,?,?)",
        ("", 50, "[]", "{}", "ws1"),
    )
    conn.commit()
    conn.close()


def test_export_from_sqlite_parses_json_columns(tmp_path):
    db = tmp_path / "fq.db"
    _build_db(db)
    out = tmp_path / "ds.jsonl"

    n = export(out, db_path=db)

    assert n == 1   # the empty-forschungsfrage row is skipped
    row = json.loads(out.read_text().splitlines()[0])
    assert row["input"]["forschungsfrage"] == "Wie beeinflusst X das Y bei Z?"
    assert row["output"]["score"] == 78
    assert row["output"]["warnings"][0]["suggestion"] == "Population eingrenzen"
    assert row["output"]["pico"]["intervention"] == "X"
    assert row["workspace_id"] == "ws1"


def test_export_requires_a_source(tmp_path):
    with pytest.raises(ValueError):
        export(tmp_path / "ds.jsonl")
