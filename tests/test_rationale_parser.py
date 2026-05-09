"""Tests for rationale-marker parser (Issue #185/#182 follow-up)."""
from __future__ import annotations
from pathlib import Path

from src.wiki_v2.rationale_parser import extract_rationale_edges


def test_extract_simple_marker_before_assign(tmp_path: Path) -> None:
    """Single-line WHY-marker direkt vor einer module-level Assignment."""
    src_file = tmp_path / "module.py"
    src_file.write_text(
        "import re\n"
        "\n"
        "# WHY(#185): path-traversal defence\n"
        "_SLUG_RE = re.compile(r'^[a-z]+$')\n"
    )
    edges = extract_rationale_edges(
        src_file, repo_slug="demo", workspace_id="bene",
    )
    assert len(edges) == 1
    e = edges[0]
    assert e["source"] == "module.py"
    assert e["target"] == "module._SLUG_RE"
    assert e["type"] == "rationale"
    assert e["context"] == "#185"
    assert e["rationale"] == "path-traversal defence"
    assert e["repo_slug"] == "demo"
    assert e["workspace_id"] == "bene"


def test_multi_line_rationale_concatenated(tmp_path: Path) -> None:
    """Folgezeilen mit '# ' (kein WHY-keyword) gehören zur rationale."""
    src_file = tmp_path / "module.py"
    src_file.write_text(
        "# WHY(#182, performance): SQLite busy_timeout=5s.\n"
        "# Single-Tx > 50 rows blockt smoke-test concurrent writes.\n"
        "# CHANGE WITH CARE.\n"
        "def commit_chunked(rows):\n"
        "    pass\n"
    )
    edges = extract_rationale_edges(
        src_file, repo_slug="demo", workspace_id="bene",
    )
    assert len(edges) == 1
    e = edges[0]
    assert e["target"] == "module.commit_chunked"
    assert e["context"] == "#182, performance"
    assert "SQLite busy_timeout=5s" in e["rationale"]
    assert "Single-Tx" in e["rationale"]
    assert "CHANGE WITH CARE" in e["rationale"]
    # Newlines preserved as join-char
    assert "\n" in e["rationale"]


def test_qualified_name_includes_class(tmp_path: Path) -> None:
    """Marker im class body produziert 'module.Class.method'."""
    src_file = tmp_path / "module.py"
    src_file.write_text(
        "class JobRunner:\n"
        "    # WHY(#100): retry-loop avoids transient deploy 502s\n"
        "    def run(self):\n"
        "        pass\n"
    )
    edges = extract_rationale_edges(
        src_file, repo_slug="demo", workspace_id="bene",
    )
    assert len(edges) == 1
    assert edges[0]["target"] == "module.JobRunner.run"


def test_handles_async_function(tmp_path: Path) -> None:
    """async def wird auch erkannt (AsyncFunctionDef-Node)."""
    src_file = tmp_path / "module.py"
    src_file.write_text(
        "# WHY(#88): async because the LLM-call blocks 30s\n"
        "async def fetch():\n"
        "    pass\n"
    )
    edges = extract_rationale_edges(
        src_file, repo_slug="demo", workspace_id="bene",
    )
    assert len(edges) == 1
    assert edges[0]["target"] == "module.fetch"


def test_skips_marker_before_for_loop(tmp_path: Path, caplog) -> None:
    """Marker vor non-trivialem Target (for/if/try/while) wird geskipped
    UND ein WARN-Log geschrieben (kein silent drop)."""
    import logging
    src_file = tmp_path / "module.py"
    src_file.write_text(
        "def main():\n"
        "    # WHY(#X): unklar wo das hin soll\n"
        "    for r in rows:\n"
        "        process(r)\n"
    )
    with caplog.at_level(logging.WARNING):
        edges = extract_rationale_edges(
            src_file, repo_slug="demo", workspace_id="bene",
        )
    assert edges == []
    assert any(
        "rationale-skipped" in rec.message and "non-trivial-target" in rec.message
        for rec in caplog.records
    )
