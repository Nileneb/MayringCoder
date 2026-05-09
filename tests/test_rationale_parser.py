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
