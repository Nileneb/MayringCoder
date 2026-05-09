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


def test_extractor_writes_relative_path_not_basename(tmp_path: Path) -> None:
    """Code-review high-issue: parser schrieb file_path.name (basename),
    aber retrieval-JOIN matcht source.path = full repo-relative-path.
    Production hätte 0 matches gehabt. Regression-guard: extracted edges
    haben die FULL relative path."""
    from src.wiki_v2 import store as wstore
    from src.wiki_v2.rationale_parser import extract_rationale_edges_for_repo

    repo_root = tmp_path / "repo"
    (repo_root / "src" / "deep").mkdir(parents=True)
    (repo_root / "src" / "deep" / "module.py").write_text(
        "# WHY(#test): regression\n"
        "VAR = 1\n"
    )
    db_path = tmp_path / "wiki.db"
    wadapter = wstore.init_wiki_db(db_path)
    extract_rationale_edges_for_repo(
        repo_root, wadapter, repo_slug="demo", workspace_id="bene",
    )
    rows = wadapter.execute(
        "SELECT source FROM wiki_edges WHERE type='rationale'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "src/deep/module.py", (
        f"source must be repo-relative full path, got {rows[0][0]!r}"
    )


def test_extractor_skips_vendor_dirs(tmp_path: Path) -> None:
    """Code-review high-issue: rglob('*.py') unter cwd zog venv/site-packages
    mit. WHY-marker in DritteParteien-Code würden ungewollt persistiert."""
    from src.wiki_v2 import store as wstore
    from src.wiki_v2.rationale_parser import extract_rationale_edges_for_repo

    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "real.py").write_text(
        "# WHY(#real): keep me\n"
        "REAL = 1\n"
    )
    for vendor in ("venv", ".venv", "node_modules", "site-packages",
                   "__pycache__", "build", "dist"):
        (repo_root / vendor).mkdir(parents=True)
        (repo_root / vendor / "vendored.py").write_text(
            "# WHY(#vendor): IGNORE ME\n"
            "VENDORED = 1\n"
        )
    db_path = tmp_path / "wiki.db"
    wadapter = wstore.init_wiki_db(db_path)
    extract_rationale_edges_for_repo(
        repo_root, wadapter, repo_slug="demo", workspace_id="bene",
    )
    rows = wadapter.execute(
        "SELECT source FROM wiki_edges WHERE type='rationale'"
    ).fetchall()
    sources = [r[0] for r in rows]
    assert sources == ["src/real.py"], (
        f"vendor dirs must be skipped, got: {sources}"
    )


def test_extractor_handles_multiline_why_block_before_far_target(tmp_path: Path) -> None:
    """Self-test fand: 6-zeilen WHY-Block + lookahead von marker_line crashte
    weil target > 5 lines weiter. Fix: lookahead-Anker ist last_comment_line."""
    from src.wiki_v2 import store as wstore
    from src.wiki_v2.rationale_parser import extract_rationale_edges_for_repo

    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "long.py").write_text(
        "# WHY(#x, security): line1\n"
        "# line2\n"
        "# line3\n"
        "# line4\n"
        "# line5\n"
        "# line6\n"
        "# line7 (delta=7 from marker)\n"
        "VAR = 1\n"
    )
    db_path = tmp_path / "wiki.db"
    wadapter = wstore.init_wiki_db(db_path)
    extract_rationale_edges_for_repo(
        repo_root, wadapter, repo_slug="demo", workspace_id="bene",
    )
    rows = wadapter.execute(
        "SELECT target FROM wiki_edges WHERE type='rationale'"
    ).fetchall()
    assert len(rows) == 1, "long WHY-block must still match the next-line target"
    assert rows[0][0] == "long.VAR"


def test_edge_extractor_persists_rationale_edges(tmp_path: Path) -> None:
    """edge_extractor-Aufruf für ein Repo mit WHY-marker → wiki_edges DB
    enthält rationale-rows."""
    import sqlite3
    from src.wiki_v2 import store as wstore
    from src.wiki_v2.rationale_parser import extract_rationale_edges_for_repo

    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "module.py").write_text(
        "# WHY(#185): path-traversal defence\n"
        "_SLUG_RE = 1\n"
    )

    db_path = tmp_path / "wiki.db"
    wadapter = wstore.init_wiki_db(db_path)

    n = extract_rationale_edges_for_repo(
        repo_root, wadapter, repo_slug="demo", workspace_id="bene",
    )
    assert n == 1

    rows = wadapter.execute(
        "SELECT source, target, type, rationale, context FROM wiki_edges "
        "WHERE type='rationale'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][3] == "path-traversal defence"
