import pytest
from pathlib import Path
from src.wiki_v2.graph import WikiGraph
from src.wiki_v2.models import WikiNode, WikiEdge
from src.wiki_v2.watcher import on_post_finding, on_post_analyze, on_post_ingest


@pytest.fixture
def tiny_graph(tmp_path):
    g = WikiGraph("ws-w", "repo", db_path=tmp_path / "wiki.db")
    g.upsert_node(WikiNode("src/api/server.py", "repo", "ws-w"))
    g.upsert_node(WikiNode("src/memory/store.py", "repo", "ws-w"))
    yield g
    g.close()


def test_on_post_finding_creates_issue_mentions_edges(tmp_path):
    """File path references in finding text → issue_mentions edges."""
    g = WikiGraph("ws-w", "repo", db_path=tmp_path / "wiki.db")
    g.upsert_node(WikiNode("src/api/server.py", "repo", "ws-w"))
    g.upsert_node(WikiNode("src/memory/store.py", "repo", "ws-w"))
    g.close()

    finding_text = "Das Problem liegt in src/memory/store.py — falsche SQL-Query."
    on_post_finding(
        workspace_id="ws-w",
        repo_slug="repo",
        analyzed_file="src/api/server.py",
        finding_text=finding_text,
        db_path=tmp_path / "wiki.db",
    )

    g2 = WikiGraph("ws-w", "repo", db_path=tmp_path / "wiki.db")
    edges = g2.edges_by_type("issue_mentions")
    g2.close()

    assert len(edges) >= 1
    assert any(e.source == "src/api/server.py" and "store.py" in e.target for e in edges)


def test_on_post_finding_ignores_self_reference(tmp_path):
    """Analyzed file itself is not added as an edge target."""
    g = WikiGraph("ws-w", "repo", db_path=tmp_path / "wiki.db")
    g.upsert_node(WikiNode("src/api/server.py", "repo", "ws-w"))
    g.close()

    on_post_finding(
        workspace_id="ws-w",
        repo_slug="repo",
        analyzed_file="src/api/server.py",
        finding_text="See src/api/server.py for details.",
        db_path=tmp_path / "wiki.db",
    )

    g2 = WikiGraph("ws-w", "repo", db_path=tmp_path / "wiki.db")
    edges = g2.edges_by_type("issue_mentions")
    self_edges = [e for e in edges if e.source == e.target]
    g2.close()
    assert self_edges == []


def test_recluster_flag_set_on_large_topology_change(tmp_path, monkeypatch):
    """recluster_needed flag is written when edge count grows >10%."""
    import src.wiki_v2.watcher as watcher_mod
    from src.config import WIKI_DIR as real_wiki_dir

    wiki_dir = tmp_path / "wiki"
    monkeypatch.setattr("src.config.WIKI_DIR", wiki_dir, raising=False)

    import src.config as conf
    conf.WIKI_DIR = wiki_dir

    try:
        g = WikiGraph("ws-flag", "repo", db_path=tmp_path / "wiki.db")
        # Seed 10 edges so that any new edge (>10%) triggers the flag
        for i in range(10):
            g.upsert_node(WikiNode(f"src/m/f{i}.py", "repo", "ws-flag"))
        for i in range(1, 10):
            g.add_edge(WikiEdge(f"src/m/f0.py", f"src/m/f{i}.py", "repo", "ws-flag", "import", 1.0))
        g.close()

        # finding that references 2 new files → 2 new edges = 22% increase
        on_post_finding(
            workspace_id="ws-flag",
            repo_slug="repo",
            analyzed_file="src/api/server.py",
            finding_text="Issues in src/m/f0.py and src/m/f1.py",
            db_path=tmp_path / "wiki.db",
        )

        flag = wiki_dir / "ws-flag" / "recluster_needed"
        assert flag.exists(), "recluster_needed flag was not written"
    finally:
        conf.WIKI_DIR = real_wiki_dir


def test_on_post_finding_no_crash_on_empty_text(tmp_path):
    """Empty finding text — no crash, no edges added."""
    on_post_finding(
        workspace_id="ws-empty",
        repo_slug="repo",
        analyzed_file="src/api/server.py",
        finding_text="",
        db_path=tmp_path / "wiki.db",
    )


def test_on_post_analyze_upserts_node(tmp_path):
    """on_post_analyze adds node for analyzed_file."""
    on_post_analyze(
        workspace_id="ws-a",
        repo_slug="repo",
        analyzed_file="src/api/server.py",
        db_path=tmp_path / "wiki.db",
    )
    g = WikiGraph("ws-a", "repo", db_path=tmp_path / "wiki.db")
    node = g.get_node("src/api/server.py")
    g.close()
    assert node is not None


def test_on_post_ingest_adds_node(tmp_path):
    """on_post_ingest creates node from source_id."""
    on_post_ingest(
        workspace_id="ws-b",
        repo_slug="repo",
        source_id="repo:repo:src/memory/store.py",
        db_path=tmp_path / "wiki.db",
    )
    g = WikiGraph("ws-b", "repo", db_path=tmp_path / "wiki.db")
    node = g.get_node("src/memory/store.py")
    g.close()
    assert node is not None


def test_on_post_analyze_stores_turbulence_tier(tmp_path):
    """turbulence_tier is persisted in the wiki node."""
    on_post_analyze(
        workspace_id="ws-c",
        repo_slug="repo",
        analyzed_file="src/api/server.py",
        db_path=tmp_path / "wiki.db",
        turbulence_tier="hot",
    )
    g = WikiGraph("ws-c", "repo", db_path=tmp_path / "wiki.db")
    node = g.get_node("src/api/server.py")
    g.close()
    assert node is not None
    assert node.turbulence_tier == "hot"


def test_on_post_ingest_no_chroma_creates_node_only(tmp_path):
    """on_post_ingest with chroma=None skips concept_link detection."""
    on_post_ingest(
        workspace_id="ws-d",
        repo_slug="repo",
        source_id="repo:repo:src/config.py",
        db_path=tmp_path / "wiki.db",
        chroma=None,
    )
    g = WikiGraph("ws-d", "repo", db_path=tmp_path / "wiki.db")
    node = g.get_node("src/config.py")
    edges = g.get_edges()
    g.close()
    assert node is not None
    assert edges == []
