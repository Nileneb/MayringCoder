import pytest
from src.wiki_v2.edge_detector import EdgeDetector
from src.wiki_v2.models import WikiEdge


@pytest.fixture
def detector():
    return EdgeDetector()


def test_detect_imports_basic(detector):
    oc = {
        "src/api/server.py": {"dependencies": ["MemoryStore"], "functions": []},
        "src/memory/store.py": {"dependencies": [], "functions": []},
    }
    edges = detector.detect_from_overview(oc, None, "ws-test", "repo")
    import_edges = [e for e in edges if e.type == "import"]
    assert len(import_edges) == 1
    assert import_edges[0].source == "src/api/server.py"
    assert import_edges[0].target == "src/memory/store.py"
    assert import_edges[0].weight == 1.0


def test_detect_no_self_loops(detector):
    oc = {
        "src/api/server.py": {"dependencies": ["server"], "functions": []},
    }
    edges = detector.detect_from_overview(oc, None, "ws-test", "repo")
    for e in edges:
        assert e.source != e.target


def test_detect_test_coverage(detector):
    files = ["src/memory/store.py", "tests/test_store.py", "src/api/server.py"]
    edges = detector.detect_test_coverage(files, "ws-test", "repo")
    assert len(edges) == 1
    assert edges[0].source == "tests/test_store.py"
    assert edges[0].target == "src/memory/store.py"
    assert edges[0].type == "test_covers"


def test_detect_calls(detector):
    oc = {
        "src/pipeline.py": {
            "dependencies": [],
            "functions": [{"name": "run", "inputs": [], "outputs": [], "calls": ["MemoryStore"]}],
        },
        "src/memory/store.py": {"dependencies": [], "functions": []},
    }
    edges = detector.detect_from_overview(oc, None, "ws", "r")
    call_edges = [e for e in edges if e.type == "call"]
    assert len(call_edges) == 1
    assert call_edges[0].target == "src/memory/store.py"


def test_deduplication(detector):
    oc = {
        "src/a.py": {"dependencies": ["B", "B"], "functions": []},
        "src/b.py": {"dependencies": [], "functions": []},
    }
    edges = detector.detect_from_overview(oc, None, "ws-test", "repo")
    import_edges = [e for e in edges if e.type == "import" and e.source == "src/a.py"]
    assert len(import_edges) == 1


def test_workspace_id_on_edges(detector):
    oc = {
        "src/a.py": {"dependencies": ["B"], "functions": []},
        "src/b.py": {"dependencies": [], "functions": []},
    }
    edges = detector.detect_from_overview(oc, None, "my-workspace", "myrepo")
    assert all(e.workspace_id == "my-workspace" for e in edges)
    assert all(e.repo_slug == "myrepo" for e in edges)


# --- #72 Akzeptanzkriterium: edge_stats() ---

def test_edge_stats_returns_correct_structure(tmp_path):
    """edge_stats() gibt Übersicht mit allen Pflichtfeldern zurück."""
    from src.wiki_v2.graph import WikiGraph
    from src.wiki_v2.models import WikiNode, WikiEdge
    from src.wiki_v2.edge_detector import edge_stats

    g = WikiGraph("ws-test", "repo", db_path=tmp_path / "wiki.db")
    g.upsert_node(WikiNode("src/a.py", "repo", "ws-test"))
    g.upsert_node(WikiNode("src/b.py", "repo", "ws-test"))
    g.upsert_node(WikiNode("src/c.py", "repo", "ws-test"))  # isolated
    g.add_edge(WikiEdge("src/a.py", "src/b.py", "repo", "ws-test", "import", weight=1.0))

    stats = edge_stats(g)
    assert stats["total_edges"] == 1
    assert stats["by_type"]["import"] == 1
    assert stats["isolated_nodes"] == 1  # src/c.py hat keine Edges
    assert len(stats["most_connected"]) == 2
    assert stats["avg_weight"] == 1.0
    g.close()


def test_edge_stats_empty_graph(tmp_path):
    """edge_stats() bricht bei leerem Graph nicht ab."""
    from src.wiki_v2.graph import WikiGraph
    from src.wiki_v2.edge_detector import edge_stats

    g = WikiGraph("ws-empty", "repo", db_path=tmp_path / "wiki.db")
    stats = edge_stats(g)
    assert stats["total_edges"] == 0
    assert stats["isolated_nodes"] == 0
    g.close()
