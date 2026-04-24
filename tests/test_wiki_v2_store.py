import json
import pytest
from pathlib import Path
from src.wiki_v2.graph import WikiGraph
from src.wiki_v2.models import WikiNode, WikiEdge, Cluster


@pytest.fixture
def tmp_graph(tmp_path):
    g = WikiGraph(workspace_id="ws-test", repo_slug="testrepo",
                  db_path=tmp_path / "wiki_test.db")
    yield g
    g.close()


def test_upsert_and_get_node(tmp_graph):
    node = WikiNode(id="src/foo.py", repo_slug="testrepo", workspace_id="ws-test",
                    labels=["api"], summary="Foo module")
    tmp_graph.upsert_node(node)
    found = tmp_graph.get_node("src/foo.py")
    assert found is not None
    assert found.summary == "Foo module"
    assert "api" in found.labels


def test_workspace_isolation(tmp_path):
    g1 = WikiGraph("ws-a", "repo", db_path=tmp_path / "wiki.db")
    g2 = WikiGraph("ws-b", "repo", db_path=tmp_path / "wiki.db")
    g1.upsert_node(WikiNode("src/a.py", "repo", "ws-a"))
    assert g1.node_count() == 1
    assert g2.node_count() == 0
    g1.close(); g2.close()


def test_add_edge(tmp_graph):
    tmp_graph.upsert_node(WikiNode("src/a.py", "testrepo", "ws-test"))
    tmp_graph.upsert_node(WikiNode("src/b.py", "testrepo", "ws-test"))
    edge = WikiEdge("src/a.py", "src/b.py", "testrepo", "ws-test", "import", weight=1.0)
    tmp_graph.add_edge(edge)
    assert tmp_graph.edge_count() == 1
    edges = tmp_graph.edges_by_type("import")
    assert len(edges) == 1
    assert edges[0].source == "src/a.py"


def test_upsert_cluster(tmp_graph):
    tmp_graph.upsert_node(WikiNode("src/a.py", "testrepo", "ws-test"))
    tmp_graph.upsert_node(WikiNode("src/b.py", "testrepo", "ws-test"))
    cluster = Cluster("api-layer", "testrepo", "ws-test", "API Layer",
                      members=["src/a.py", "src/b.py"])
    tmp_graph.upsert_cluster(cluster)
    clusters = tmp_graph.get_clusters()
    assert len(clusters) == 1
    assert clusters[0].name == "API Layer"
    assert len(clusters[0].members) == 2


def test_to_json_writes_file(tmp_graph, tmp_path, monkeypatch):
    import src.wiki_v2.graph as gmod
    wiki_dir = tmp_path / "wiki"
    monkeypatch.setattr(gmod, "WIKI_DIR", wiki_dir)
    tmp_graph.upsert_node(WikiNode("src/a.py", "testrepo", "ws-test", labels=["api"]))
    edge = WikiEdge("src/a.py", "src/b.py", "testrepo", "ws-test", "import")
    tmp_graph.upsert_node(WikiNode("src/b.py", "testrepo", "ws-test"))
    tmp_graph.add_edge(edge)
    data = tmp_graph.to_json()
    graph_file = wiki_dir / "ws-test" / "graph.json"
    assert graph_file.exists()
    loaded = json.loads(graph_file.read_text())
    assert len(loaded["nodes"]) == 2
    assert len(loaded["edges"]) == 1
    assert loaded["workspace_id"] == "ws-test"


def test_duplicate_edge_upsert(tmp_graph):
    tmp_graph.upsert_node(WikiNode("src/a.py", "testrepo", "ws-test"))
    tmp_graph.upsert_node(WikiNode("src/b.py", "testrepo", "ws-test"))
    e = WikiEdge("src/a.py", "src/b.py", "testrepo", "ws-test", "import", weight=1.0)
    tmp_graph.add_edge(e)
    tmp_graph.add_edge(e)  # duplicate — should not throw
    assert tmp_graph.edge_count() == 1


# --- #71 Akzeptanzkriterien: wiki_contributions + user_id ---

def test_wiki_contributions_table_exists(tmp_path):
    """wiki_contributions-Tabelle wird bei DB-Init angelegt."""
    from src.wiki_v2 import store as s
    db_path = tmp_path / "wiki.db"
    conn = s.init_wiki_db(db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "wiki_contributions" in tables
    conn.close()


def test_upsert_node_logs_contribution(tmp_path):
    """upsert_node schreibt Eintrag in wiki_contributions."""
    from src.wiki_v2 import store as s
    db_path = tmp_path / "wiki.db"
    conn = s.init_wiki_db(db_path)
    node = WikiNode("src/a.py", "repo", "ws-a")
    s.upsert_node(conn, node, user_id="user-42")
    rows = conn.execute("SELECT * FROM wiki_contributions WHERE user_id='user-42'").fetchall()
    assert len(rows) == 1
    assert rows[0]["action"] == "upsert_node"
    assert rows[0]["target_node"] == "src/a.py"
    conn.close()


def test_upsert_cluster_logs_contribution(tmp_path):
    """upsert_cluster schreibt Eintrag in wiki_contributions."""
    from src.wiki_v2 import store as s
    db_path = tmp_path / "wiki.db"
    conn = s.init_wiki_db(db_path)
    cluster = Cluster("api-layer", "repo", "ws-a", "API Layer")
    s.upsert_cluster(conn, cluster, user_id="user-99")
    rows = conn.execute("SELECT * FROM wiki_contributions WHERE user_id='user-99'").fetchall()
    assert len(rows) == 1
    assert rows[0]["action"] == "upsert_cluster"
    conn.close()
