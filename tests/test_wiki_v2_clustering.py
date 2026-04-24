import pytest
from src.wiki_v2.graph import WikiGraph
from src.wiki_v2.models import WikiNode, WikiEdge
from src.wiki_v2.clustering import ClusterEngine, cluster_quality


@pytest.fixture
def populated_graph(tmp_path):
    g = WikiGraph("ws-test", "repo", db_path=tmp_path / "wiki.db")
    for nid in ["src/api/server.py", "src/api/web_ui.py", "src/api/auth.py",
                "src/memory/store.py", "src/memory/retrieval.py", "src/memory/ingest.py"]:
        g.upsert_node(WikiNode(nid, "repo", "ws-test"))
    g.add_edge(WikiEdge("src/api/server.py", "src/api/auth.py", "repo", "ws-test", "import", 1.0))
    g.add_edge(WikiEdge("src/api/web_ui.py", "src/api/auth.py", "repo", "ws-test", "import", 1.0))
    g.add_edge(WikiEdge("src/memory/retrieval.py", "src/memory/store.py", "repo", "ws-test", "import", 1.0))
    g.add_edge(WikiEdge("src/memory/ingest.py", "src/memory/store.py", "repo", "ws-test", "import", 1.0))
    yield g
    g.close()


def test_louvain_clusters_produced(populated_graph):
    engine = ClusterEngine()
    clusters = engine.cluster(populated_graph, strategy="louvain")
    assert len(clusters) >= 1
    all_members = [m for c in clusters for m in c.members]
    assert len(all_members) == 6


def test_all_nodes_assigned(populated_graph):
    engine = ClusterEngine()
    clusters = engine.cluster(populated_graph, strategy="louvain")
    all_members = set(m for c in clusters for m in c.members)
    assert "src/api/server.py" in all_members
    assert "src/memory/store.py" in all_members


def test_clusters_persisted_in_graph(populated_graph):
    engine = ClusterEngine()
    clusters = engine.cluster(populated_graph, strategy="louvain")
    persisted = populated_graph.get_clusters()
    assert len(persisted) == len(clusters)


def test_cluster_quality_metrics(populated_graph):
    engine = ClusterEngine()
    clusters = engine.cluster(populated_graph, strategy="louvain")
    metrics = cluster_quality(clusters, populated_graph)
    assert "num_clusters" in metrics
    assert metrics["total_edges"] == 4
    assert metrics["internal_edges"] + metrics["external_edges"] <= 4


def test_empty_graph_no_crash(tmp_path):
    g = WikiGraph("ws-empty", "repo", db_path=tmp_path / "wiki.db")
    engine = ClusterEngine()
    clusters = engine.cluster(g)
    assert clusters == []
    g.close()


def test_single_node_graph(tmp_path):
    g = WikiGraph("ws-single", "repo", db_path=tmp_path / "wiki.db")
    g.upsert_node(WikiNode("src/main.py", "repo", "ws-single"))
    engine = ClusterEngine()
    clusters = engine.cluster(g)
    assert len(clusters) == 1
    g.close()
