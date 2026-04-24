import json
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


# --- #73 Akzeptanzkriterien: Embedding-Layer + clusters.json + 10-Node-Szenario ---

class MockChroma:
    """Minimaler ChromaDB-Mock mit vorberechneten Embeddings."""
    def __init__(self, emb_map: dict):
        # emb_map: {source_id -> embedding_vector}
        self._emb_map = emb_map

    def get(self, where=None, include=None):
        embeddings, metadatas = [], []
        for sid, emb in self._emb_map.items():
            embeddings.append(emb)
            metadatas.append({"source_id": sid, "workspace_id": "ws-test"})
        return {"embeddings": embeddings, "metadatas": metadatas}


def _make_vec(val: float, dim: int = 4) -> list[float]:
    return [val] * dim


@pytest.fixture
def ten_node_graph(tmp_path):
    """10-Dateien-Graph in 2 natürlichen Gruppen (api: 5 Nodes, memory: 5 Nodes)."""
    g = WikiGraph("ws-test", "repo", db_path=tmp_path / "wiki10.db")
    api_files = [f"src/api/f{i}.py" for i in range(5)]
    mem_files = [f"src/memory/f{i}.py" for i in range(5)]
    for f in api_files + mem_files:
        g.upsert_node(WikiNode(f, "repo", "ws-test"))
    # Import-Kanten innerhalb jeder Gruppe
    for i in range(1, 5):
        g.add_edge(WikiEdge(api_files[0], api_files[i], "repo", "ws-test", "import", 1.0))
        g.add_edge(WikiEdge(mem_files[0], mem_files[i], "repo", "ws-test", "import", 1.0))
    yield g
    g.close()


def test_10_node_louvain_produces_2_clusters(ten_node_graph):
    """Mit 10 Nodes in 2 klar getrennten Gruppen → min. 2 Cluster."""
    engine = ClusterEngine()
    clusters = engine.cluster(ten_node_graph, strategy="louvain")
    assert len(clusters) >= 2
    all_members = [m for c in clusters for m in c.members]
    assert len(all_members) == 10


def test_clusters_json_written(ten_node_graph, tmp_path, monkeypatch):
    """cluster() schreibt clusters.json pro Workspace."""
    import src.wiki_v2.clustering as cm
    wiki_dir = tmp_path / "wiki"
    monkeypatch.setattr(cm, "WIKI_DIR", wiki_dir, raising=False)

    # Patch WIKI_DIR im clustering Modul
    import importlib, src.config as conf
    real_wiki_dir = conf.WIKI_DIR
    conf.WIKI_DIR = wiki_dir
    try:
        engine = ClusterEngine()
        engine.cluster(ten_node_graph, strategy="louvain")
        out = wiki_dir / "ws-test" / "clusters.json"
        assert out.exists(), f"clusters.json not found at {out}"
        data = json.loads(out.read_text())
        assert len(data) >= 2
        assert "cluster_id" in data[0]
        assert "members" in data[0]
    finally:
        conf.WIKI_DIR = real_wiki_dir


def test_embedding_layer_merges_similar_nodes(tmp_path):
    """strategy='full' nutzt Embedding-Layer zur Cluster-Verfeinerung."""
    g = WikiGraph("ws-test", "repo", db_path=tmp_path / "wiki_emb.db")

    # 4 Nodes ohne Edges (Louvain kann keine Strukturen finden → 4 Singles)
    for nid in ["src/a.py", "src/b.py", "src/c.py", "src/d.py"]:
        g.upsert_node(WikiNode(nid, "repo", "ws-test"))

    # Embeddings: a+b ähnlich (vec 1.0), c+d ähnlich (vec 0.0)
    mock_chroma = MockChroma({
        "repo:repo:src/a.py": _make_vec(1.0),
        "repo:repo:src/b.py": _make_vec(0.98),  # sehr ähnlich zu a
        "repo:repo:src/c.py": _make_vec(0.0),
        "repo:repo:src/d.py": _make_vec(0.02),  # sehr ähnlich zu c
    })

    engine = ClusterEngine()
    clusters = engine.cluster(g, strategy="full", chroma=mock_chroma)

    # Mit Embedding-Merging sollte es 2 Cluster geben (a+b, c+d)
    # (ohne Embedding wären es 4 Singles)
    all_members = [m for c in clusters for m in c.members]
    assert len(all_members) == 4
    assert len(clusters) <= 4  # darf nicht mehr als 4 geben
    # a und b sollten im selben Cluster sein
    ab_clusters = [c for c in clusters if "src/a.py" in c.members]
    assert len(ab_clusters) == 1
    # Wenn Embedding funktioniert: b auch drin
    ab_cluster = ab_clusters[0]
    assert "src/b.py" in ab_cluster.members, "Embedding-Layer hat a+b nicht gemergt"
    g.close()


def test_strategy_full_without_chroma_falls_back_to_louvain(ten_node_graph):
    """strategy='full' ohne chroma → graceful fallback auf Louvain-only."""
    engine = ClusterEngine()
    clusters = engine.cluster(ten_node_graph, strategy="full", chroma=None)
    assert len(clusters) >= 1  # kein Crash, Louvain-Ergebnis
