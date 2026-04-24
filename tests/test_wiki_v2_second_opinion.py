import pytest
from src.wiki_v2.graph import WikiGraph
from src.wiki_v2.models import WikiNode, WikiEdge, Cluster
from src.wiki_v2.second_opinion import WikiSecondOpinion, ClusterVerdict, EdgeVerdict


@pytest.fixture
def so_graph(tmp_path):
    g = WikiGraph("ws-so", "repo", db_path=tmp_path / "wiki.db")
    for nid in ["src/a.py", "src/b.py", "src/c.py", "src/d.py"]:
        g.upsert_node(WikiNode(nid, "repo", "ws-so", summary=f"Module {nid}"))
    g.add_edge(WikiEdge("src/a.py", "src/b.py", "repo", "ws-so", "concept_link", 0.8))
    g.add_edge(WikiEdge("src/c.py", "src/d.py", "repo", "ws-so", "concept_link", 0.75))
    g.upsert_cluster(Cluster("grp1", "repo", "ws-so", "Group 1",
                             members=["src/a.py", "src/b.py"]))
    g.upsert_cluster(Cluster("grp2", "repo", "ws-so", "Group 2",
                             members=["src/c.py", "src/d.py"]))
    yield g
    g.close()


class MockOllamaSecondOpinion(WikiSecondOpinion):
    """Override _call_ollama to return deterministic verdicts without a real server."""
    def __init__(self, verdict_word: str = "BESTÄTIGT"):
        self._verdict_word = verdict_word

    def _call_ollama(self, model, ollama_url, prompt, timeout=60):
        return f"{self._verdict_word}: Das ist korrekt."


def test_validate_clusters_returns_verdicts_for_each_cluster(so_graph):
    so = MockOllamaSecondOpinion("BESTÄTIGT")
    clusters = so_graph.get_clusters()
    verdicts = so.validate_clusters(clusters, so_graph, "test-model", "http://localhost")
    assert len(verdicts) == len(clusters)
    assert all(isinstance(v, ClusterVerdict) for v in verdicts)


def test_validate_clusters_confirmed_verdict(so_graph):
    so = MockOllamaSecondOpinion("BESTÄTIGT")
    clusters = so_graph.get_clusters()
    verdicts = so.validate_clusters(clusters, so_graph, "test-model", "http://localhost")
    assert all(v.action == "BESTÄTIGT" for v in verdicts)


def test_validate_clusters_split_verdict(so_graph):
    so = MockOllamaSecondOpinion("SPLIT_VORSCHLAG")
    clusters = so_graph.get_clusters()
    verdicts = so.validate_clusters(clusters, so_graph, "test-model", "http://localhost")
    assert all(v.action == "SPLIT_VORSCHLAG" for v in verdicts)


def test_validate_edges_returns_verdicts(so_graph):
    so = MockOllamaSecondOpinion("BESTÄTIGT")
    edges = so_graph.get_edges()
    verdicts = so.validate_edges(edges, so_graph, "test-model", "http://localhost",
                                  edge_types=["concept_link"])
    assert len(verdicts) == 2
    assert all(isinstance(v, EdgeVerdict) for v in verdicts)


def test_validate_edges_only_targets_specified_types(so_graph):
    """import edges should not be validated."""
    so_graph.add_edge(WikiEdge("src/a.py", "src/c.py", "repo", "ws-so", "import", 1.0))
    so = MockOllamaSecondOpinion("BESTÄTIGT")
    edges = so_graph.get_edges()
    verdicts = so.validate_edges(edges, so_graph, "test-model", "http://localhost",
                                  edge_types=["concept_link"])
    validated_types = {v.edge_type for v in verdicts}
    assert "import" not in validated_types


def test_validate_edges_sets_validated_flag_on_confirmed(so_graph):
    """BESTÄTIGT verdict updates validated=1 in DB."""
    so = MockOllamaSecondOpinion("BESTÄTIGT")
    edges = so_graph.get_edges()
    so.validate_edges(edges, so_graph, "test-model", "http://localhost",
                      edge_types=["concept_link"])
    row = so_graph._conn.execute(
        "SELECT validated FROM wiki_edges WHERE source='src/a.py' AND target='src/b.py' "
        "AND workspace_id='ws-so'"
    ).fetchone()
    assert row is not None
    assert row[0] == 1


def test_second_opinion_report_counts(so_graph):
    so = MockOllamaSecondOpinion("BESTÄTIGT")
    c_verdicts = [
        ClusterVerdict("g1", "BESTÄTIGT"),
        ClusterVerdict("g2", "SPLIT_VORSCHLAG"),
    ]
    e_verdicts = [
        EdgeVerdict("a", "b", "concept_link", "BESTÄTIGT"),
        EdgeVerdict("c", "d", "concept_link", "ABGELEHNT"),
        EdgeVerdict("e", "f", "concept_link", "PRÄZISIERT"),
    ]
    report = so.second_opinion_report(c_verdicts, e_verdicts)
    assert "Bestätigt: 1" in report
    assert "Split: 1" in report
    assert "Abgelehnt: 1" in report
    assert "Präzisiert: 1" in report


def test_cli_has_wiki_second_opinion_flag():
    """--wiki-second-opinion flag is registered in CLI."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--help"],
        capture_output=True, text=True, cwd="/home/nileneb/Desktop/MayringCoder"
    )
    assert "--wiki-second-opinion" in result.stdout
