import pytest
import sqlite3
from src.wiki_v2.graph import WikiGraph
from src.wiki_v2.models import WikiNode, WikiEdge, Cluster
from src.wiki_v2.history import WikiHistory, WikiDiff, team_activity, ConflictResolver
from src.wiki_v2 import store as wiki_store


@pytest.fixture
def hist_graph(tmp_path):
    g = WikiGraph("ws-hist", "repo", db_path=tmp_path / "wiki.db")
    for nid in ["src/a.py", "src/b.py", "src/c.py"]:
        g.upsert_node(WikiNode(nid, "repo", "ws-hist"))
    g.add_edge(WikiEdge("src/a.py", "src/b.py", "repo", "ws-hist", "import", 1.0))
    g.upsert_cluster(Cluster("grp1", "repo", "ws-hist", "Group 1",
                             members=["src/a.py", "src/b.py"]))
    yield g
    g.close()


def test_create_snapshot_returns_id(hist_graph, monkeypatch, tmp_path):
    import src.config as conf
    conf.WIKI_DIR = tmp_path / "wiki"
    hist = WikiHistory()
    sid = hist.create_snapshot(hist_graph, trigger="test")
    assert isinstance(sid, int)
    assert sid > 0


def test_snapshot_persisted_in_db(hist_graph, monkeypatch, tmp_path):
    import src.config as conf
    conf.WIKI_DIR = tmp_path / "wiki"
    hist = WikiHistory()
    hist.create_snapshot(hist_graph, trigger="rebuild")
    snaps = hist.timeline(hist_graph._conn, "ws-hist", limit=20)
    assert len(snaps) >= 1
    assert snaps[0].trigger == "rebuild"
    assert snaps[0].node_count == 3
    assert snaps[0].edge_count == 1


def test_timeline_newest_first(hist_graph, monkeypatch, tmp_path):
    import src.config as conf
    conf.WIKI_DIR = tmp_path / "wiki"
    hist = WikiHistory()
    hist.create_snapshot(hist_graph, trigger="first")
    hist.create_snapshot(hist_graph, trigger="second")
    snaps = hist.timeline(hist_graph._conn, "ws-hist")
    assert snaps[0].trigger == "second"


def test_timeline_respects_limit(hist_graph, monkeypatch, tmp_path):
    import src.config as conf
    conf.WIKI_DIR = tmp_path / "wiki"
    hist = WikiHistory()
    for i in range(5):
        hist.create_snapshot(hist_graph, trigger=f"snap{i}")
    snaps = hist.timeline(hist_graph._conn, "ws-hist", limit=3)
    assert len(snaps) <= 3


def test_cleanup_removes_oldest(hist_graph, monkeypatch, tmp_path):
    import src.config as conf
    conf.WIKI_DIR = tmp_path / "wiki"
    hist = WikiHistory()
    for i in range(5):
        hist.create_snapshot(hist_graph, trigger=f"snap{i}")
    deleted = hist.cleanup(hist_graph._conn, "ws-hist", keep=2)
    assert deleted == 3
    remaining = hist.timeline(hist_graph._conn, "ws-hist")
    assert len(remaining) == 2


def test_team_activity_counts_contributions(tmp_path):
    from src.wiki_v2 import store as s
    conn = s.init_wiki_db(tmp_path / "wiki.db")
    s.log_contribution(conn, "ws-team", "user-a", "upsert_node", "src/a.py")
    s.log_contribution(conn, "ws-team", "user-a", "upsert_node", "src/b.py")
    s.log_contribution(conn, "ws-team", "user-b", "add_edge", "src/a.py→src/b.py")
    activity = team_activity(conn, "ws-team")
    assert activity.get("user-a", 0) == 2
    assert activity.get("user-b", 0) == 1
    conn.close()


def test_conflict_resolver_manual_edge_wins():
    from src.wiki_v2.models import WikiEdge
    resolver = ConflictResolver()
    manual = WikiEdge("a.py", "b.py", "repo", "ws", "import", validated=True)
    auto = WikiEdge("a.py", "b.py", "repo", "ws", "import", validated=False)
    winner, needs_review = resolver.resolve_edge(manual, auto)
    assert winner is manual
    assert needs_review is False


def test_conflict_resolver_auto_updated_by_newer():
    from src.wiki_v2.models import WikiEdge
    resolver = ConflictResolver()
    old_auto = WikiEdge("a.py", "b.py", "repo", "ws", "import", validated=False)
    new_auto = WikiEdge("a.py", "b.py", "repo", "ws", "import", weight=2.0, validated=False)
    winner, needs_review = resolver.resolve_edge(old_auto, new_auto)
    assert winner is new_auto
    assert needs_review is False


def test_conflict_resolver_both_manual_needs_review():
    from src.wiki_v2.models import WikiEdge
    resolver = ConflictResolver()
    m1 = WikiEdge("a.py", "b.py", "repo", "ws", "import", validated=True)
    m2 = WikiEdge("a.py", "b.py", "repo", "ws", "call", validated=True)
    _, needs_review = resolver.resolve_edge(m1, m2)
    assert needs_review is True


def test_wiki_history_endpoint_registered():
    from src.api.server import app
    paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/wiki/history" in paths
    assert "/wiki/diff" in paths
    assert "/wiki/team" in paths


def test_cli_history_flags():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--help"],
        capture_output=True, text=True, cwd="/home/nileneb/Desktop/MayringCoder"
    )
    assert "--wiki-history" in result.stdout
    assert "--wiki-team-activity" in result.stdout
    assert "--wiki-history-cleanup" in result.stdout
