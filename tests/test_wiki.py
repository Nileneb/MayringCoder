import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.wiki import WikiEdge, WikiCluster, build_category_matrix


def _make_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            source_id TEXT,
            category_labels TEXT,
            is_active INTEGER DEFAULT 1
        )
    """)
    return conn


def test_build_category_matrix_empty():
    conn = _make_conn()
    result = build_category_matrix(conn)
    assert result == {}
    conn.close()


def test_build_category_matrix_groups_by_label():
    conn = _make_conn()
    conn.execute("INSERT INTO chunks VALUES ('c1', 'src1', 'auth,api', 1)")
    conn.execute("INSERT INTO chunks VALUES ('c2', 'src2', 'auth,domain', 1)")
    conn.execute("INSERT INTO chunks VALUES ('c3', 'src3', '', 1)")  # empty labels
    conn.execute("INSERT INTO chunks VALUES ('c4', 'src4', 'api', 0)")  # inactive
    result = build_category_matrix(conn)
    assert set(result['auth']) == {'src1', 'src2'}
    assert set(result['api']) == {'src1'}
    assert 'domain' in result
    assert 'src4' not in result.get('api', [])  # inactive excluded
    conn.close()


def test_wiki_edge_dataclass():
    edge = WikiEdge(file_a="a.py", file_b="b.py", weight=1.0, rule="import")
    assert edge.file_a == "a.py"
    assert edge.weight == 1.0


def test_wiki_cluster_dataclass():
    cluster = WikiCluster(name="auth", files=["a.py", "b.py"], labels=["auth"], edges=[])
    assert cluster.name == "auth"
    assert len(cluster.files) == 2
