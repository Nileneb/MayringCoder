import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.wiki import (
    WikiEdge, WikiCluster, build_category_matrix,
    _build_class_index, _resolve_dep,
    find_import_pairs, find_shared_types, find_call_pairs,
    find_label_overlap, find_event_pairs,
)


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


def test_build_class_index():
    oc = {"app/Services/CreditService.php": {}, "app/Models/User.php": {}}
    idx = _build_class_index(oc)
    assert idx["creditservice"] == "app/Services/CreditService.php"
    assert idx["user"] == "app/Models/User.php"


def test_resolve_dep_fqn():
    idx = {"creditservice": "app/Services/CreditService.php"}
    assert _resolve_dep("App\\Services\\CreditService", idx) == "app/Services/CreditService.php"
    assert _resolve_dep("UnknownClass", idx) is None


def test_find_import_pairs():
    oc2 = {
        "app/BService.php": {"dependencies": [], "functions": []},
        "app/AController.php": {"dependencies": ["App\\BService"], "functions": []},
    }
    edges = find_import_pairs(oc2)
    assert len(edges) == 1
    assert edges[0].rule == "import"
    assert edges[0].weight == 1.0


def test_find_shared_types_no_shared():
    oc = {
        "a.php": {"file_summary": "Uses AuthService", "functions": []},
        "b.php": {"file_summary": "Uses PaymentService", "functions": []},
    }
    edges = find_shared_types(oc)
    assert all(e.rule == "shared_type" for e in edges)
    assert edges == []


def test_find_shared_types_with_shared():
    oc = {
        "a.php": {"file_summary": "Uses CreditService", "functions": []},
        "b.php": {"file_summary": "Uses CreditService for billing", "functions": []},
    }
    edges = find_shared_types(oc)
    assert len(edges) == 1
    assert edges[0].weight == 0.8


def test_find_call_pairs():
    oc = {
        "app/AService.php": {
            "dependencies": [],
            "functions": [{"name": "run", "calls": ["BService::execute"], "inputs": [], "outputs": []}],
        },
        "app/BService.php": {"dependencies": [], "functions": []},
    }
    edges = find_call_pairs(oc)
    assert len(edges) == 1
    assert edges[0].rule == "function_call"
    assert edges[0].weight == 0.9


def test_find_label_overlap():
    conn = _make_conn()
    conn.execute("INSERT INTO chunks VALUES ('c1', 'src1', 'auth', 1)")
    conn.execute("INSERT INTO chunks VALUES ('c2', 'src2', 'auth', 1)")
    conn.execute("INSERT INTO chunks VALUES ('c3', 'src3', 'domain', 1)")
    edges = find_label_overlap(conn, {})
    assert any(e.rule == "label_cooccurrence" for e in edges)
    conn.close()


def test_find_event_pairs_empty():
    oc = {"a.php": {"file_summary": "plain service", "functions": []}}
    edges = find_event_pairs(oc)
    assert edges == []
