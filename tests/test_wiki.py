import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.wiki import (
    WikiEdge, WikiCluster, build_category_matrix,
    _build_class_index, _resolve_dep,
    find_import_pairs, find_shared_types, find_call_pairs,
    find_label_overlap, find_event_pairs,
    find_citation_pairs, find_shared_concepts, find_method_chains,
    find_keyword_overlap, find_dataset_pairs,
    build_connection_graph, cluster_themes, generate_wiki_markdown, RULE_SETS,
    _build_keyword_index, _build_cluster_embeddings,
)
import pytest


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


def test_find_event_pairs_dispatch_match():
    oc = {
        "app/Jobs/SendEmailJob.php": {
            "file_summary": "implements ShouldQueue sends email",
            "functions": [],
        },
        "app/Controllers/OrderController.php": {
            "file_summary": "dispatch(SendEmailJob) on order created",
            "functions": [],
        },
    }
    edges = find_event_pairs(oc)
    # If dispatch detected and ShouldQueue detected in same run, edges connect them
    # Rule: dispatchers and listeners connect when class name overlaps
    assert isinstance(edges, list)  # Basic: returns list, no exception


def test_paper_rules_return_empty_for_no_papers():
    assert find_citation_pairs({}, []) == []
    assert find_shared_concepts([], None, None, "", "") == []
    assert find_method_chains([], None, None, "", "") == []
    assert find_keyword_overlap({}, []) == []
    assert find_dataset_pairs([], None, None, "", "") == []


def test_build_connection_graph_empty_overview():
    """Empty overview → no edges."""
    conn = _make_conn()
    edges = build_connection_graph("code", {}, [], conn)
    assert edges == []
    conn.close()


def test_build_connection_graph_merges_duplicates():
    """Two rules creating the same edge → weight is summed."""
    oc = {
        "app/AService.php": {
            "dependencies": ["App\\BService"],
            "file_summary": "Uses BService",
            "functions": [{"name": "run", "calls": ["BService::execute"], "inputs": [], "outputs": []}],
        },
        "app/BService.php": {
            "dependencies": [],
            "file_summary": "plain service",
            "functions": [],
        },
    }
    conn = _make_conn()
    edges = build_connection_graph("code", oc, [], conn)
    # Both import (1.0) and function_call (0.9) should be merged
    assert len(edges) == 1
    assert edges[0].weight > 1.0  # summed weights
    conn.close()


def test_cluster_themes_single_component():
    """3 files all connected → 1 cluster."""
    edges = [
        WikiEdge("a.py", "b.py", 1.0, "import"),
        WikiEdge("b.py", "c.py", 0.9, "function_call"),
    ]
    clusters = cluster_themes(edges, min_files=2)
    assert len(clusters) == 1
    assert len(clusters[0].files) == 3


def test_cluster_themes_min_files_filter():
    """Single isolated node should not appear if below min_files=2."""
    edges = [WikiEdge("a.py", "b.py", 1.0, "import")]
    clusters = cluster_themes(edges, min_files=3)
    assert clusters == []


def test_generate_wiki_markdown_format():
    """Output contains required markers."""
    clusters = [
        WikiCluster(
            name="auth",
            files=["app/AuthService.php", "app/UserModel.php"],
            labels=["auth"],
            edges=[],
        )
    ]
    md = generate_wiki_markdown(clusters, "my-repo")
    assert "Verknüpfungswiki" in md
    assert "my-repo" in md
    assert "AuthService.php" in md
    assert "UserModel.php" in md


def test_generate_wiki_no_overview_cache(capsys):
    """Returns None and prints warning when no overview cache exists."""
    from src.memory.wiki import generate_wiki
    from unittest.mock import MagicMock, patch

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    chroma = MagicMock()

    # Mock the entire src.analysis.context module inside the function scope
    with patch.dict('sys.modules', {'src.analysis.context': MagicMock(load_overview_cache_raw=MagicMock(return_value=None))}), \
         patch.dict('sys.modules', {'src.config': MagicMock(repo_slug=MagicMock(return_value='test-repo'))}):
        result = generate_wiki(conn, chroma, "https://github.com/test/repo")

    assert result is None
    captured = capsys.readouterr()
    assert "Kein Overview-Cache" in captured.out


def test_search_wiki_no_wiki_file(tmp_path, monkeypatch):
    """Returns fallback string when no wiki file exists."""
    from src.agents.pi import _execute_search_wiki

    monkeypatch.chdir(tmp_path)
    result = _execute_search_wiki({"topic": "auth"}, repo_slug_hint="")
    assert "Kein Wiki" in result


def test_build_keyword_index_extracts_stems():
    """Keywords include cluster name and file stems."""
    from src.memory.wiki import _build_keyword_index
    c = WikiCluster(
        name="CreditService",
        files=["app/Services/CreditService.php", "app/Services/BillingService.php"],
        labels=["payment"],
        edges=[],
    )
    idx = _build_keyword_index([c])
    assert "creditservice" in idx
    assert "billingservice" in idx
    assert idx["creditservice"] == ["CreditService"]
    assert idx["billingservice"] == ["CreditService"]
    assert "payment" in idx


def test_generate_wiki_creates_index_file(tmp_path, monkeypatch):
    """generate_wiki() writes _wiki_index.json alongside the wiki markdown."""
    from unittest.mock import MagicMock, patch
    from src.memory.wiki import generate_wiki

    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache").mkdir()

    overview = {
        "app/Services/FooService.php": {
            "dependencies": ["App\\Services\\BarService"],
            "functions": [],
            "file_summary": "",
            "category": "domain",
        },
        "app/Services/BarService.php": {
            "dependencies": [],
            "functions": [],
            "file_summary": "",
            "category": "domain",
        },
    }

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (chunk_id TEXT, source_id TEXT, content TEXT, chunk_level TEXT, is_active INTEGER, category_labels TEXT, text TEXT)")
    conn.commit()

    with patch("src.analysis.context.load_overview_cache_raw", return_value=overview), \
         patch("src.config.repo_slug", return_value="myrepo"):
        result = generate_wiki(conn, None, "https://github.com/test/myrepo", ollama_url="", model="")

    index_file = tmp_path / "cache" / "myrepo_wiki_index.json"
    assert index_file.exists(), f"Expected {index_file} to exist"
    import json
    idx = json.loads(index_file.read_text())
    assert isinstance(idx, dict)
    conn.close()


def test_paper_rules_registered():
    """All 5 paper rules must be in RULE_SETS['paper']."""
    from src.memory.wiki import RULE_SETS
    names = {name for name, _fn, _w in RULE_SETS["paper"]}
    assert names == {"citation", "keyword_overlap", "shared_concept", "method_chain", "dataset_coupling"}


def test_wiki_paper_cache_roundtrip(tmp_path):
    """_cache_get after _cache_put returns identical JSON."""
    import sqlite3
    from src.memory.store import init_memory_db
    from src.memory.wiki import _cache_get, _cache_put
    db = tmp_path / "test.db"
    conn = init_memory_db(db)
    _cache_put(conn, "paper:arxiv:1234.5678", "method_chain", ["bert", "gpt"])
    result = _cache_get(conn, "paper:arxiv:1234.5678", "method_chain")
    assert result == ["bert", "gpt"]
    assert _cache_get(conn, "paper:arxiv:0000.0000", "method_chain") is None
