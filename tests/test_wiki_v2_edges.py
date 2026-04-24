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
