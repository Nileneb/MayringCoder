import pytest
from src.wiki_v2.graph import WikiGraph
from src.wiki_v2.models import WikiNode, WikiEdge, Cluster
from src.wiki_v2.injection import WikiContextInjector


@pytest.fixture
def injector_graph(tmp_path):
    g = WikiGraph("ws-inj", "repo", db_path=tmp_path / "wiki.db")
    g.upsert_node(WikiNode("src/api/server.py", "repo", "ws-inj",
                           summary="FastAPI-Server", turbulence_tier="hot"))
    g.upsert_node(WikiNode("src/api/auth.py", "repo", "ws-inj", summary="Auth logic"))
    g.upsert_node(WikiNode("src/memory/store.py", "repo", "ws-inj",
                           summary="SQLite store", turbulence_tier="hot"))
    g.upsert_node(WikiNode("src/api/web_ui.py", "repo", "ws-inj", summary="Gradio UI"))
    g.add_edge(WikiEdge("src/api/server.py", "src/api/auth.py", "repo", "ws-inj", "import", 1.0))
    g.add_edge(WikiEdge("src/api/server.py", "src/memory/store.py", "repo", "ws-inj", "import", 1.0))
    g.upsert_cluster(Cluster("api", "repo", "ws-inj", "API Layer",
                             description="HTTP endpoints",
                             members=["src/api/server.py", "src/api/auth.py", "src/api/web_ui.py"]))
    yield g
    g.close()


def test_build_context_contains_cluster_name(injector_graph):
    inj = WikiContextInjector()
    ctx = inj.build_context("src/api/server.py", injector_graph)
    assert "API Layer" in ctx


def test_build_context_contains_dependencies(injector_graph):
    inj = WikiContextInjector()
    ctx = inj.build_context("src/api/server.py", injector_graph)
    assert "auth.py" in ctx or "store.py" in ctx


def test_build_context_hot_zone_neighbors(injector_graph):
    inj = WikiContextInjector()
    ctx = inj.build_context("src/api/server.py", injector_graph)
    assert "store.py" in ctx or "Hot-Zone" in ctx


def test_build_context_max_chars_respected(injector_graph):
    inj = WikiContextInjector()
    ctx = inj.build_context("src/api/server.py", injector_graph, max_chars=50)
    assert len(ctx) <= 50


def test_build_context_empty_for_unknown_file(injector_graph):
    inj = WikiContextInjector()
    ctx = inj.build_context("src/nonexistent.py", injector_graph)
    assert ctx == ""


def test_build_context_empty_graph(tmp_path):
    g = WikiGraph("ws-empty", "repo", db_path=tmp_path / "wiki.db")
    inj = WikiContextInjector()
    ctx = inj.build_context("src/foo.py", g)
    assert ctx == ""
    g.close()


def test_analyze_with_memory_accepts_wiki_context():
    """analyze_with_memory accepts wiki_context kwarg without error (signature test)."""
    import inspect
    from src.agents.pi import analyze_with_memory
    sig = inspect.signature(analyze_with_memory)
    assert "wiki_context" in sig.parameters


def test_analyze_file_accepts_wiki_context():
    """analyze_file accepts wiki_context kwarg without error (signature test)."""
    import inspect
    from src.analysis.analyzer import analyze_file
    sig = inspect.signature(analyze_file)
    assert "wiki_context" in sig.parameters


def test_analyze_files_accepts_wiki_context_map():
    """analyze_files accepts wiki_context_map kwarg."""
    import inspect
    from src.analysis.analyzer import analyze_files
    sig = inspect.signature(analyze_files)
    assert "wiki_context_map" in sig.parameters
