import pytest
from src.wiki_v2.renderer import to_mermaid, to_markdown


_SAMPLE_GRAPH = {
    "workspace_id": "ws-test",
    "nodes": [
        {"id": "src/api/server.py", "cluster_id": "api"},
        {"id": "src/memory/store.py", "cluster_id": "memory"},
    ],
    "edges": [
        {"source": "src/api/server.py", "target": "src/memory/store.py",
         "type": "import", "weight": 1.0},
    ],
    "clusters": [
        {"cluster_id": "api", "name": "API Layer", "members": ["src/api/server.py"]},
        {"cluster_id": "memory", "name": "Memory Layer", "members": ["src/memory/store.py"]},
    ],
}


def test_to_mermaid_starts_with_graph_lr():
    result = to_mermaid(_SAMPLE_GRAPH)
    assert result.startswith("graph LR")


def test_to_mermaid_contains_subgraph_per_cluster():
    result = to_mermaid(_SAMPLE_GRAPH)
    assert "subgraph api" in result
    assert "subgraph memory" in result


def test_to_mermaid_contains_node_labels():
    result = to_mermaid(_SAMPLE_GRAPH)
    assert "server.py" in result
    assert "store.py" in result


def test_to_mermaid_contains_edge():
    result = to_mermaid(_SAMPLE_GRAPH)
    # Edge from server.py to store.py must appear (IDs are sanitized)
    assert "src_api_server_py" in result
    assert "src_memory_store_py" in result
    assert "-->" in result


def test_to_mermaid_empty_graph():
    result = to_mermaid({"nodes": [], "edges": [], "clusters": []})
    assert result.startswith("graph LR")


def test_to_mermaid_call_edge_uses_dashed_arrow():
    graph = {
        "nodes": [{"id": "a.py"}, {"id": "b.py"}],
        "edges": [{"source": "a.py", "target": "b.py", "type": "call"}],
        "clusters": [],
    }
    result = to_mermaid(graph)
    assert "-.->|call|" in result


def test_to_markdown_contains_workspace():
    result = to_markdown(_SAMPLE_GRAPH)
    assert "ws-test" in result


def test_to_markdown_contains_cluster_names():
    result = to_markdown(_SAMPLE_GRAPH)
    assert "API Layer" in result
    assert "Memory Layer" in result
