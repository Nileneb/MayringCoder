from __future__ import annotations
import re as _re
from pathlib import Path


def _mermaid_id(nid: str) -> str:
    return _re.sub(r'[^a-zA-Z0-9]', '_', nid)


_MERMAID_ARROW = {
    "import": "-->",
    "call": "-.->",
    "test_covers": "==>",
    "concept_link": "~~~",
    "issue_mentions": "--o",
    "event_dispatch": "-->>",
    "label_cooccurrence": "---",
    "shared_type": "-->",
}


def to_mermaid(graph_data: dict) -> str:
    """Render graph.json dict to Mermaid flowchart with subgraphs per cluster."""
    lines = ["graph LR"]
    clusters = graph_data.get("clusters", [])
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    clustered: set[str] = set()
    for c in clusters:
        cid = _mermaid_id(c.get("cluster_id", c.get("name", "cluster")))
        cname = c.get("name", cid).replace('"', "'")
        members = c.get("members", [])
        lines.append(f'    subgraph {cid}["{cname}"]')
        for m in members:
            mid = _mermaid_id(m)
            label = Path(m).name.replace('"', "'")
            lines.append(f'        {mid}["{label}"]')
            clustered.add(m)
        lines.append("    end")

    for n in nodes:
        if n["id"] not in clustered:
            mid = _mermaid_id(n["id"])
            label = Path(n["id"]).name.replace('"', "'")
            lines.append(f'    {mid}["{label}"]')

    for e in edges:
        src = _mermaid_id(e.get("source", ""))
        tgt = _mermaid_id(e.get("target", ""))
        if src and tgt:
            arrow = _MERMAID_ARROW.get(e.get("type", "import"), "-->")
            etype = e.get("type", "")
            lines.append(f"    {src} {arrow}|{etype}| {tgt}")

    return "\n".join(lines)


def to_markdown(graph_data: dict) -> str:
    """Render graph.json dict to Markdown wiki."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    ws = graph_data.get("workspace_id", "")
    clusters = graph_data.get("clusters", [])
    edges = graph_data.get("edges", [])
    lines = [
        f"# Wiki — {ws}",
        f"_Stand: {ts} | {len(clusters)} Cluster · {len(edges)} Verbindungen_\n",
    ]
    for c in sorted(clusters, key=lambda x: len(x.get("members", [])), reverse=True):
        lines.append(f"## 🔗 {c['name']}")
        members = c.get("members", [])
        lines.append(f"**Dateien ({len(members)}):** {', '.join(Path(f).name for f in members[:10])}")
        if c.get("description"):
            lines.append(c["description"])
        lines.append("")
    return "\n".join(lines)
