from __future__ import annotations
from pathlib import Path


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
