from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.wiki_v2.graph import WikiGraph


class WikiContextInjector:
    """Builds a wiki context block for analysis prompts from WikiGraph data."""

    def build_context(
        self,
        filename: str,
        graph: "WikiGraph",
        max_chars: int = 2000,
    ) -> str:
        """Return a formatted context string (≤ max_chars) for use in analysis prompts.

        Includes: cluster membership, direct dependencies, hot-zone neighbours.
        Returns "" when no relevant graph data exists.
        """
        clusters = graph.get_clusters()
        cluster = next((c for c in clusters if filename in c.members), None)

        edges = graph.get_edges(filename)
        outgoing = [e for e in edges if e.source == filename]
        incoming = [e for e in edges if e.target == filename]

        sections: list[str] = []

        if cluster:
            others = [m for m in cluster.members if m != filename][:8]
            s = f"### Cluster: {cluster.name}"
            if cluster.description:
                s += f"\n{cluster.description}"
            if others:
                s += f"\nGleicher Cluster: {', '.join(others)}"
            sections.append(s)

        if outgoing or incoming:
            parts = []
            if outgoing:
                parts.append(f"→ importiert: {', '.join(e.target for e in outgoing[:5])}")
            if incoming:
                parts.append(f"← importiert von: {', '.join(e.source for e in incoming[:5])}")
            sections.append("### Abhängigkeiten\n" + "\n".join(parts))

        neighbor_ids = {e.target for e in outgoing} | {e.source for e in incoming}
        hot = [
            nid for nid in list(neighbor_ids)[:10]
            if (n := graph.get_node(nid)) and n.turbulence_tier == "hot"
        ]
        if hot:
            sections.append(f"### Hot-Zone Nachbarn\n{', '.join(hot)}")

        if not sections:
            return ""
        return ("\n\n".join(sections))[:max_chars]
