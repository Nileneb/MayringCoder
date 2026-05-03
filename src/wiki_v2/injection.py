from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.db_adapter import DBAdapter
    from src.wiki_v2.graph import WikiGraph


class WikiContextInjector:
    """Builds a wiki context block for analysis prompts from WikiGraph data.

    With an optional `memory_conn` (DBAdapter) the injector also adds an
    IGIO section listing the chunk-level axis distribution and the highest-
    confidence sample per axis. The structural sections are unchanged so
    existing callers/tests continue to work.
    """

    def build_context(
        self,
        filename: str,
        graph: "WikiGraph",
        max_chars: int = 2000,
        *,
        memory_conn: "DBAdapter | None" = None,
    ) -> str:
        """Return a formatted context string (≤ max_chars).

        Sections (in order):
          1. Cluster membership
          2. Direct dependencies (imports)
          3. Hot-zone neighbours
          4. IGIO position — only when memory_conn is provided and the file
             has classified chunks
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

        if memory_conn is not None:
            igio_section = self._build_igio_section(
                filename, memory_conn,
                workspace_id=getattr(graph, "workspace_id", None),
            )
            if igio_section:
                sections.append(igio_section)

        if not sections:
            return ""
        return ("\n\n".join(sections))[:max_chars]

    @staticmethod
    def _build_igio_section(
        filename: str,
        memory_conn: "DBAdapter",
        *,
        workspace_id: str | None = None,
    ) -> str:
        """Render an IGIO-position section for `filename` from chunks.

        Strategy: source_id heuristically contains the filename (e.g.
        `repo:owner/name:src/foo.py`). We aggregate axis counts per file and
        pick the top-confidence chunk per axis as a one-line sample.
        """
        if not filename:
            return ""
        where = ["igio_axis != ''", "is_active = 1", "source_id LIKE ?"]
        params: list = [f"%{filename}%"]
        if workspace_id:
            where.append("workspace_id = ?")
            params.append(workspace_id)
        sql = (
            "SELECT igio_axis, COUNT(*) AS n FROM chunks "
            f"WHERE {' AND '.join(where)} GROUP BY igio_axis"
        )
        counts = memory_conn.execute(sql, tuple(params)).fetchall()
        if not counts:
            return ""

        lines = ["### IGIO-Position"]
        for r in counts:
            axis = r["igio_axis"]
            n = r["n"]
            sample_sql = (
                "SELECT substr(text, 1, 100) AS preview, igio_confidence "
                f"FROM chunks WHERE {' AND '.join(where)} AND igio_axis = ? "
                "ORDER BY igio_confidence DESC LIMIT 1"
            )
            sample_params = (*params, axis)
            sample = memory_conn.execute(sample_sql, sample_params).fetchone()
            if sample:
                preview = (sample["preview"] or "").replace("\n", " ").strip()
                lines.append(
                    f"- {axis} ({n}): {preview[:80]}…"
                    if len(preview) > 80
                    else f"- {axis} ({n}): {preview}"
                )
            else:
                lines.append(f"- {axis} ({n})")
        return "\n".join(lines)
