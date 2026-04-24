from __future__ import annotations
import json
import re
from pathlib import Path

import networkx as nx

from src.wiki_v2.graph import WikiGraph
from src.wiki_v2.models import Cluster


class ClusterEngine:
    """Cluster-Strategie: Louvain für Struktur + Ollama für LLM-Benennung."""

    def cluster(
        self,
        graph: WikiGraph,
        strategy: str = "louvain",
        ollama_url: str = "",
        model: str = "qwen2.5-coder:14b",
    ) -> list[Cluster]:
        """Run clustering and persist results in graph.

        strategy: "louvain" — Louvain + LLM-Namen (wenn ollama_url gesetzt)
                  "full"    — reserviert für spätere Embedding-Layer
        """
        communities = self._louvain_communities(graph)

        if not communities:
            return []

        if ollama_url and model:
            clusters = self._llm_name_clusters(communities, graph, ollama_url, model)
        else:
            clusters = self._default_name_clusters(communities, graph)

        for c in clusters:
            graph.upsert_cluster(c)
        return clusters

    def _louvain_communities(self, graph: WikiGraph) -> list[set[str]]:
        """NetworkX Louvain auf Import+Call-Edges.

        Fallback-Kette:
        1. louvain_communities (nx >= 3.0)
        2. greedy_modularity_communities
        3. cluster_themes (Union-Find) aus wiki.py
        """
        nodes = graph.all_nodes()
        if not nodes:
            return []

        import_edges = graph.edges_by_type("import")
        call_edges = graph.edges_by_type("call")
        all_edges = import_edges + call_edges

        G = nx.Graph()
        for n in nodes:
            G.add_node(n.id)
        for e in all_edges:
            if G.has_node(e.source) and G.has_node(e.target):
                if G.has_edge(e.source, e.target):
                    G[e.source][e.target]["weight"] = G[e.source][e.target].get("weight", 1.0) + e.weight
                else:
                    G.add_edge(e.source, e.target, weight=e.weight)

        if G.number_of_edges() < 3:
            return [{n.id} for n in nodes]

        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, resolution=1.2, seed=42)
            return [set(c) for c in communities]
        except Exception:
            pass

        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(G)
            return [set(c) for c in communities]
        except Exception:
            pass

        from src.memory.wiki import cluster_themes, WikiEdge as OldEdge
        old_edges = [
            OldEdge(file_a=e.source, file_b=e.target, weight=e.weight, rule=e.type)
            for e in all_edges
        ]
        old_clusters = cluster_themes(old_edges, min_files=1)
        return [set(c.files) for c in old_clusters] or [{n.id} for n in nodes]

    def _llm_name_clusters(
        self,
        communities: list[set[str]],
        graph: WikiGraph,
        ollama_url: str,
        model: str,
    ) -> list[Cluster]:
        """LLM benennt Cluster. Fallback auf _default_name_clusters bei Fehler/Timeout."""
        node_summaries = {n.id: n.summary for n in graph.all_nodes() if n.summary}

        cluster_descriptions = []
        for i, members in enumerate(communities):
            summaries = [f"  - {m}: {node_summaries.get(m, '')}" for m in sorted(members)[:10]]
            cluster_descriptions.append(
                f"Cluster {i} ({len(members)} Dateien):\n" + "\n".join(summaries)
            )

        prompt = (
            "Du bist Software-Architekt. Benenne diese Code-Cluster mit sprechenden Namen.\n\n"
            + "\n\n".join(cluster_descriptions[:20])
            + '\n\nOutput: JSON-Array, ein Objekt pro Cluster in derselben Reihenfolge:\n'
            '[{"cluster_id": "kebab-case-slug", "name": "Kurzer Name", '
            '"description": "1-2 Sätze", "rationale": "Warum diese Gruppe?"}]\n'
            "Nur JSON, kein Prosa."
        )

        try:
            import urllib.request
            payload = json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 2000},
            }).encode()
            req = urllib.request.Request(
                f"{ollama_url.rstrip('/')}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read())
            response_text = raw.get("response", "")
            m = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not m:
                raise ValueError("No JSON array found")
            llm_data = json.loads(m.group(0))
        except Exception:
            return self._default_name_clusters(communities, graph)

        clusters = []
        for i, members in enumerate(communities):
            if i < len(llm_data):
                d = llm_data[i] if isinstance(llm_data[i], dict) else {}
                cid = re.sub(r'[^a-z0-9-]', '-', d.get("cluster_id", f"cluster-{i}").lower())
                clusters.append(Cluster(
                    cluster_id=cid,
                    repo_slug=graph.repo_slug,
                    workspace_id=graph.workspace_id,
                    name=d.get("name", f"Cluster {i}"),
                    description=d.get("description", ""),
                    rationale=d.get("rationale", ""),
                    strategy_used="louvain+llm",
                    members=sorted(members),
                ))
            else:
                clusters.extend(self._default_name_clusters([members], graph))
        return clusters

    def _default_name_clusters(
        self,
        communities: list[set[str]],
        graph: WikiGraph,
    ) -> list[Cluster]:
        """Fallback-Benennung: Cluster heißen nach dem häufigsten gemeinsamen Pfad-Prefix."""
        clusters = []
        for i, members in enumerate(communities):
            sorted_members = sorted(members)
            if sorted_members:
                parts = Path(sorted_members[0]).parts
                name = parts[1] if len(parts) > 1 else parts[0]
            else:
                name = f"cluster-{i}"
            cid = f"cluster-{i}"
            clusters.append(Cluster(
                cluster_id=cid,
                repo_slug=graph.repo_slug,
                workspace_id=graph.workspace_id,
                name=name,
                description="",
                rationale="",
                strategy_used="louvain",
                members=sorted_members,
            ))
        return clusters


def cluster_quality(clusters: list[Cluster], graph: WikiGraph) -> dict:
    """Cluster-Qualitätsmetriken."""
    total_edges = graph.edge_count()
    member_sets = {c.cluster_id: set(c.members) for c in clusters}

    internal = 0
    external = 0
    for e in graph.get_edges():
        src_cluster = next((cid for cid, members in member_sets.items() if e.source in members), None)
        tgt_cluster = next((cid for cid, members in member_sets.items() if e.target in members), None)
        if src_cluster and tgt_cluster:
            if src_cluster == tgt_cluster:
                internal += 1
            else:
                external += 1

    return {
        "num_clusters": len(clusters),
        "total_edges": total_edges,
        "internal_edges": internal,
        "external_edges": external,
        "singleton_clusters": sum(1 for c in clusters if len(c.members) <= 1),
        "largest_cluster": max((len(c.members) for c in clusters), default=0),
    }
