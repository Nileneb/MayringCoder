from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

from src.wiki_v2.graph import WikiGraph
from src.wiki_v2.models import Cluster
from src.wiki_v2._path_utils import safe_workspace_id, confined_path


class ClusterEngine:
    """3-Layer Clustering: Struktur (Louvain) → Semantik (Embedding) → LLM-Benennung."""

    def cluster(
        self,
        graph: WikiGraph,
        strategy: str = "louvain",
        ollama_url: str = "",
        model: str = "qwen2.5-coder:14b",
        chroma: Any = None,
        embedding_threshold: float = 0.65,
    ) -> list[Cluster]:
        """Run clustering and persist results in graph.

        strategy: "louvain" — Layer 1 (Louvain) + LLM-Namen
                  "full"    — Layer 1 (Louvain) + Layer 2 (Embedding) + Layer 3 (LLM)
        """
        # Layer 1: Strukturell
        structural = self._louvain_communities(graph)
        if not structural:
            return []

        # Layer 2: Semantisch (nur bei strategy="full" und chroma vorhanden)
        if strategy == "full" and chroma is not None:
            semantic = self._embedding_clusters(graph, chroma, embedding_threshold)
            communities = self._merge_communities(structural, semantic, graph)
        else:
            communities = structural

        # Layer 3: LLM-Benennung
        if ollama_url and model:
            clusters = self._llm_name_clusters(communities, graph, ollama_url, model)
        else:
            clusters = self._default_name_clusters(communities, graph)

        for c in clusters:
            graph.upsert_cluster(c)

        self._write_clusters_json(clusters, graph)
        return clusters

    def _write_clusters_json(self, clusters: list[Cluster], graph: WikiGraph) -> None:
        """Schreibt clusters.json pro Workspace (Akzeptanzkriterium #73)."""
        try:
            from src.config import WIKI_DIR
            data = [
                {
                    "cluster_id": c.cluster_id,
                    "name": c.name,
                    "description": c.description,
                    "rationale": c.rationale,
                    "strategy_used": c.strategy_used,
                    "members": c.members,
                    "member_count": len(c.members),
                }
                for c in clusters
            ]
            out_path = confined_path(WIKI_DIR, safe_workspace_id(graph.workspace_id), "clusters.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            pass

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

    def _embedding_clusters(
        self,
        graph: WikiGraph,
        chroma: Any,
        threshold: float = 0.65,
    ) -> list[set[str]]:
        """Layer 2: Cosine-Similarity auf gemittelte Chunk-Embeddings pro Node.
        Nutzt bestehende ChromaDB-Embeddings — kein Re-Embed.
        """
        if chroma is None:
            return []
        nodes = graph.all_nodes()
        if len(nodes) < 2:
            return []

        workspace_id = graph.workspace_id
        node_ids = [n.id for n in nodes]

        try:
            result = chroma.get(
                where={"workspace_id": {"$eq": workspace_id}},
                include=["embeddings", "metadatas"],
            )
            raw_embeddings = result.get("embeddings") or []
            metadatas = result.get("metadatas") or []
        except Exception:
            return []

        # Akkumuliere Embeddings pro Node-ID (average über alle Chunks)
        node_vecs: dict[str, list[list[float]]] = {}
        for emb, meta in zip(raw_embeddings, metadatas):
            if not emb or not meta:
                continue
            source_id = meta.get("source_id", "")
            path = source_id.split(":")[-1] if ":" in source_id else source_id
            for nid in node_ids:
                if path == nid or path.endswith("/" + nid) or nid.endswith("/" + path):
                    node_vecs.setdefault(nid, []).append(list(emb))
                    break

        def _avg(vecs: list[list[float]]) -> list[float] | None:
            if not vecs:
                return None
            dim = len(vecs[0])
            return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]

        def _cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            return dot / (na * nb) if na > 0 and nb > 0 else 0.0

        avg_embs: dict[str, list[float]] = {
            nid: emb for nid, vecs in node_vecs.items()
            if (emb := _avg(vecs)) is not None
        }
        if len(avg_embs) < 2:
            return []

        # Union-Find: Nodes mit Cosine-Similarity >= threshold zusammenführen
        parent: dict[str, str] = {nid: nid for nid in avg_embs}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb

        nids = list(avg_embs.keys())
        for i, a in enumerate(nids):
            for b in nids[i + 1:]:
                if _cosine(avg_embs[a], avg_embs[b]) >= threshold:
                    union(a, b)

        groups: dict[str, set[str]] = {}
        for nid in nids:
            root = find(nid)
            groups.setdefault(root, set()).add(nid)

        return list(groups.values())

    def _merge_communities(
        self,
        structural: list[set[str]],
        semantic: list[set[str]],
        graph: WikiGraph,
    ) -> list[set[str]]:
        """Merge Layer-1 (Louvain) und Layer-2 (Embedding) Communities.

        Strategie: Wenn eine semantische Community Mitglieder aus mehreren
        strukturellen Clustern hat, werden diese strukturellen Cluster zusammengeführt.
        Isolierte Nodes (nur in semantischer, nicht in struktureller Community)
        werden dem nächsten strukturellen Cluster zugeteilt.
        """
        all_nodes = {n.id for n in graph.all_nodes()}

        # Node → struktureller Community-Index
        node_struct: dict[str, int] = {}
        for i, comm in enumerate(structural):
            for n in comm:
                node_struct[n] = i

        # Union-Find auf Community-Indizes
        parent_idx = list(range(len(structural)))

        def find_idx(x: int) -> int:
            while parent_idx[x] != x:
                parent_idx[x] = parent_idx[parent_idx[x]]
                x = parent_idx[x]
            return x

        def union_idx(a: int, b: int) -> None:
            pa, pb = find_idx(a), find_idx(b)
            if pa != pb:
                parent_idx[pa] = pb

        for sem_comm in semantic:
            struct_idxs = list({node_struct[n] for n in sem_comm if n in node_struct})
            for j in range(1, len(struct_idxs)):
                union_idx(struct_idxs[0], struct_idxs[j])

        # Zusammenführen
        merged: dict[int, set[str]] = {}
        for i, comm in enumerate(structural):
            root = find_idx(i)
            merged.setdefault(root, set()).update(comm)

        # Nodes die in keiner strukturellen Community sind (sollte nicht passieren)
        covered = {n for comm in merged.values() for n in comm}
        for n in all_nodes - covered:
            next_key = max(merged.keys(), default=-1) + 1
            merged[next_key] = {n}

        return list(merged.values())

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
