from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

from src.wiki_v2.graph import WikiGraph
from src.wiki_v2.models import Cluster
from src.wiki_v2._path_utils import confined_path


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
            out_path = confined_path(WIKI_DIR, graph.workspace_id, "clusters.json")
            # Defense-in-depth: confined_path already enforces this, but
            # writing the property explicitly here makes the safety
            # invariant visible to CodeQL's `py/path-injection` data-flow
            # tracker (which can't follow our private sanitisation
            # helper). If the resolved path escapes WIKI_DIR for any
            # reason (symlink, race), bail silently.
            wiki_root = Path(WIKI_DIR).resolve()
            resolved = out_path.resolve()
            if not resolved.is_relative_to(wiki_root):
                return
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            pass

    # Path-prefix coalesce: orphan singletons get attached to the cluster of
    # their path-sibling (same first two segments — "src/api", "src/wiki_v2",
    # "tests", "tools" etc.). Without this Louvain on a sparse import graph
    # produces 695 singletons over 1128 files because half the codebase has
    # no incoming/outgoing imports (configs, docs, tests, tooling).
    _MIN_CLUSTER_SIZE = 3

    @staticmethod
    def _path_prefix(node_id: str) -> str:
        """Best 'this file lives here' bucket name for grouping orphans.

        Layout examples:
          • src/api/foo.py        → "src/api"   (3+ parts, take first two)
          • tests/t1.py           → "tests"     (2 parts, the second IS the file
                                                — using "tests/t1.py" would
                                                give every test file its own
                                                bucket which defeats the point)
          • README.md             → "README.md" (single part, only option)
        """
        parts = Path(node_id).parts
        if len(parts) >= 3:
            return "/".join(parts[:2])
        if len(parts) == 2:
            return parts[0]
        return parts[0] if parts else "_root"

    def _path_prefix_clusters(self, node_ids: set[str] | list[str]) -> list[set[str]]:
        groups: dict[str, set[str]] = {}
        for nid in node_ids:
            groups.setdefault(self._path_prefix(nid), set()).add(nid)
        return list(groups.values())

    def _adaptive_resolution(self, n_nodes: int) -> float:
        """Smaller graphs need lower resolution to avoid Louvain singletons.

        Tuned by repo size — large monorepos can absorb resolution=1.0 fine,
        small libraries get crushed into one big cluster at that setting and
        fragmented at 1.2 (the old default). 0.4..1.0 covers what we see in
        practice: tiny libs → coarse buckets, mature repos → real communities.
        """
        if n_nodes < 50:
            return 0.4
        if n_nodes < 200:
            return 0.7
        if n_nodes < 500:
            return 0.9
        return 1.0

    def _louvain_communities(self, graph: WikiGraph) -> list[set[str]]:
        """3-pass clustering: build edge graph → Louvain → singleton absorption.

        Pass 1 — graph build: weighted multigraph from import + call edges.
        Pass 2 — Louvain at adaptive resolution (or greedy modularity fallback).
        Pass 3 — clusters with < _MIN_CLUSTER_SIZE members get folded into a
                 path-sibling big cluster, or — failing that — into the big
                 cluster they share the most edges with. This kills the
                 long-tail of orphan files (configs, READMEs, tests with no
                 imports) that would otherwise show up as 1-member "clusters".
        """
        nodes = graph.all_nodes()
        if not nodes:
            return []
        node_ids = [n.id for n in nodes]

        all_edges = graph.edges_by_type("import") + graph.edges_by_type("call")
        G = nx.Graph()
        G.add_nodes_from(node_ids)
        for e in all_edges:
            if not (G.has_node(e.source) and G.has_node(e.target)):
                continue
            if G.has_edge(e.source, e.target):
                G[e.source][e.target]["weight"] = (
                    G[e.source][e.target].get("weight", 1.0) + e.weight
                )
            else:
                G.add_edge(e.source, e.target, weight=e.weight)

        # Edge-poor: don't bother running Louvain on a graph that's almost
        # disconnected. Path-prefix grouping is the more honest signal.
        if G.number_of_edges() < max(3, len(node_ids) // 30):
            return self._path_prefix_clusters(node_ids)

        resolution = self._adaptive_resolution(len(node_ids))
        try:
            from networkx.algorithms.community import louvain_communities
            raw = louvain_communities(G, resolution=resolution, seed=42)
            comms = [set(c) for c in raw]
        except Exception:
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                comms = [set(c) for c in greedy_modularity_communities(G)]
            except Exception:
                return self._path_prefix_clusters(node_ids)

        return self._absorb_small_clusters(comms, G)

    def _absorb_small_clusters(
        self, comms: list[set[str]], G: "nx.Graph"
    ) -> list[set[str]]:
        """Fold clusters with < _MIN_CLUSTER_SIZE into a path-sibling big cluster.

        Strategy per small cluster:
          1. Try path-prefix match: if any small-member shares a path prefix
             with a big cluster, merge into that big cluster.
          2. Fall back to edge-density: merge into the big cluster that has
             the most graph edges into the small one.
          3. If everything is tiny (no big clusters at all), regroup the
             entire set by path prefix — that's still a real signal.
        """
        big = [c for c in comms if len(c) >= self._MIN_CLUSTER_SIZE]
        small = [c for c in comms if len(c) < self._MIN_CLUSTER_SIZE]
        if not small:
            return big
        if not big:
            all_ids = {nid for c in small for nid in c}
            return self._path_prefix_clusters(all_ids)

        prefix_to_big: dict[str, int] = {}
        for i, c in enumerate(big):
            for nid in c:
                prefix_to_big.setdefault(self._path_prefix(nid), i)

        for s in small:
            target = None
            for nid in s:
                hit = prefix_to_big.get(self._path_prefix(nid))
                if hit is not None:
                    target = hit
                    break
            if target is None:
                best_score = 0
                best_idx = 0
                for i, c in enumerate(big):
                    score = sum(
                        1 for nid in s for m in c if G.has_edge(nid, m)
                    )
                    if score > best_score:
                        best_score = score
                        best_idx = i
                target = best_idx
            big[target].update(s)
        return big

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
            from src.ollama_client import generate as _ollama_gen
            response_text = _ollama_gen(
                ollama_url, model, prompt,
                stream=False,
                options={"temperature": 0.1},
                num_predict=2000,
                timeout=60.0,
            )
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

    sizes = sorted(len(c.members) for c in clusters)
    n = len(sizes)

    def _pct(p: float) -> int:
        if not sizes:
            return 0
        idx = max(0, min(n - 1, int(p * n)))
        return sizes[idx]

    return {
        "num_clusters": n,
        "total_edges": total_edges,
        "internal_edges": internal,
        "external_edges": external,
        "internal_ratio": internal / max(1, internal + external),
        "singleton_clusters": sum(1 for s in sizes if s <= 1),
        "small_clusters": sum(1 for s in sizes if s < 3),
        "median_cluster_size": _pct(0.5),
        "p10_cluster_size": _pct(0.1),
        "p90_cluster_size": _pct(0.9),
        "largest_cluster": sizes[-1] if sizes else 0,
    }
