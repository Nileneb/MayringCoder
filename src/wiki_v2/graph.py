from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import os as _os_g
from src.config import CACHE_DIR, WIKI_DIR
from src.wiki_v2.models import WikiNode, WikiEdge, Cluster
from src.memory.db_adapter import DBAdapter
from src.wiki_v2 import store as _store
from src.wiki_v2._path_utils import confined_path


class WikiGraph:
    def __init__(self, workspace_id: str, repo_slug: str, db_path: Path | None = None):
        self.workspace_id = _os_g.path.basename(workspace_id.replace('/', '_').replace('\\', '_'))
        self.repo_slug = repo_slug
        self._db_path = db_path or (CACHE_DIR / "wiki_v2.db")
        self._conn: DBAdapter = _store.init_wiki_db(self._db_path)

    def upsert_node(self, node: WikiNode) -> None:
        _store.upsert_node(self._conn, node)

    def get_node(self, node_id: str) -> WikiNode | None:
        return _store.get_node(self._conn, node_id, self.workspace_id)

    def all_nodes(self) -> list[WikiNode]:
        return _store.get_all_nodes(self._conn, self.workspace_id)

    def add_edge(self, edge: WikiEdge) -> None:
        _store.add_edge(self._conn, edge)

    def get_edges(self, node_id: str | None = None, edge_type: str | None = None) -> list[WikiEdge]:
        return _store.get_edges(self._conn, self.workspace_id, node_id, edge_type)

    def edges_by_type(self, edge_type: str) -> list[WikiEdge]:
        return _store.get_edges_by_type(self._conn, self.workspace_id, edge_type)

    def upsert_cluster(self, cluster: Cluster) -> None:
        _store.upsert_cluster(self._conn, cluster)

    def get_clusters(self) -> list[Cluster]:
        return _store.get_clusters(self._conn, self.workspace_id)

    def node_count(self) -> int:
        return len(self.all_nodes())

    def edge_count(self) -> int:
        return len(self.get_edges())

    def to_json(self) -> dict:
        nodes = self.all_nodes()
        edges = self.get_edges()
        clusters = self.get_clusters()

        in_deg: dict[str, int] = {}
        out_deg: dict[str, int] = {}
        for e in edges:
            out_deg[e.source] = out_deg.get(e.source, 0) + 1
            in_deg[e.target] = in_deg.get(e.target, 0) + 1

        data = {
            "workspace_id": self.workspace_id,
            "repo_slug": self.repo_slug,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type,
                    "cluster_id": n.cluster_id,
                    "labels": n.labels,
                    "summary": n.summary,
                    "turbulence_tier": n.turbulence_tier,
                    "loc": n.loc,
                    "connections_in": in_deg.get(n.id, 0),
                    "connections_out": out_deg.get(n.id, 0),
                }
                for n in nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.type,
                    "weight": e.weight,
                    "context": e.context,
                }
                for e in edges
            ],
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "name": c.name,
                    "description": c.description,
                    "members": c.members,
                    "strategy_used": c.strategy_used,
                }
                for c in clusters
            ],
        }

        out_path = confined_path(WIKI_DIR, self.workspace_id, "graph.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return data

    def snapshot(self) -> None:
        data = self.to_json()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap_path = confined_path(WIKI_DIR, self.workspace_id, "history", f"{ts}.json")
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        self._conn.execute(
            "INSERT INTO wiki_snapshots (repo_slug, workspace_id, snapshot_json) VALUES (?,?,?)",
            (self.repo_slug, self.workspace_id, json.dumps(data)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
