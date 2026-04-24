from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from src.wiki_v2._path_utils import safe_filename_part, confined_path

if TYPE_CHECKING:
    from src.wiki_v2.graph import WikiGraph
    from src.wiki_v2.models import WikiEdge, Cluster


@dataclass
class SnapshotSummary:
    snapshot_id: int
    workspace_id: str
    trigger: str
    node_count: int
    edge_count: int
    cluster_count: int
    created_at: str


@dataclass
class WikiDiff:
    nodes_added: list[str] = field(default_factory=list)
    nodes_removed: list[str] = field(default_factory=list)
    edges_added: list[dict] = field(default_factory=list)
    edges_removed: list[dict] = field(default_factory=list)
    clusters_added: list[str] = field(default_factory=list)
    clusters_removed: list[str] = field(default_factory=list)
    contributors: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"+{len(self.nodes_added)}/-{len(self.nodes_removed)} Nodes  "
            f"+{len(self.edges_added)}/-{len(self.edges_removed)} Edges  "
            f"+{len(self.clusters_added)}/-{len(self.clusters_removed)} Cluster  "
            f"{len(self.contributors)} User aktiv"
        )


class WikiHistory:
    """Manages graph snapshots, diffs, and team activity."""

    def create_snapshot(self, graph: "WikiGraph", trigger: str = "manual") -> int:
        """Persist a snapshot. Returns the new snapshot_id."""
        nodes = graph.all_nodes()
        edges = graph.get_edges()
        clusters = graph.get_clusters()
        data = {
            "trigger": trigger,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "cluster_count": len(clusters),
            "nodes": [n.id for n in nodes],
            "edges": [{"source": e.source, "target": e.target, "type": e.type}
                      for e in edges],
            "clusters": [c.cluster_id for c in clusters],
        }
        cur = graph._conn.execute(
            "INSERT INTO wiki_snapshots (repo_slug, workspace_id, snapshot_json) VALUES (?,?,?)",
            (graph.repo_slug, graph.workspace_id, json.dumps(data)),
        )
        graph._conn.commit()

        # Also write to history/ dir
        try:
            from src.config import WIKI_DIR
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            snap_path = confined_path(
                WIKI_DIR,
                graph.workspace_id,
                "history",
                f"{ts}_{safe_filename_part(trigger)}.json",
            )
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            snap_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            pass

        return cur.lastrowid

    def timeline(
        self,
        conn: sqlite3.Connection,
        workspace_id: str,
        limit: int = 20,
    ) -> list[SnapshotSummary]:
        """Return last `limit` snapshots for a workspace, newest first."""
        rows = conn.execute(
            "SELECT id, workspace_id, snapshot_json, created_at FROM wiki_snapshots "
            "WHERE workspace_id=? ORDER BY id DESC LIMIT ?",
            (workspace_id, limit),
        ).fetchall()
        result = []
        for r in rows:
            try:
                data = json.loads(r[2] if isinstance(r, (list, tuple)) else r["snapshot_json"])
            except Exception:
                data = {}
            snap_id = r[0] if isinstance(r, (list, tuple)) else r["id"]
            created = r[3] if isinstance(r, (list, tuple)) else r["created_at"]
            result.append(SnapshotSummary(
                snapshot_id=snap_id,
                workspace_id=workspace_id,
                trigger=data.get("trigger", "unknown"),
                node_count=data.get("node_count", 0),
                edge_count=data.get("edge_count", 0),
                cluster_count=data.get("cluster_count", 0),
                created_at=str(created),
            ))
        return result

    def diff(
        self,
        conn: sqlite3.Connection,
        workspace_id: str,
        from_date: str,
        to_date: str,
    ) -> WikiDiff:
        """Diff snapshots between two ISO-8601 dates. Compares nearest snapshots."""
        rows = conn.execute(
            "SELECT id, snapshot_json, created_at FROM wiki_snapshots "
            "WHERE workspace_id=? AND created_at BETWEEN ? AND ? ORDER BY id",
            (workspace_id, from_date, to_date + "Z" if not to_date.endswith("Z") else to_date),
        ).fetchall()

        if len(rows) < 2:
            # Try: first snapshot before from_date, last snapshot before to_date
            r_from = conn.execute(
                "SELECT snapshot_json FROM wiki_snapshots WHERE workspace_id=? "
                "AND created_at <= ? ORDER BY id DESC LIMIT 1",
                (workspace_id, from_date),
            ).fetchone()
            r_to = conn.execute(
                "SELECT snapshot_json FROM wiki_snapshots WHERE workspace_id=? "
                "AND created_at <= ? ORDER BY id DESC LIMIT 1",
                (workspace_id, to_date),
            ).fetchone()
            if not r_from or not r_to:
                return WikiDiff()
            snap_a = json.loads(r_from[0])
            snap_b = json.loads(r_to[0])
        else:
            snap_a = json.loads(rows[0][1] if isinstance(rows[0], (list, tuple)) else rows[0]["snapshot_json"])
            snap_b = json.loads(rows[-1][1] if isinstance(rows[-1], (list, tuple)) else rows[-1]["snapshot_json"])

        nodes_a = set(snap_a.get("nodes", []))
        nodes_b = set(snap_b.get("nodes", []))
        edges_a = {(e["source"], e["target"], e["type"]) for e in snap_a.get("edges", [])}
        edges_b = {(e["source"], e["target"], e["type"]) for e in snap_b.get("edges", [])}
        clusters_a = set(snap_a.get("clusters", []))
        clusters_b = set(snap_b.get("clusters", []))

        contributors = team_activity(conn, workspace_id, from_date=from_date, to_date=to_date)

        return WikiDiff(
            nodes_added=sorted(nodes_b - nodes_a),
            nodes_removed=sorted(nodes_a - nodes_b),
            edges_added=[{"source": s, "target": t, "type": et} for s, t, et in edges_b - edges_a],
            edges_removed=[{"source": s, "target": t, "type": et} for s, t, et in edges_a - edges_b],
            clusters_added=sorted(clusters_b - clusters_a),
            clusters_removed=sorted(clusters_a - clusters_b),
            contributors=contributors,
        )

    def cleanup(
        self,
        conn: sqlite3.Connection,
        workspace_id: str,
        keep: int = 20,
    ) -> int:
        """Delete oldest snapshots, keeping only the `keep` most recent. Returns deleted count."""
        rows = conn.execute(
            "SELECT id FROM wiki_snapshots WHERE workspace_id=? ORDER BY id DESC",
            (workspace_id,),
        ).fetchall()
        ids = [r[0] if isinstance(r, (list, tuple)) else r["id"] for r in rows]
        to_delete = ids[keep:]
        if not to_delete:
            return 0
        conn.execute(
            f"DELETE FROM wiki_snapshots WHERE id IN ({','.join('?' * len(to_delete))})",
            to_delete,
        )
        conn.commit()
        return len(to_delete)


def team_activity(
    conn: sqlite3.Connection,
    workspace_id: str,
    days: int = 30,
    from_date: str = "",
    to_date: str = "",
) -> dict[str, int]:
    """Return {user_id: action_count} for the workspace within the time window."""
    if from_date and to_date:
        rows = conn.execute(
            "SELECT user_id, COUNT(*) as cnt FROM wiki_contributions "
            "WHERE workspace_id=? AND timestamp BETWEEN ? AND ? GROUP BY user_id",
            (workspace_id, from_date, to_date),
        ).fetchall()
    else:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = conn.execute(
            "SELECT user_id, COUNT(*) as cnt FROM wiki_contributions "
            "WHERE workspace_id=? AND timestamp >= ? GROUP BY user_id",
            (workspace_id, cutoff),
        ).fetchall()
    return {
        (r[0] if isinstance(r, (list, tuple)) else r["user_id"]):
        (r[1] if isinstance(r, (list, tuple)) else r["cnt"])
        for r in rows
    }


class ConflictResolver:
    """Last-write-wins with user-defined edges taking precedence over auto-detected ones."""

    def resolve_node(self, existing, incoming):
        """Auto-update fields (turbulence_tier, last_analyzed_by) always overwrite.
        Manual labels are merged (union).
        """
        existing.turbulence_tier = incoming.turbulence_tier or existing.turbulence_tier
        existing.last_analyzed_by = incoming.last_analyzed_by or existing.last_analyzed_by
        existing.summary = incoming.summary or existing.summary
        existing.labels = list(set((existing.labels or []) + (incoming.labels or [])))
        return existing

    def resolve_edge(self, existing, incoming) -> tuple:
        """Validated (user) edges take precedence over auto-detected ones.
        Returns (winner, needs_review: bool).
        """
        if existing.validated and not incoming.validated:
            return existing, False
        if not existing.validated and incoming.validated:
            return incoming, False
        if existing.validated and incoming.validated:
            return existing, True  # both manual → needs review
        return incoming, False  # both auto → latest wins
