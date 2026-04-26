from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Any

from src.memory.db_adapter import DBAdapter

from src.wiki_v2.models import WikiNode, WikiEdge, Cluster


def init_wiki_db(db_path: Path) -> DBAdapter:
    """Create/open wiki_v2.db and ensure schema exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    adapter = DBAdapter.create(db_path, check_same_thread=False)
    _init_schema(adapter)
    return adapter


def _init_schema(conn: DBAdapter) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS wiki_nodes (
        id TEXT NOT NULL,
        repo_slug TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        type TEXT DEFAULT 'file',
        cluster_id TEXT DEFAULT '',
        labels_json TEXT DEFAULT '[]',
        summary TEXT DEFAULT '',
        turbulence_tier TEXT DEFAULT '',
        loc INTEGER DEFAULT 0,
        last_analyzed_by TEXT DEFAULT '',
        updated_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (id, workspace_id)
    );

    CREATE TABLE IF NOT EXISTS wiki_edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        target TEXT NOT NULL,
        repo_slug TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        type TEXT NOT NULL,
        weight REAL DEFAULT 1.0,
        context TEXT DEFAULT '',
        validated INTEGER DEFAULT 0,
        discovered_at TEXT DEFAULT (datetime('now')),
        UNIQUE(source, target, type, workspace_id)
    );

    CREATE TABLE IF NOT EXISTS wiki_clusters (
        cluster_id TEXT NOT NULL,
        repo_slug TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT DEFAULT '',
        rationale TEXT DEFAULT '',
        strategy_used TEXT DEFAULT 'louvain',
        member_count INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (cluster_id, workspace_id)
    );

    CREATE TABLE IF NOT EXISTS wiki_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_slug TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        snapshot_json TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS wiki_contributions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        action TEXT NOT NULL,
        target_node TEXT NOT NULL,
        timestamp TEXT DEFAULT (datetime('now'))
    );
    """)
    conn.commit()


# --- Node CRUD ---

def log_contribution(
    conn: DBAdapter,
    workspace_id: str,
    user_id: str,
    action: str,
    target_node: str,
) -> None:
    conn.execute(
        "INSERT INTO wiki_contributions (workspace_id, user_id, action, target_node) VALUES (?,?,?,?)",
        (workspace_id, user_id, action, target_node),
    )
    conn.commit()


def upsert_node(conn: DBAdapter, node: WikiNode, user_id: str = "system") -> None:
    conn.execute(
        """INSERT INTO wiki_nodes (id, repo_slug, workspace_id, type, cluster_id,
           labels_json, summary, turbulence_tier, loc, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?, datetime('now'))
           ON CONFLICT(id, workspace_id) DO UPDATE SET
           cluster_id=excluded.cluster_id, labels_json=excluded.labels_json,
           summary=excluded.summary, turbulence_tier=excluded.turbulence_tier,
           loc=excluded.loc, updated_at=excluded.updated_at""",
        (node.id, node.repo_slug, node.workspace_id, node.type, node.cluster_id,
         json.dumps(node.labels), node.summary, node.turbulence_tier, node.loc),
    )
    conn.commit()
    log_contribution(conn, node.workspace_id, user_id, "upsert_node", node.id)


def get_node(conn: DBAdapter, node_id: str, workspace_id: str) -> WikiNode | None:
    row = conn.execute(
        "SELECT * FROM wiki_nodes WHERE id=? AND workspace_id=?", (node_id, workspace_id)
    ).fetchone()
    if not row:
        return None
    return WikiNode(
        id=row["id"], repo_slug=row["repo_slug"], workspace_id=row["workspace_id"],
        type=row["type"], cluster_id=row["cluster_id"] or "",
        labels=json.loads(row["labels_json"] or "[]"),
        summary=row["summary"] or "", turbulence_tier=row["turbulence_tier"] or "",
        loc=row["loc"] or 0,
    )


def get_all_nodes(conn: DBAdapter, workspace_id: str) -> list[WikiNode]:
    rows = conn.execute(
        "SELECT * FROM wiki_nodes WHERE workspace_id=?", (workspace_id,)
    ).fetchall()
    return [WikiNode(
        id=r["id"], repo_slug=r["repo_slug"], workspace_id=r["workspace_id"],
        type=r["type"], cluster_id=r["cluster_id"] or "",
        labels=json.loads(r["labels_json"] or "[]"),
        summary=r["summary"] or "", turbulence_tier=r["turbulence_tier"] or "",
        loc=r["loc"] or 0,
    ) for r in rows]


# --- Edge CRUD ---

def add_edge(conn: DBAdapter, edge: WikiEdge) -> None:
    try:
        conn.execute(
            """INSERT INTO wiki_edges (source, target, repo_slug, workspace_id, type, weight, context, validated)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(source, target, type, workspace_id) DO UPDATE SET
               weight=excluded.weight, context=excluded.context""",
            (edge.source, edge.target, edge.repo_slug, edge.workspace_id,
             edge.type, edge.weight, edge.context, int(edge.validated)),
        )
        conn.commit()
    except Exception:
        pass


def get_edges(conn: DBAdapter, workspace_id: str,
              node_id: str | None = None, edge_type: str | None = None) -> list[WikiEdge]:
    q = "SELECT * FROM wiki_edges WHERE workspace_id=?"
    params: list[Any] = [workspace_id]
    if node_id:
        q += " AND (source=? OR target=?)"
        params += [node_id, node_id]
    if edge_type:
        q += " AND type=?"
        params.append(edge_type)
    rows = conn.execute(q, params).fetchall()
    return [WikiEdge(
        source=r["source"], target=r["target"], repo_slug=r["repo_slug"],
        workspace_id=r["workspace_id"], type=r["type"], weight=r["weight"],
        context=r["context"] or "", validated=bool(r["validated"]),
    ) for r in rows]


def get_edges_by_type(conn: DBAdapter, workspace_id: str, edge_type: str) -> list[WikiEdge]:
    return get_edges(conn, workspace_id, edge_type=edge_type)


# --- Cluster CRUD ---

def upsert_cluster(conn: DBAdapter, cluster: Cluster, user_id: str = "system") -> None:
    conn.execute(
        """INSERT INTO wiki_clusters (cluster_id, repo_slug, workspace_id, name, description, rationale, strategy_used, member_count)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(cluster_id, workspace_id) DO UPDATE SET
           name=excluded.name, description=excluded.description,
           rationale=excluded.rationale, strategy_used=excluded.strategy_used,
           member_count=excluded.member_count""",
        (cluster.cluster_id, cluster.repo_slug, cluster.workspace_id, cluster.name,
         cluster.description, cluster.rationale, cluster.strategy_used, len(cluster.members)),
    )
    conn.execute(
        "UPDATE wiki_nodes SET cluster_id=? WHERE workspace_id=? AND id IN ({})".format(
            ",".join("?" * len(cluster.members))
        ),
        [cluster.cluster_id, cluster.workspace_id] + cluster.members,
    )
    conn.commit()
    log_contribution(conn, cluster.workspace_id, user_id, "upsert_cluster", cluster.cluster_id)


def get_clusters(conn: DBAdapter, workspace_id: str) -> list[Cluster]:
    rows = conn.execute(
        "SELECT * FROM wiki_clusters WHERE workspace_id=?", (workspace_id,)
    ).fetchall()
    result = []
    for r in rows:
        members_rows = conn.execute(
            "SELECT id FROM wiki_nodes WHERE cluster_id=? AND workspace_id=?",
            (r["cluster_id"], workspace_id),
        ).fetchall()
        result.append(Cluster(
            cluster_id=r["cluster_id"], repo_slug=r["repo_slug"],
            workspace_id=r["workspace_id"], name=r["name"],
            description=r["description"] or "", rationale=r["rationale"] or "",
            strategy_used=r["strategy_used"] or "louvain",
            members=[m["id"] for m in members_rows],
        ))
    return result


def get_feedback_matrix(memory_conn: DBAdapter, limit: int = 50) -> list[dict]:
    """Return feedback aggregated per chunk, joined with source/category info.

    Args:
        memory_conn: Connection to the main memory DB (contains chunk_feedback + chunks).
        limit: Max chunks to return (ordered by total feedback desc).

    Returns:
        List of {chunk_id, source_id, category_labels, positive, negative, neutral, net_score}
    """
    rows = memory_conn.execute(
        """
        SELECT
            cf.chunk_id,
            COALESCE(c.source_id, '') AS source_id,
            COALESCE(c.category_labels, '') AS category_labels,
            SUM(CASE WHEN cf.signal = 'positive' THEN 1 ELSE 0 END) AS positive,
            SUM(CASE WHEN cf.signal = 'negative' THEN 1 ELSE 0 END) AS negative,
            SUM(CASE WHEN cf.signal NOT IN ('positive','negative') THEN 1 ELSE 0 END) AS neutral,
            COUNT(*) AS total
        FROM chunk_feedback cf
        LEFT JOIN chunks c ON c.chunk_id = cf.chunk_id
        GROUP BY cf.chunk_id
        ORDER BY total DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    result = []
    for r in rows:
        pos = r[3] if isinstance(r, tuple) else r["positive"]
        neg = r[4] if isinstance(r, tuple) else r["negative"]
        total = r[6] if isinstance(r, tuple) else r["total"]
        net = round((pos - neg) / max(total, 1), 2)
        result.append({
            "chunk_id": r[0] if isinstance(r, tuple) else r["chunk_id"],
            "source_id": r[1] if isinstance(r, tuple) else r["source_id"],
            "category_labels": r[2] if isinstance(r, tuple) else r["category_labels"],
            "positive": pos,
            "negative": neg,
            "neutral": r[5] if isinstance(r, tuple) else r["neutral"],
            "net_score": net,
        })
    return result


def get_task_feedback_matrix(
    memory_conn: DBAdapter,
    limit: int = 50,
    query_filter: str | None = None,
) -> list[dict]:
    """Return feedback grouped by query/task context.

    Each entry represents one query session with the list of chunks
    that were rated (positive/negative/neutral) during that query.

    Returns:
        List of {query, chunks: [{chunk_id, source_id, signal, created_at}]}
    """
    where = "WHERE json_extract(cf.metadata, '$.query_context') IS NOT NULL"
    params: list = []
    if query_filter:
        where += " AND json_extract(cf.metadata, '$.query_context') LIKE ?"
        params.append(f"%{query_filter}%")

    rows = memory_conn.execute(
        f"""
        SELECT
            json_extract(cf.metadata, '$.query_context') AS query,
            cf.chunk_id,
            COALESCE(c.source_id, '') AS source_id,
            cf.signal,
            cf.created_at
        FROM chunk_feedback cf
        LEFT JOIN chunks c ON c.chunk_id = cf.chunk_id
        {where}
        ORDER BY cf.created_at DESC
        LIMIT ?
        """,
        params + [limit],
    ).fetchall()

    tasks: dict[str, dict] = {}
    for r in rows:
        query = r[0] if isinstance(r, tuple) else r["query"]
        if query not in tasks:
            tasks[query] = {"query": query, "chunks": []}
        tasks[query]["chunks"].append({
            "chunk_id": r[1] if isinstance(r, tuple) else r["chunk_id"],
            "source_id": r[2] if isinstance(r, tuple) else r["source_id"],
            "signal": r[3] if isinstance(r, tuple) else r["signal"],
            "created_at": r[4] if isinstance(r, tuple) else r["created_at"],
        })
    return list(tasks.values())
