"""Predictive topic transitions — Markov chain over conversation-summary topic sequences."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TopicTransition:
    from_topic: str
    to_topic: str
    count: int
    probability: float


def _extract_topics_from_text(text: str, keyword_index: dict[str, list[str]]) -> list[str]:
    """Return ordered list of cluster-topics mentioned in text (dedup preserving order)."""
    if not text or not keyword_index:
        return []
    text_lower = text.lower()
    seen: list[str] = []
    words = re.findall(r'\b[a-z][a-z0-9_-]{2,}\b', text_lower)
    for w in words:
        clusters = keyword_index.get(w, [])
        for c in clusters:
            if c not in seen:
                seen.append(c)
    return seen


def _load_keyword_index(repo_slug: str) -> dict[str, list[str]]:
    """Load wiki_index.json for repo_slug. Returns {} if missing."""
    path = Path("cache") / f"{repo_slug}_wiki_index.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_transition_matrix(
    conn: Any,
    repo_slug: str = "",
    limit: int = 100,
) -> dict[str, dict[str, int]]:
    """Scan last `limit` conversation-summary chunks, extract topic sequences,
    build sparse Markov counts {from_topic: {to_topic: count}}.
    """
    rows = conn.execute(
        """SELECT c.text FROM chunks c
           JOIN sources s ON c.source_id = s.source_id
           WHERE s.source_type = 'conversation_summary'
             AND (s.repo = ? OR ? = '')
             AND c.is_active = 1
           ORDER BY s.captured_at DESC
           LIMIT ?""",
        (repo_slug, repo_slug, limit),
    ).fetchall()

    kw_index = _load_keyword_index(repo_slug)
    if not kw_index:
        return {}

    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        text = row[0]
        topics = _extract_topics_from_text(text, kw_index)
        for a, b in zip(topics, topics[1:]):
            if a == b:
                continue
            matrix.setdefault(a, {})
            matrix[a][b] = matrix[a].get(b, 0) + 1
    return matrix


def predict_next_topics(
    current_topic: str,
    matrix: dict[str, dict[str, int]],
    top_k: int = 3,
) -> list[TopicTransition]:
    """Return top_k TopicTransition entries for current_topic, sorted by probability desc."""
    transitions = matrix.get(current_topic, {})
    total = sum(transitions.values())
    if total == 0:
        return []
    scored = [
        TopicTransition(current_topic, to_t, cnt, cnt / total)
        for to_t, cnt in transitions.items()
    ]
    scored.sort(key=lambda t: (-t.probability, t.to_topic))
    return scored[:top_k]


def persist_transitions(matrix: dict[str, dict[str, int]], conn: Any) -> None:
    """Upsert matrix into topic_transitions table."""
    now = datetime.utcnow().isoformat()
    for from_t, inner in matrix.items():
        for to_t, cnt in inner.items():
            conn.execute(
                """INSERT INTO topic_transitions(from_topic, to_topic, count, last_seen)
                   VALUES(?,?,?,?)
                   ON CONFLICT(from_topic, to_topic) DO UPDATE SET
                       count = ?,
                       last_seen = ?""",
                (from_t, to_t, cnt, now, cnt, now),
            )
    conn.commit()


def load_transitions(conn: Any) -> dict[str, dict[str, int]]:
    """Load persisted matrix from topic_transitions table."""
    rows = conn.execute(
        "SELECT from_topic, to_topic, count FROM topic_transitions"
    ).fetchall()
    matrix: dict[str, dict[str, int]] = {}
    for from_t, to_t, cnt in rows:
        matrix.setdefault(from_t, {})[to_t] = cnt
    return matrix
