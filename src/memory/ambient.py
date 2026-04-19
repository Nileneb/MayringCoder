"""Ambient Context Layer — kompakter Projekt-Snapshot für Pi-Agent."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

@dataclass
class TriggerResult:
    """Result of trigger_scan() — context string + which triggers fired."""
    context: str
    trigger_ids: list[str]  # e.g. ["keyword:creditservice", "cluster:CreditCluster"]


@dataclass
class ContextFeedback:
    """Implicit feedback record for one Pi-Agent interaction."""
    trigger_ids: list[str]
    context_text: str
    was_referenced: bool
    led_to_retrieval: bool
    relevance_score: float   # 0.0–1.0
    captured_at: str


_SNAPSHOT_SYSTEM = (
    "Du bist ein präziser Assistent. Erstelle einen kompakten Projekt-Snapshot auf Deutsch. "
    "Keine Begrüßung, kein Nachsatz. Maximal 600 Wörter."
)

_SNAPSHOT_PROMPT = """\
## Aktuelle Conversations (letzte 5 Zusammenfassungen)
{conversation_summaries}

## Offene Issues / bekannte Probleme
{issue_summaries}

## Top-Verbindungen im Wiki (stärkste Abhängigkeiten)
{wiki_connections}

---
Erstelle einen Projekt-Snapshot im Format:

**Aktueller Stand:** <1-2 Sätze was zuletzt implementiert/geändert wurde>

**Architektur-Hotspots:** <die 3-5 wichtigsten Dateien/Module mit kurzer Rolle>

**Offene Punkte:** <max. 5 konkrete TODOs oder bekannte Probleme>

**Wichtige Zusammenhänge:** <max. 5 Datei/Modul-Paare mit Erklärung warum sie zusammengehören>
"""

def _load_recent_conversations(conn: Any, repo_slug: str, limit: int = 5) -> list[str]:
    """Lade die letzten N Conversation-Summaries aus SQLite."""
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
    return [r[0][:500] for r in rows]


def _load_recent_issues(conn: Any, repo_slug: str, limit: int = 10) -> list[str]:
    """Lade die letzten N Issue-Summaries aus SQLite."""
    rows = conn.execute(
        """SELECT c.text FROM chunks c
           JOIN sources s ON c.source_id = s.source_id
           WHERE s.source_type = 'github_issue'
             AND (s.repo = ? OR ? = '')
             AND c.is_active = 1
           ORDER BY s.captured_at DESC
           LIMIT ?""",
        (repo_slug, repo_slug, limit),
    ).fetchall()
    return [r[0][:300] for r in rows]


def _load_wiki_top_connections(repo_slug: str, limit: int = 10) -> str:
    """Lade die stärksten Wiki-Verbindungen aus dem Wiki-Markdown."""
    wiki_path = Path("cache") / f"{repo_slug}_wiki.md"
    if not wiki_path.exists():
        return "(kein Wiki vorhanden)"
    content = wiki_path.read_text(encoding="utf-8")
    lines = [line for line in content.splitlines() if line.startswith("- →")]
    return "\n".join(lines[:limit]) or "(keine Verbindungen)"


def _cosine(a: list[float], b: list[float]) -> float:
    """Inline cosine similarity. Returns 0.0 on zero-vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _is_trigger_active(trigger_id: str, conn: Any) -> bool:
    """Returns True if trigger is active or not yet in trigger_stats."""
    row = conn.execute(
        "SELECT is_active FROM trigger_stats WHERE trigger_id = ?", (trigger_id,)
    ).fetchone()
    return row is None or bool(row[0])


def trigger_scan(
    user_message: str,
    keyword_index: dict[str, list[str]],
    cluster_embs: dict[str, list[float]],
    ollama_url: str,
    conn: Any = None,
    threshold: float = 0.75,
    max_tokens: int = 400,
) -> TriggerResult:
    """Cascaded trigger scan: keyword-first (< 1ms), embedding only on miss.

    Returns TriggerResult with context string (max_tokens chars) and fired trigger_ids.
    """
    msg_lower = user_message.lower()
    words = set(w.strip(".,!?;:()[]{}\"'") for w in msg_lower.split() if len(w) > 2)

    matched_clusters: list[str] = []
    fired_ids: list[str] = []

    # Stage 1: keyword match (< 1ms)
    for word in words:
        if word in keyword_index:
            trigger_id = f"keyword:{word}"
            if conn is not None and not _is_trigger_active(trigger_id, conn):
                continue
            matched_clusters.extend(keyword_index[word])
            fired_ids.append(trigger_id)

    if matched_clusters:
        top = list(dict.fromkeys(matched_clusters))[:3]
        ctx = f"[Relevante Cluster: {', '.join(top)}]"[:max_tokens]
        return TriggerResult(context=ctx, trigger_ids=fired_ids)

    # Stage 2: embedding cosine (only if no keyword hit)
    if not cluster_embs or not ollama_url:
        return TriggerResult(context="", trigger_ids=[])

    try:
        from src.analysis.context import _embed_texts
        vecs = _embed_texts([user_message], ollama_url)
        if not vecs:
            return TriggerResult(context="", trigger_ids=[])
        q_vec = vecs[0]
        scores = [(name, _cosine(q_vec, emb)) for name, emb in cluster_embs.items()]
        scores.sort(key=lambda x: -x[1])
        active_pairs = [
            (name, f"cluster:{name}")
            for name, _ in [(n, s) for n, s in scores[:3] if s >= threshold]
            if conn is None or _is_trigger_active(f"cluster:{name}", conn)
        ]
        top_names = [n for n, _ in active_pairs]
        trigger_ids_emb = [t for _, t in active_pairs]
        if top_names:
            ctx = f"[Relevante Cluster: {', '.join(top_names)}]"[:max_tokens]
            return TriggerResult(context=ctx, trigger_ids=trigger_ids_emb)
    except Exception as _exc:
        import warnings
        warnings.warn(f"trigger_scan embedding failed: {_exc}", stacklevel=2)

    return TriggerResult(context="", trigger_ids=[])


def compute_feedback(
    context_text: str,
    llm_response: str,
    trigger_ids: list[str],
    led_to_retrieval: bool,
    conn: Any,
    ollama_url: str,
) -> ContextFeedback:
    """Compute implicit feedback by embedding-cosine context vs. response.

    Persists result to context_feedback_log. Returns ContextFeedback.
    """
    was_referenced = False
    relevance_score = 0.0

    if context_text and llm_response and ollama_url:
        try:
            from src.analysis.context import _embed_texts
            vecs = _embed_texts([context_text, llm_response], ollama_url)
            if len(vecs) == 2:
                relevance_score = round(_cosine(vecs[0], vecs[1]), 4)
                was_referenced = relevance_score >= 0.65
        except Exception:
            pass

    captured_at = datetime.utcnow().isoformat()
    fb = ContextFeedback(
        trigger_ids=trigger_ids,
        context_text=context_text[:500],
        was_referenced=was_referenced,
        led_to_retrieval=led_to_retrieval,
        relevance_score=relevance_score,
        captured_at=captured_at,
    )

    try:
        conn.execute(
            """INSERT INTO context_feedback_log
               (trigger_ids, context_text, was_referenced, led_to_retrieval, relevance_score, captured_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (json.dumps(trigger_ids), context_text[:500],
             int(was_referenced), int(led_to_retrieval),
             relevance_score, captured_at),
        )
        conn.commit()
    except Exception:
        pass

    return fb


def update_trigger_stats(
    trigger_ids: list[str],
    was_referenced: bool,
    conn: Any,
    deactivate_threshold: float = 0.10,
    min_fires_for_deactivation: int = 50,
) -> None:
    """Increment fire_count + ref_count. Deactivate if below threshold after min_fires."""
    now = datetime.utcnow().isoformat()
    for trigger_id in trigger_ids:
        conn.execute(
            """INSERT INTO trigger_stats (trigger_id, fire_count, ref_count, is_active, last_fired)
               VALUES (?, 1, ?, 1, ?)
               ON CONFLICT(trigger_id) DO UPDATE SET
                   fire_count = fire_count + 1,
                   ref_count  = ref_count + ?,
                   last_fired = ?""",
            (trigger_id, int(was_referenced), now, int(was_referenced), now),
        )
        row = conn.execute(
            "SELECT fire_count, ref_count FROM trigger_stats WHERE trigger_id = ?",
            (trigger_id,),
        ).fetchone()
        if row:
            fires, refs = row
            if fires >= min_fires_for_deactivation:
                rate = refs / fires
                if rate < deactivate_threshold:
                    conn.execute(
                        "UPDATE trigger_stats SET is_active = 0 WHERE trigger_id = ?",
                        (trigger_id,),
                    )
    conn.commit()


def generate_ambient_snapshot(
    conn: Any,
    ollama_url: str,
    model: str,
    repo_slug: str = "",
    workspace_id: str = "system",
) -> str | None:
    """Generiere Ambient-Snapshot via LLM und speichere in SQLite.

    Returns snapshot text on success, None if model is empty.
    """
    if not model:
        return None

    from src.analysis.analyzer import _ollama_generate
    from src.memory.schema import Source
    from src.memory.ingest import ingest

    convs = _load_recent_conversations(conn, repo_slug)
    issues = _load_recent_issues(conn, repo_slug)
    wiki_top = _load_wiki_top_connections(repo_slug)

    prompt = _SNAPSHOT_PROMPT.format(
        conversation_summaries="\n".join(f"- {s}" for s in convs) or "(keine)",
        issue_summaries="\n".join(f"- {s}" for s in issues) or "(keine)",
        wiki_connections=wiki_top,
    )

    try:
        snapshot_text = _ollama_generate(
            prompt, ollama_url, model, "ambient_snapshot",
            system_prompt=_SNAPSHOT_SYSTEM,
        )
    except Exception:
        return None

    source_id = f"ambient:{repo_slug or 'global'}:snapshot"
    content_hash = "sha256:" + hashlib.sha256(snapshot_text.encode()).hexdigest()[:16]

    src = Source(
        source_id=source_id,
        source_type="ambient_snapshot",
        repo=repo_slug,
        path="ambient/snapshot",
        branch="local",
        commit="",
        content_hash=content_hash,
    )
    ingest(
        src, snapshot_text, conn, None,
        ollama_url, "",
        opts={"categorize": False, "chunk_level": "ambient_snapshot"},
        workspace_id=workspace_id,
    )
    return snapshot_text


def load_ambient_snapshot(conn: Any, repo_slug: str = "") -> str | None:
    """Lade den letzten Ambient-Snapshot aus SQLite."""
    source_id = f"ambient:{repo_slug or 'global'}:snapshot"
    row = conn.execute(
        """SELECT c.text FROM chunks c
           JOIN sources s ON c.source_id = s.source_id
           WHERE s.source_id = ? AND c.is_active = 1
           ORDER BY s.captured_at DESC LIMIT 1""",
        (source_id,),
    ).fetchone()
    return row[0] if row else None


def _safe_repo_slug(repo_slug: str) -> str:
    """Return a deterministic filesystem-safe cache key for repo_slug, else empty string."""
    if not repo_slug:
        return ""
    return hashlib.sha256(repo_slug.encode("utf-8")).hexdigest()


def _safe_cache_file(cache_dir: Path, repo_slug: str, suffix: str) -> Path | None:
    """Build a cache file path under cache_dir from repo_slug, else return None."""
    safe_slug = _safe_repo_slug(repo_slug)
    if not safe_slug:
        return None

    base_dir = cache_dir.resolve()
    candidate = (base_dir / f"{safe_slug}_{suffix}").resolve()

    try:
        candidate.relative_to(base_dir)
    except Exception:
        return None

    return candidate


def build_context(
    task: str,
    conn: Any,
    ollama_url: str,
    repo_slug: str = "",
    _out_trigger_ids: list | None = None,
) -> str:
    """Orchestrator: lädt Snapshot + Trigger-Scan → kompakter Kontext-String.

    Leise skippen wenn kein Snapshot vorhanden (kein LLM-Call).
    """
    safe_repo_slug = _safe_repo_slug(repo_slug)
    snapshot = load_ambient_snapshot(conn, safe_repo_slug)
    if not snapshot:
        return ""

    keyword_index: dict[str, list[str]] = {}
    cluster_embs: dict[str, list[float]] = {}
    cache_dir = Path("cache").resolve()
    idx_path = _safe_cache_file(cache_dir, safe_repo_slug, "wiki_index.json")
    emb_path = _safe_cache_file(cache_dir, safe_repo_slug, "wiki_clusters_emb.json")
    if idx_path and idx_path.exists():
        try:
            keyword_index = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if emb_path and emb_path.exists():
        try:
            cluster_embs = json.loads(emb_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    result = trigger_scan(task, keyword_index, cluster_embs, ollama_url, conn=conn)
    if _out_trigger_ids is not None:
        _out_trigger_ids.extend(result.trigger_ids)
    trigger_hint = result.context

    parts = [f"## Projekt-Snapshot\n{snapshot}"]
    if trigger_hint:
        parts.append(f"## Trigger-Kontext\n{trigger_hint}")
    return "\n\n".join(parts)
