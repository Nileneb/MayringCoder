"""Memory Retrieval Policy — 4-stage hybrid search.

Stage 1: SQLite scope filter (repo, category, source_type, active)
Stage 2: Symbolic matching (token overlap + path bonus)
Stage 3: Vector retrieval (ChromaDB similarity)
Stage 4: Re-ranking (weighted combination)
Stage 5: compress_for_prompt() — format results for Claude
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from src.memory_schema import Chunk, RetrievalRecord
from src.memory_store import get_chunk, kv_get

try:
    from src.context import _embed_texts
    _HAS_EMBED = True
except ImportError:
    _HAS_EMBED = False

_WEIGHTS = {
    "vector": 0.45,
    "symbolic": 0.25,
    "recency": 0.15,
    "source_affinity": 0.15,
}

_RECENCY_DECAY_DAYS = 30.0


# ---------------------------------------------------------------------------
# Stage 1: Scope filter
# ---------------------------------------------------------------------------

def _scope_filter(
    conn: sqlite3.Connection,
    repo: str | None = None,
    categories: list[str] | None = None,
    source_type: str | None = None,
) -> list[str]:
    """Return chunk_ids of active chunks matching hard scope filters."""
    query = """
        SELECT c.chunk_id, c.category_labels
        FROM chunks c
        JOIN sources s ON c.source_id = s.source_id
        WHERE c.is_active = 1
    """
    params: list[str] = []

    if repo:
        query += " AND s.repo = ?"
        params.append(repo)
    if source_type:
        query += " AND s.source_type = ?"
        params.append(source_type)

    rows = conn.execute(query, params).fetchall()

    if not categories:
        return [row[0] for row in rows]

    # Category filter: post-query (comma-split stored labels)
    cat_set = {c.lower() for c in categories}
    result = []
    for chunk_id, labels_str in rows:
        chunk_cats = {l.strip().lower() for l in (labels_str or "").split(",") if l.strip()}
        if chunk_cats & cat_set:
            result.append(chunk_id)
    return result


# ---------------------------------------------------------------------------
# Stage 2: Symbolic scoring
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Split text into lowercase alphanumeric tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _symbolic_score(chunk: Chunk, query_terms: set[str]) -> float:
    """0.0–1.0 score based on token overlap + path/category bonuses."""
    if not query_terms:
        return 0.0

    chunk_terms = _tokenize(chunk.text) | _tokenize(chunk.summary)
    overlap = len(query_terms & chunk_terms) / len(query_terms)

    # Bonus: query term appears in source path (file relevance)
    path_bonus = 0.0
    source_path = chunk.source_id.lower()
    for term in query_terms:
        if term in source_path:
            path_bonus = 0.2
            break

    # Bonus: query term matches a category label
    cat_bonus = 0.0
    cat_terms = _tokenize(",".join(chunk.category_labels))
    if query_terms & cat_terms:
        cat_bonus = 0.1

    return min(1.0, overlap + path_bonus + cat_bonus)


# ---------------------------------------------------------------------------
# Stage 3 helper: normalize ChromaDB distances to scores
# ---------------------------------------------------------------------------

def _normalize_vector_scores(
    chroma_ids: list[str],
    chroma_distances: list[float],
    candidate_ids: set[str],
) -> dict[str, float]:
    """Convert ChromaDB distances to similarity scores (0.0–1.0), filtered to candidates."""
    scores: dict[str, float] = {}
    for cid, dist in zip(chroma_ids, chroma_distances):
        if cid not in candidate_ids:
            continue
        # ChromaDB cosine distance: score = 1 - distance (clamped)
        scores[cid] = max(0.0, min(1.0, 1.0 - dist))
    return scores


# ---------------------------------------------------------------------------
# Stage 4: Recency and affinity
# ---------------------------------------------------------------------------

def _recency_score(chunk: Chunk) -> float:
    """Decay: 1.0 for brand-new chunks, 0.0 for chunks >30 days old."""
    try:
        created = datetime.fromisoformat(chunk.created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_old = (now - created).total_seconds() / 86400.0
        return max(0.0, 1.0 - days_old / _RECENCY_DECAY_DAYS)
    except Exception:
        return 0.0


def _source_affinity_score(chunk: Chunk, affinity_source_id: str | None) -> float:
    """1.0 if chunk belongs to the affinity source, else 0.0."""
    if affinity_source_id and chunk.source_id == affinity_source_id:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Stage 4: Re-ranking
# ---------------------------------------------------------------------------

def _rerank(
    candidates: list[Chunk],
    vector_scores: dict[str, float],
    symbolic_scores: dict[str, float],
    top_k: int,
    affinity_source_id: str | None = None,
) -> list[RetrievalRecord]:
    """Combine scores and return top_k RetrievalRecords sorted by score_final DESC."""
    records: list[RetrievalRecord] = []

    for chunk in candidates:
        sv = vector_scores.get(chunk.chunk_id, 0.0)
        ss = symbolic_scores.get(chunk.chunk_id, 0.0)
        sr = _recency_score(chunk)
        sa = _source_affinity_score(chunk, affinity_source_id)

        score_final = (
            _WEIGHTS["vector"] * sv
            + _WEIGHTS["symbolic"] * ss
            + _WEIGHTS["recency"] * sr
            + _WEIGHTS["source_affinity"] * sa
        )

        reasons: list[str] = []
        if sv > 0.5:
            reasons.append("embedding_similarity")
        if ss > 0.3:
            reasons.append("token_overlap")
        if sr > 0.8:
            reasons.append("recent_chunk")
        if sa > 0:
            reasons.append("source_affinity_match")

        records.append(
            RetrievalRecord(
                chunk_id=chunk.chunk_id,
                score_vector=sv,
                score_symbolic=ss,
                score_recency=sr,
                score_source_affinity=sa,
                score_final=score_final,
                reasons=reasons,
                source_id=chunk.source_id,
                text=chunk.text,
                summary=chunk.summary,
                category_labels=chunk.category_labels,
            )
        )

    records.sort(key=lambda r: r.score_final, reverse=True)
    return records[:top_k]


# ---------------------------------------------------------------------------
# Main search entry point
# ---------------------------------------------------------------------------

def search(
    query: str,
    conn: sqlite3.Connection,
    chroma_collection: Any,
    ollama_url: str,
    opts: dict | None = None,
) -> list[RetrievalRecord]:
    """4-stage hybrid memory search.

    opts:
        repo (str): filter by repo
        categories (list[str]): filter by category labels
        source_type (str): filter by source_type
        top_k (int, default 8): max results
        include_text (bool, default True): populate .text on results
        source_affinity (str): source_id to boost
    """
    opts = opts or {}
    top_k: int = int(opts.get("top_k", 8))
    repo: str | None = opts.get("repo")
    categories: list[str] | None = opts.get("categories")
    source_type: str | None = opts.get("source_type")
    affinity_source_id: str | None = opts.get("source_affinity")

    # Stage 1: scope filter
    candidate_ids = _scope_filter(conn, repo=repo, categories=categories, source_type=source_type)
    if not candidate_ids:
        return []

    # Load candidate chunks (KV cache first, then SQLite)
    candidates: list[Chunk] = []
    for cid in candidate_ids:
        cached = kv_get(cid)
        if cached is not None:
            try:
                chunk = Chunk.from_dict(cached)
                candidates.append(chunk)
                continue
            except Exception:
                pass
        chunk = get_chunk(conn, cid)
        if chunk is not None:
            candidates.append(chunk)

    if not candidates:
        return []

    # Stage 2: symbolic scoring
    query_terms = _tokenize(query)
    symbolic_scores: dict[str, float] = {
        c.chunk_id: _symbolic_score(c, query_terms) for c in candidates
    }

    # Stage 3: vector retrieval
    vector_scores: dict[str, float] = {}
    if chroma_collection is not None and _HAS_EMBED:
        try:
            query_emb = _embed_texts([query], ollama_url)[0]
            n_results = min(top_k * 2, chroma_collection.count())
            if n_results > 0:
                results = chroma_collection.query(
                    query_embeddings=[query_emb],
                    n_results=n_results,
                    include=["distances"],
                )
                ids_list = results.get("ids", [[]])[0]
                dist_list = results.get("distances", [[]])[0]
                candidate_set = {c.chunk_id for c in candidates}
                vector_scores = _normalize_vector_scores(ids_list, dist_list, candidate_set)
        except Exception:
            pass  # Vector retrieval is best-effort

    # Stage 4: re-rank
    return _rerank(candidates, vector_scores, symbolic_scores, top_k, affinity_source_id)


# ---------------------------------------------------------------------------
# Stage 5: compress_for_prompt
# ---------------------------------------------------------------------------

def compress_for_prompt(results: list[RetrievalRecord], char_budget: int) -> str:
    """Format retrieval results as a context string within char_budget.

    Strategy:
    1. Deduplicate by source_id (keep highest-scoring chunk per source)
    2. Prefer summary over text when text > 500 chars
    3. Stop adding entries when budget would be exceeded
    """
    if not results:
        return ""

    # Step 1: dedup by source_id
    seen_sources: dict[str, RetrievalRecord] = {}
    for r in results:
        if r.source_id not in seen_sources:
            seen_sources[r.source_id] = r

    deduped = list(seen_sources.values())

    lines = ["## Memory Context", ""]
    used = sum(len(l) + 1 for l in lines)

    for r in deduped:
        cats = ", ".join(r.category_labels) if r.category_labels else "?"
        body = r.summary if (r.summary and len(r.text) > 500) else r.text
        # Truncate body to remaining budget divided by remaining entries
        remaining_entries = len(deduped) - deduped.index(r)
        per_entry_budget = max(100, (char_budget - used) // max(1, remaining_entries))
        if len(body) > per_entry_budget:
            body = body[:per_entry_budget - 3] + "..."

        entry = f"- [{cats}] {r.source_id}\n  {body}"
        entry_len = len(entry) + 1

        if used + entry_len > char_budget:
            break

        lines.append(entry)
        used += entry_len

    if len(lines) <= 2:
        return ""

    return "\n".join(lines)
