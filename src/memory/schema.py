"""Memory data model — Source, Chunk, MemoryKey, RetrievalRecord.

These dataclasses are the single source of truth for the Memory layer.
They are used by memory_store.py, memory_ingest.py, and memory_retrieval.py.

Key format:  memory:{scope}:{category}:{source_fingerprint}:{chunk_hash_prefix}
Example:     memory:repo:auth:owner-name-src-user_service.py:9f3a1b2c
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """Represents a single ingested source (file, doc, snippet, …)."""

    source_id: str          # Canonical ID, e.g. "repo:owner/name:path/to/file.py"
    source_type: str        # "repo_file" | "doc" | "note" | "conversation"
    repo: str               # "owner/name" or empty for non-repo sources
    path: str               # Relative path within repo, or arbitrary label
    branch: str = "main"
    commit: str = ""
    content_hash: str = ""  # sha256 of raw source content
    captured_at: str = field(default_factory=lambda: _now_iso())
    visibility: str = "private"   # "private" | "org" | "public"
    org_id: str | None = None

    @staticmethod
    def make_id(repo: str, path: str) -> str:
        """Build a canonical source_id from repo and path."""
        return f"repo:{repo}:{path}"

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "repo": self.repo,
            "path": self.path,
            "branch": self.branch,
            "commit": self.commit,
            "content_hash": self.content_hash,
            "captured_at": self.captured_at,
            "visibility": self.visibility,
            "org_id": self.org_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        data = {k: d.get(k, "") for k in cls.__dataclass_fields__}
        data["org_id"] = d.get("org_id")  # None when absent or NULL
        return cls(**data)


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A segmented, versioned unit of memory derived from a Source."""

    chunk_id: str
    source_id: str
    parent_chunk_id: str | None = None
    chunk_level: str = "file"     # "file" | "class" | "function" | "section" | "block"
    ordinal: int = 0              # Position within source (0-based)
    start_offset: int = 0
    end_offset: int = 0
    text: str = ""
    text_hash: str = ""           # sha256 of text (dedup key for exact match)
    summary: str = ""
    category_labels: list[str] = field(default_factory=list)
    category_version: str = "mayring-inductive-v1"
    embedding_model: str = "nomic-embed-text"
    embedding_id: str = ""
    quality_score: float = 0.0
    dedup_key: str = ""           # sha256 of normalized text (for near-dedup)
    category_source: str = ""     # "deductive"|"inductive"|"hybrid"|"fallback"|"manual"|""
    category_confidence: float = 0.0
    igio_axis: str = ""           # "" | "issue" | "goal" | "intervention" | "outcome"
    igio_confidence: float = 0.0
    igio_classified_at: str = ""
    created_at: str = field(default_factory=lambda: _now_iso())
    workspace_id: str = "default"
    superseded_by: str | None = None
    is_active: bool = True

    @staticmethod
    def make_id(source_id: str, ordinal: int, chunk_level: str) -> str:
        raw = f"{source_id}:{chunk_level}:{ordinal}"
        return "chk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def compute_text_hash(text: str) -> str:
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "parent_chunk_id": self.parent_chunk_id,
            "chunk_level": self.chunk_level,
            "ordinal": self.ordinal,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "text": self.text,
            "text_hash": self.text_hash,
            "summary": self.summary,
            "category_labels": self.category_labels,
            "category_version": self.category_version,
            "embedding_model": self.embedding_model,
            "embedding_id": self.embedding_id,
            "quality_score": self.quality_score,
            "dedup_key": self.dedup_key,
            "category_source": self.category_source,
            "category_confidence": self.category_confidence,
            "igio_axis": self.igio_axis,
            "igio_confidence": self.igio_confidence,
            "igio_classified_at": self.igio_classified_at,
            "created_at": self.created_at,
            "superseded_by": self.superseded_by,
            "is_active": self.is_active,
            "workspace_id": self.workspace_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        data = dict(d)
        # category_labels is stored as comma-separated string in SQLite
        if isinstance(data.get("category_labels"), str):
            raw = data["category_labels"]
            data["category_labels"] = [c for c in raw.split(",") if c] if raw else []
        if "is_active" in data:
            data["is_active"] = bool(data["is_active"])
        return cls(**{k: data.get(k, v) for k, v in cls.__dataclass_fields__.items()
                      if k not in ("category_labels",)} | {
            "category_labels": data.get("category_labels", []),
        })


# ---------------------------------------------------------------------------
# MemoryKey
# ---------------------------------------------------------------------------

def make_memory_key(scope: str, category: str, source_fingerprint: str, chunk_hash_prefix: str) -> str:
    """Build the canonical memory key.

    Format: memory:{scope}:{category}:{source_fingerprint}:{chunk_hash_prefix}
    Example: memory:repo:auth:owner-name-src-user_service.py:9f3a1b2c
    """
    return f"memory:{scope}:{category}:{source_fingerprint}:{chunk_hash_prefix}"


def source_fingerprint(source_id: str) -> str:
    """Derive a short filesystem-safe fingerprint from a source_id."""
    # e.g. "repo:owner/name:src/user_service.py" → "owner-name-src-user_service.py"
    parts = source_id.split(":", 2)
    path = parts[2] if len(parts) == 3 else source_id
    slug = path.replace("/", "-").replace("\\", "-").replace(":", "-")
    # Keep it short
    if len(slug) > 48:
        slug = slug[:40] + "-" + hashlib.sha256(slug.encode()).hexdigest()[:7]
    return slug


# ---------------------------------------------------------------------------
# RetrievalRecord
# ---------------------------------------------------------------------------

@dataclass
class RetrievalRecord:
    """One ranked result from a hybrid memory search."""

    chunk_id: str
    score_vector: float = 0.0
    score_symbolic: float = 0.0
    score_recency: float = 0.0
    score_source_affinity: float = 0.0
    score_final: float = 0.0
    reasons: list[str] = field(default_factory=list)
    source_id: str = ""
    text: str = ""
    summary: str = ""
    category_labels: list[str] = field(default_factory=list)
    also_in_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "score_vector": round(self.score_vector, 4),
            "score_symbolic": round(self.score_symbolic, 4),
            "score_recency": round(self.score_recency, 4),
            "score_source_affinity": round(self.score_source_affinity, 4),
            "score_final": round(self.score_final, 4),
            "reasons": self.reasons,
            "source_id": self.source_id,
            "text": self.text,
            "summary": self.summary,
            "category_labels": self.category_labels,
            "also_in_sources": self.also_in_sources,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
