"""Paper-specific edge detection rules for wiki_v2.

Rules 1 (citation) and 4 (keyword_cooccurrence) are LLM-free.
Rules 2 (shared_concept), 3 (method_chain), 5 (dataset) use a mini-LLM
prompt on compressed chunk text, cached in wiki_paper_cache.
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

from src.memory.schema import Chunk
from src.wiki_v2.models import WikiEdge

_DOI_RE = re.compile(r"10\.\d{4,}/\S+")
_BRACKET_REF_RE = re.compile(r"\[(\w[\w\s,\.]+\d{4}[a-z]?)\]")
_AUTHOR_YEAR_RE = re.compile(r"\(([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s+\d{4}\)")

_MINI_LLM_MAX_CHARS = 2000
_TOP_CHUNKS = 8

PAPER_EDGE_TYPES = ["citation", "shared_concept", "method_chain", "keyword_cooccurrence", "dataset_coupling"]


def _compress_chunks(chunks: list[Chunk], char_budget: int = _MINI_LLM_MAX_CHARS) -> str:
    return "\n\n".join(c.text for c in chunks[:_TOP_CHUNKS] if c.text)[:char_budget]


def _call_ollama(text: str, instruction: str, ollama_url: str, model: str) -> list[str]:
    prompt = (
        f"Aufgabe: {instruction}\n\nText:\n{text}\n\n"
        f"Antworte NUR mit einer JSON-Liste von Strings, z.B. [\"Begriff1\"]. Keine Erklärungen."
    )
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read()).get("response", "[]")
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return []


def _chunks_by_source(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    out: dict[str, list[Chunk]] = {}
    for c in chunks:
        out.setdefault(c.source_id, []).append(c)
    return out


def detect_citations(chunks: list[Chunk], workspace_id: str, repo_slug: str) -> list[WikiEdge]:
    by_source = _chunks_by_source(chunks)
    refs: dict[str, set[str]] = {}
    for sid, sc in by_source.items():
        combined = " ".join(c.text for c in sc)
        found: set[str] = set()
        for m in _DOI_RE.finditer(combined):
            found.add(m.group().lower())
        for m in _BRACKET_REF_RE.finditer(combined):
            found.add(m.group(1).lower().strip())
        for m in _AUTHOR_YEAR_RE.finditer(combined):
            found.add(m.group(1).lower().strip())
        refs[sid] = found

    source_ids = list(by_source)
    edges: list[WikiEdge] = []
    for i, sid_a in enumerate(source_ids):
        for sid_b in source_ids[i + 1:]:
            overlap = refs.get(sid_a, set()) & refs.get(sid_b, set())
            if overlap:
                edges.append(WikiEdge(
                    source=sid_a, target=sid_b, repo_slug=repo_slug, workspace_id=workspace_id,
                    type="citation", weight=1.0, context=", ".join(sorted(overlap)[:3]),
                ))
    return edges


def detect_keyword_overlap(
    chunks: list[Chunk], workspace_id: str, repo_slug: str, min_shared: int = 1,
) -> list[WikiEdge]:
    by_source = _chunks_by_source(chunks)
    source_ids = list(by_source)
    labels: dict[str, set[str]] = {
        sid: {lbl.strip() for c in sc for lbl in (c.category_labels or []) if lbl.strip()}
        for sid, sc in by_source.items()
    }
    edges: list[WikiEdge] = []
    for i, sid_a in enumerate(source_ids):
        for sid_b in source_ids[i + 1:]:
            shared = labels.get(sid_a, set()) & labels.get(sid_b, set())
            if len(shared) >= min_shared:
                edges.append(WikiEdge(
                    source=sid_a, target=sid_b, repo_slug=repo_slug, workspace_id=workspace_id,
                    type="keyword_cooccurrence", weight=0.5, context=", ".join(sorted(shared)[:5]),
                ))
    return edges


def _llm_pair_edges(
    chunks: list[Chunk], conn: Any, ollama_url: str, model: str,
    workspace_id: str, repo_slug: str,
    rule_name: str, instruction: str, edge_type: str, weight: float,
) -> list[WikiEdge]:
    from src.memory.store import get_paper_cache, set_paper_cache
    by_source = _chunks_by_source(chunks)
    per_source: dict[str, set[str]] = {}
    for sid, sc in by_source.items():
        cached = get_paper_cache(conn, sid, rule_name)
        if cached is not None:
            per_source[sid] = set(cached)
            continue
        extracted = _call_ollama(_compress_chunks(sc), instruction, ollama_url, model)
        set_paper_cache(conn, sid, rule_name, extracted)
        per_source[sid] = set(extracted)

    source_ids = list(by_source)
    edges: list[WikiEdge] = []
    for i, sid_a in enumerate(source_ids):
        for sid_b in source_ids[i + 1:]:
            shared = per_source.get(sid_a, set()) & per_source.get(sid_b, set())
            if shared:
                edges.append(WikiEdge(
                    source=sid_a, target=sid_b, repo_slug=repo_slug, workspace_id=workspace_id,
                    type=edge_type, weight=weight, context=", ".join(sorted(shared)[:5]),
                ))
    return edges


def detect_shared_concepts(chunks, conn, ollama_url, model, workspace_id, repo_slug):
    return _llm_pair_edges(
        chunks, conn, ollama_url, model, workspace_id, repo_slug,
        "shared_concept",
        "Liste die domänenspezifischen Fachbegriffe aus diesem Text",
        "shared_concept", 0.8,
    )


def detect_method_chains(chunks, conn, ollama_url, model, workspace_id, repo_slug):
    return _llm_pair_edges(
        chunks, conn, ollama_url, model, workspace_id, repo_slug,
        "method_chain",
        "Welche externen Methoden, Frameworks oder Ansätze werden hier referenziert oder genutzt?",
        "method_chain", 0.7,
    )


def detect_dataset_pairs(chunks, conn, ollama_url, model, workspace_id, repo_slug):
    return _llm_pair_edges(
        chunks, conn, ollama_url, model, workspace_id, repo_slug,
        "dataset",
        "Welche konkreten Datensätze, Corpora oder empirischen Datenquellen werden genannt?",
        "dataset_coupling", 0.6,
    )


def detect_from_papers(
    chunks: list[Chunk],
    conn: Any | None,
    ollama_url: str,
    model: str,
    workspace_id: str,
    repo_slug: str,
) -> list[WikiEdge]:
    if not chunks:
        return []
    edges: list[WikiEdge] = []
    edges += detect_citations(chunks, workspace_id, repo_slug)
    edges += detect_keyword_overlap(chunks, workspace_id, repo_slug)
    if conn is not None and ollama_url:
        edges += detect_shared_concepts(chunks, conn, ollama_url, model, workspace_id, repo_slug)
        edges += detect_method_chains(chunks, conn, ollama_url, model, workspace_id, repo_slug)
        edges += detect_dataset_pairs(chunks, conn, ollama_url, model, workspace_id, repo_slug)
    seen: dict[tuple, WikiEdge] = {}
    for e in edges:
        key = (min(e.source, e.target), max(e.source, e.target), e.type, e.workspace_id)
        if key not in seen or e.weight > seen[key].weight:
            seen[key] = e
    return list(seen.values())
