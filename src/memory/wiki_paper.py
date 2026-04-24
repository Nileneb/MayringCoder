"""Paper-spezifische Wiki-Regeln und Cache-Helpers."""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


_CITE_NUM_RE = re.compile(r'\[(\d+(?:,\s*\d+)*)\]')
_CITE_AUTH_RE = re.compile(
    r'\(([A-Z][a-z]+ et al\.?,?\s*\d{4}|[A-Z][a-z]+\s*&\s*[A-Z][a-z]+,?\s*\d{4})\)'
)

_KNOWN_METHODS = {
    "bert", "gpt", "gpt-2", "gpt-3", "gpt-4", "t5", "llama", "mistral", "qwen",
    "transformer", "attention", "lora", "qlora", "rag", "svm", "lstm", "cnn",
    "diffusion", "vae", "gan", "clip", "vit", "resnet", "bert-base", "roberta",
    "xlm", "deberta", "electra", "sentence-bert", "faiss", "bm25",
}
_KNOWN_DATASETS = {
    "mmlu", "squad", "squad2", "naturalquestions", "triviaqa", "hotpotqa",
    "imagenet", "coco", "humaneval", "mbpp", "gsm8k", "math", "arc", "hellaswag",
    "winogrande", "piqa", "sst-2", "mnli", "snli", "commonsenseqa", "boolq",
    "ms marco", "beir", "mteb", "glue", "superglue", "c4", "pile", "openwebtext",
}


def _cache_get(conn: Any, source_id: str, rule_name: str) -> list | None:
    row = conn.execute(
        "SELECT extracted FROM wiki_paper_cache WHERE source_id=? AND rule_name=?",
        (source_id, rule_name),
    ).fetchone()
    return json.loads(row[0]) if row else None


def _cache_put(conn: Any, source_id: str, rule_name: str, extracted: list) -> None:
    from datetime import datetime
    conn.execute(
        "INSERT OR REPLACE INTO wiki_paper_cache(source_id, rule_name, extracted, created_at) VALUES(?,?,?,?)",
        (source_id, rule_name, json.dumps(extracted), datetime.utcnow().isoformat()),
    )
    conn.commit()


def _paper_source_ids(chunks: list) -> list[str]:
    seen: list[str] = []
    for c in chunks:
        sid = c.source_id if hasattr(c, "source_id") else c.get("source_id", "")
        if sid.startswith("paper:arxiv:") and sid not in seen:
            seen.append(sid)
    return seen


def _chunk_text(c) -> str:
    return c.text if hasattr(c, "text") else c.get("text", "")


def find_citation_pairs(overview_cache: dict, chunks: list) -> list:
    """Papers that cite each other (title-token overlap) → weight 1.0."""
    from src.memory.wiki_core import WikiEdge
    paper_ids = _paper_source_ids(chunks)
    if len(paper_ids) < 2:
        return []

    title_index: dict[str, set[str]] = {}
    for c in chunks:
        sid = c.source_id if hasattr(c, "source_id") else c.get("source_id", "")
        if not sid.startswith("paper:arxiv:"):
            continue
        text = _chunk_text(c)
        first_line = text.split("\n")[0].lstrip("# ").lower()
        tokens = set(w.strip(".,") for w in first_line.split() if len(w) > 3)
        title_index.setdefault(sid, set()).update(tokens)

    edges = []
    for c in chunks:
        sid = c.source_id if hasattr(c, "source_id") else c.get("source_id", "")
        if not sid.startswith("paper:arxiv:"):
            continue
        text = _chunk_text(c).lower()
        for other_sid, tokens in title_index.items():
            if other_sid == sid:
                continue
            if len(tokens) >= 3 and sum(1 for t in tokens if t in text) >= 3:
                edges.append(WikiEdge(sid, other_sid, 1.0, "citation"))
    return edges


def find_shared_concepts(chunks: list, conn: Any, chroma: Any, ollama_url: str, model: str) -> list:
    """Papers sharing similar abstract embeddings → weight = cosine similarity (>= 0.75)."""
    from src.memory.wiki_core import WikiEdge
    if chroma is None:
        return []
    paper_ids = _paper_source_ids(chunks)
    if len(paper_ids) < 2:
        return []

    abstract_map: dict[str, list[float]] = {}
    try:
        result = chroma.get(
            where={"$and": [{"source_id": {"$in": paper_ids}}, {"chunk_level": {"$eq": "abstract"}}]},
            include=["embeddings", "metadatas"],
        )
        embeddings = result.get("embeddings") or []
        metadatas = result.get("metadatas") or []
        for emb, meta in zip(embeddings, metadatas):
            if emb and meta:
                src = meta.get("source_id", "")
                if src:
                    abstract_map[src] = emb
    except Exception:
        return []

    if len(abstract_map) < 2:
        return []

    def _cosine(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    edges = []
    sids = list(abstract_map.keys())
    for i, a in enumerate(sids):
        for b in sids[i + 1:]:
            score = _cosine(abstract_map[a], abstract_map[b])
            if score >= 0.75:
                edges.append(WikiEdge(a, b, round(score, 3), "shared_concept"))
    return edges


def _extract_methods_from_chunks(paper_sid: str, chunks: list) -> list[str]:
    found = []
    for c in chunks:
        sid = c.source_id if hasattr(c, "source_id") else c.get("source_id", "")
        if sid != paper_sid:
            continue
        text = _chunk_text(c).lower()
        for method in _KNOWN_METHODS:
            if method not in found and re.search(r'\b' + re.escape(method) + r'\b', text):
                found.append(method)
    return found


def find_method_chains(chunks: list, conn: Any, chroma: Any, ollama_url: str, model: str) -> list:
    """Papers using the same ML method → weight 0.7."""
    from src.memory.wiki_core import WikiEdge
    paper_ids = _paper_source_ids(chunks)
    if len(paper_ids) < 2:
        return []

    method_papers: dict[str, list[str]] = defaultdict(list)
    for paper_sid in paper_ids:
        cached = _cache_get(conn, paper_sid, "method_chain") if conn else None
        if cached is not None:
            methods_for_paper = cached
        else:
            methods_for_paper = _extract_methods_from_chunks(paper_sid, chunks)
            if conn:
                _cache_put(conn, paper_sid, "method_chain", methods_for_paper)
        for method in methods_for_paper:
            if paper_sid not in method_papers[method]:
                method_papers[method].append(paper_sid)

    edges = []
    for method, sids in method_papers.items():
        unique = list(dict.fromkeys(sids))
        for i, a in enumerate(unique):
            for b in unique[i + 1:]:
                edges.append(WikiEdge(a, b, 0.7, f"method:{method}"))
    return edges


def find_keyword_overlap(overview_cache: dict, chunks: list) -> list:
    """Papers with overlapping keyword sets (Jaccard >= 0.2) → weight = Jaccard score."""
    from src.memory.wiki_core import WikiEdge
    paper_ids = _paper_source_ids(chunks)
    if len(paper_ids) < 2:
        return []

    _STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "are", "from", "our",
        "we", "in", "of", "to", "a", "an", "is", "on", "as", "by", "at",
        "be", "it", "its", "or", "not", "but", "can", "has", "have", "also",
    }

    paper_keywords: dict[str, set[str]] = {}
    for c in chunks:
        sid = c.source_id if hasattr(c, "source_id") else c.get("source_id", "")
        if not sid.startswith("paper:arxiv:"):
            continue
        text = _chunk_text(c).lower()
        words = re.findall(r'\b[a-z][a-z-]{2,}\b', text)
        keywords = {w for w in words if w not in _STOPWORDS}
        paper_keywords.setdefault(sid, set()).update(keywords)

    edges = []
    sids = list(paper_keywords.keys())
    for i, a in enumerate(sids):
        for b in sids[i + 1:]:
            ka, kb = paper_keywords[a], paper_keywords[b]
            union = len(ka | kb)
            if union == 0:
                continue
            jaccard = len(ka & kb) / union
            if jaccard >= 0.2:
                edges.append(WikiEdge(a, b, round(jaccard, 3), "keyword_overlap"))
    return edges


def _extract_datasets_from_chunks(paper_sid: str, chunks: list) -> list[str]:
    found = []
    for c in chunks:
        sid = c.source_id if hasattr(c, "source_id") else c.get("source_id", "")
        if sid != paper_sid:
            continue
        text = _chunk_text(c).lower()
        for ds in _KNOWN_DATASETS:
            if ds not in found and re.search(r'\b' + re.escape(ds) + r'\b', text):
                found.append(ds)
    return found


def find_dataset_pairs(chunks: list, conn: Any, chroma: Any, ollama_url: str, model: str) -> list:
    """Papers using the same dataset → weight 0.8."""
    from src.memory.wiki_core import WikiEdge
    paper_ids = _paper_source_ids(chunks)
    if len(paper_ids) < 2:
        return []

    dataset_papers: dict[str, list[str]] = defaultdict(list)
    for paper_sid in paper_ids:
        cached = _cache_get(conn, paper_sid, "dataset_coupling") if conn else None
        if cached is not None:
            datasets_for_paper = cached
        else:
            datasets_for_paper = _extract_datasets_from_chunks(paper_sid, chunks)
            if conn:
                _cache_put(conn, paper_sid, "dataset_coupling", datasets_for_paper)
        for ds in datasets_for_paper:
            if paper_sid not in dataset_papers[ds]:
                dataset_papers[ds].append(paper_sid)

    edges = []
    for ds, sids in dataset_papers.items():
        unique = list(dict.fromkeys(sids))
        for i, a in enumerate(unique):
            for b in unique[i + 1:]:
                edges.append(WikiEdge(a, b, 0.8, f"dataset:{ds}"))
    return edges
