"""Verknüpfungswiki — funktionale Zusammenhänge via Label-Co-Occurrence und Dependency-Analyse."""
from __future__ import annotations
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WikiEdge:
    file_a: str
    file_b: str
    weight: float
    rule: str


@dataclass
class WikiCluster:
    name: str
    files: list[str]
    labels: list[str]
    edges: list[tuple[str, float, list[str]]]  # (target_cluster, total_weight, rules)


def build_category_matrix(conn: Any) -> dict[str, list[str]]:
    """Label → [source_ids] Mapping aus aktiven Chunks."""
    rows = conn.execute(
        "SELECT source_id, category_labels FROM chunks WHERE is_active=1 AND category_labels != ''"
    ).fetchall()
    matrix: dict[str, list[str]] = defaultdict(list)
    for source_id, labels_str in rows:
        for label in labels_str.split(","):
            label = label.strip()
            if label:
                matrix[label].append(source_id)
    return dict(matrix)


# Regex constants for rule extractors
_TYPE_RE = re.compile(r'\b([A-Z][A-Za-z0-9]+(?:Service|Repository|Model|Manager|Handler|Controller|Helper))\b')
_DISPATCH_RE = re.compile(
    r'dispatch\s*\(|ShouldQueue|ProcessesJobs|implements\s+ShouldQueue|event\(|Event::dispatch'
)
_JOB_CLASS_RE = re.compile(r'\b([A-Z][A-Za-z0-9]+(?:Job|Event|Listener|Notification))\b')


def _build_class_index(overview_cache: dict[str, dict]) -> dict[str, str]:
    """lowercase stem → filename index for dependency resolution."""
    index = {}
    for fname in overview_cache:
        stem = Path(fname).stem.lower()
        index[stem] = fname
    return index


def _resolve_dep(dep_string: str, class_index: dict[str, str]) -> str | None:
    """Resolve dependency string like 'App\\Services\\CreditService' → filename."""
    last = dep_string.split("\\")[-1].split("::")[0].lower()
    return class_index.get(last)


def find_import_pairs(overview_cache: dict[str, dict]) -> list[WikiEdge]:
    """Rule 1: File A imports File B → weight 1.0."""
    index = _build_class_index(overview_cache)
    edges = []
    for fname, entry in overview_cache.items():
        for dep in entry.get("dependencies", []):
            target = _resolve_dep(dep, index)
            if target and target != fname:
                edges.append(WikiEdge(fname, target, 1.0, "import"))
    return edges


def find_shared_types(overview_cache: dict[str, dict]) -> list[WikiEdge]:
    """Rule 2: Files sharing same class types in signatures → weight 0.8."""
    type_files: dict[str, list[str]] = defaultdict(list)
    for fname, entry in overview_cache.items():
        text = entry.get("file_summary", "")
        for fn in entry.get("functions", []):
            text += " ".join(fn.get("inputs", [])) + " " + " ".join(fn.get("outputs", []))
        for m in _TYPE_RE.finditer(text):
            type_files[m.group(1)].append(fname)
    edges = []
    for files in type_files.values():
        unique = list(dict.fromkeys(files))
        for i, a in enumerate(unique):
            for b in unique[i + 1:]:
                edges.append(WikiEdge(a, b, 0.8, "shared_type"))
    return edges


def find_call_pairs(overview_cache: dict[str, dict]) -> list[WikiEdge]:
    """Rule 3: Function A calls Function B → weight 0.9."""
    index = _build_class_index(overview_cache)
    edges = []
    for fname, entry in overview_cache.items():
        for fn in entry.get("functions", []):
            for call in fn.get("calls", []):
                target = _resolve_dep(call, index)
                if target and target != fname:
                    edges.append(WikiEdge(fname, target, 0.9, "function_call"))
    return edges


def find_label_overlap(conn: Any, overview_cache: dict[str, dict]) -> list[WikiEdge]:
    """Rule 4: Files sharing Mayring category labels → weight 0.5."""
    matrix = build_category_matrix(conn)
    edges = []
    for label, source_ids in matrix.items():
        if len(source_ids) < 2:
            continue
        unique = list(dict.fromkeys(source_ids))
        capped = unique[:10]
        for i, a in enumerate(capped):
            for b in capped[i + 1:]:
                edges.append(WikiEdge(a, b, 0.5, "label_cooccurrence"))
    return edges


def find_event_pairs(overview_cache: dict[str, dict]) -> list[WikiEdge]:
    """Rule 5: File A dispatches Job/Event that File B listens to → weight 0.7."""
    dispatchers: dict[str, list[str]] = defaultdict(list)
    listeners: dict[str, list[str]] = defaultdict(list)
    for fname, entry in overview_cache.items():
        text = entry.get("file_summary", "")
        if _DISPATCH_RE.search(text):
            for m in _JOB_CLASS_RE.finditer(text):
                dispatchers[m.group(1)].append(fname)
        if "ShouldQueue" in text or "implements ShouldQueue" in text:
            for m in _JOB_CLASS_RE.finditer(text):
                listeners[m.group(1)].append(fname)
    edges = []
    for cls, dispatch_files in dispatchers.items():
        for listen_file in listeners.get(cls, []):
            for df in dispatch_files:
                if df != listen_file:
                    edges.append(WikiEdge(df, listen_file, 0.7, "event_dispatch"))
    return edges


RULE_SETS: dict[str, list[tuple[str, Any, float]]] = {
    "code": [
        ("import",             lambda oc, c, conn, ch: find_import_pairs(oc),        1.0),
        ("shared_type",        lambda oc, c, conn, ch: find_shared_types(oc),         0.8),
        ("function_call",      lambda oc, c, conn, ch: find_call_pairs(oc),           0.9),
        ("label_cooccurrence", lambda oc, c, conn, ch: find_label_overlap(conn, oc),  0.5),
        ("event_dispatch",     lambda oc, c, conn, ch: find_event_pairs(oc),          0.7),
    ],
    "paper": [],
}


def build_connection_graph(
    doc_type: str,
    overview_cache: dict[str, dict],
    chunks: list,
    conn: Any,
    chroma: Any = None,
) -> list[WikiEdge]:
    """Run all rules for doc_type, merge duplicate edges by summing weights."""
    edges: list[WikiEdge] = []
    for _rule_name, finder_fn, _weight in RULE_SETS.get(doc_type, []):
        try:
            edges.extend(finder_fn(overview_cache, chunks, conn, chroma))
        except NotImplementedError:
            pass

    merged: dict[tuple[str, str], WikiEdge] = {}
    for e in edges:
        key = (min(e.file_a, e.file_b), max(e.file_a, e.file_b))
        if key in merged:
            merged[key].weight = round(merged[key].weight + e.weight, 2)
            existing_rules = set(merged[key].rule.split(","))
            if e.rule not in existing_rules:
                merged[key].rule += f",{e.rule}"
        else:
            merged[key] = WikiEdge(key[0], key[1], e.weight, e.rule)
    return list(merged.values())


def cluster_themes(edges: list[WikiEdge], min_files: int = 2) -> list[WikiCluster]:
    """Group connected files into clusters via Union-Find."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges:
        union(e.file_a, e.file_b)

    groups: dict[str, list[str]] = defaultdict(list)
    for node in parent:
        groups[find(node)].append(node)

    clusters = []
    for _root, files in groups.items():
        if len(files) < min_files:
            continue
        cross: dict[str, tuple[float, list[str]]] = {}
        for e in edges:
            if e.file_a in files and e.file_b not in files:
                other_root = find(e.file_b)
                if other_root not in cross:
                    cross[other_root] = (0.0, [])
                w, rules = cross[other_root]
                rules_new = list(set(rules + e.rule.split(",")))
                cross[other_root] = (round(w + e.weight, 2), rules_new)
        cross_list = [(k, v[0], v[1]) for k, v in cross.items()]
        name = Path(files[0]).stem
        clusters.append(WikiCluster(name=name, files=sorted(files), labels=[], edges=cross_list))
    return clusters


def generate_wiki_markdown(clusters: list[WikiCluster], repo_slug: str) -> str:
    """Render clusters as Markdown wiki."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Verknüpfungswiki — {repo_slug}",
        f"_Stand: {ts} | {len(clusters)} Themen-Cluster_\n",
    ]
    for c in sorted(clusters, key=lambda x: len(x.files), reverse=True):
        lines.append(f"## 🔗 {c.name}")
        lines.append(f"**Dateien ({len(c.files)}):** {', '.join(Path(f).name for f in c.files)}")
        if c.labels:
            lines.append(f"**Labels:** {', '.join(c.labels)}")
        if c.edges:
            lines.append("**Querverweise:**")
            for target, weight, rules in sorted(c.edges, key=lambda x: -x[1])[:5]:
                lines.append(f"- → {target} (Gewicht {weight}: {', '.join(rules)})")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper rule stubs — implement when paper sources exist in memory
# ---------------------------------------------------------------------------

def find_citation_pairs(overview_cache: dict, chunks: list) -> list[WikiEdge]:
    raise NotImplementedError("Paper rules not yet implemented (no paper sources in memory)")


def find_shared_concepts(chunks: list, conn: Any, chroma: Any, ollama_url: str, model: str) -> list[WikiEdge]:
    raise NotImplementedError("Paper rules not yet implemented")


def find_method_chains(chunks: list, conn: Any, chroma: Any, ollama_url: str, model: str) -> list[WikiEdge]:
    raise NotImplementedError("Paper rules not yet implemented")


def find_keyword_overlap(overview_cache: dict, chunks: list) -> list[WikiEdge]:
    raise NotImplementedError("Paper rules not yet implemented")


def find_dataset_pairs(chunks: list, conn: Any, chroma: Any, ollama_url: str, model: str) -> list[WikiEdge]:
    raise NotImplementedError("Paper rules not yet implemented")


def _build_keyword_index(clusters: list[WikiCluster]) -> dict[str, list[str]]:
    """Keyword → [cluster_name, ...] Mapping."""
    index: dict[str, list[str]] = {}
    for c in clusters:
        keywords = set()
        keywords.add(c.name.lower())
        for f in c.files:
            keywords.add(Path(f).stem.lower())
        for label in c.labels:
            keywords.update(label.lower().split())
        for kw in keywords:
            if len(kw) > 2:
                index.setdefault(kw, [])
                if c.name not in index[kw]:
                    index[kw].append(c.name)
    return index


def _build_cluster_embeddings(
    clusters: list[WikiCluster],
    ollama_url: str,
) -> dict[str, list[float]]:
    """cluster_name → embedding vector. Skips silently if ollama_url empty."""
    if not ollama_url or not clusters:
        return {}
    from src.analysis.context import _embed_texts
    texts = [f"{c.name} {' '.join(c.labels)}" for c in clusters]
    try:
        vecs = _embed_texts(texts, ollama_url)
        return {c.name: vec for c, vec in zip(clusters, vecs)}
    except Exception:
        return {}


def generate_wiki(
    conn: Any,
    chroma: Any,
    repo_url: str,
    ollama_url: str = "",
    model: str = "",
    workspace_id: str = "default",
    doc_type: str = "code",
) -> Path | None:
    """Orchestrate wiki generation: load overview cache → build graph → cluster → write markdown."""
    from src.analysis.context import load_overview_cache_raw
    from src.config import repo_slug as _repo_slug

    slug = _repo_slug(repo_url)
    overview_cache = load_overview_cache_raw(repo_url) or {}
    if not overview_cache:
        print(f"[wiki] Kein Overview-Cache für {slug} — erst --mode overview ausführen")
        return None

    chunks: list = conn.execute(
        "SELECT source_id, category_labels FROM chunks WHERE is_active=1"
    ).fetchall()

    edges = build_connection_graph(doc_type, overview_cache, chunks, conn, chroma)
    clusters = cluster_themes(edges)

    out = Path("cache") / f"{slug}_wiki.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate_wiki_markdown(clusters, slug), encoding="utf-8")
    print(f"[wiki] {len(clusters)} Cluster → {out}")

    # Keyword-Index
    idx_path = Path("cache") / f"{slug}_wiki_index.json"
    idx_path.write_text(json.dumps(_build_keyword_index(clusters), ensure_ascii=False), encoding="utf-8")

    # Cluster-Embeddings (optional, skip if no ollama_url)
    emb_path = Path("cache") / f"{slug}_wiki_clusters_emb.json"
    emb = _build_cluster_embeddings(clusters, ollama_url)
    if emb:
        emb_path.write_text(json.dumps(emb, ensure_ascii=False), encoding="utf-8")

    print(f"[wiki] index → {idx_path}")
    return out
