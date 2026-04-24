"""Kern-Datenstrukturen, Code-Regeln, Graph-Building, Clustering, Markdown."""
from __future__ import annotations
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


def _fn_field(fn, key: str) -> list:
    """Tolerant reader für overview_cache-Einträge.

    Ältere Caches speichern `functions` als liste von String-Namen, neuere
    als dicts mit `inputs`/`outputs`/`calls`. Beide Formen unterstützen,
    damit ein cachekompatibler Wiki-Build nicht an Formatdrift scheitert.
    """
    if isinstance(fn, dict):
        v = fn.get(key, [])
        return v if isinstance(v, list) else []
    return []


def find_shared_types(overview_cache: dict[str, dict]) -> list[WikiEdge]:
    """Rule 2: Files sharing same class types in signatures → weight 0.8."""
    type_files: dict[str, list[str]] = defaultdict(list)
    for fname, entry in overview_cache.items():
        text = entry.get("file_summary", "")
        for fn in entry.get("functions", []):
            text += " ".join(str(x) for x in _fn_field(fn, "inputs") if x is not None) + " " + " ".join(str(x) for x in _fn_field(fn, "outputs") if x is not None)
            if isinstance(fn, str):
                text += " " + fn
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
            for call in _fn_field(fn, "calls"):
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


from src.memory.wiki_paper import (  # noqa: E402 — must come after WikiEdge is defined
    find_citation_pairs,
    find_keyword_overlap,
    find_shared_concepts,
    find_method_chains,
    find_dataset_pairs,
)

RULE_SETS: dict[str, list[tuple[str, Any, float]]] = {
    "code": [
        ("import",             lambda oc, c, conn, ch: find_import_pairs(oc),        1.0),
        ("shared_type",        lambda oc, c, conn, ch: find_shared_types(oc),         0.8),
        ("function_call",      lambda oc, c, conn, ch: find_call_pairs(oc),           0.9),
        ("label_cooccurrence", lambda oc, c, conn, ch: find_label_overlap(conn, oc),  0.5),
        ("event_dispatch",     lambda oc, c, conn, ch: find_event_pairs(oc),          0.7),
    ],
    "paper": [
        ("citation",          lambda oc, c, conn, ch: find_citation_pairs(oc, c),                1.0),
        ("keyword_overlap",   lambda oc, c, conn, ch: find_keyword_overlap(oc, c),               0.5),
        ("shared_concept",    lambda oc, c, conn, ch: find_shared_concepts(c, conn, ch, "", ""), 0.8),
        ("method_chain",      lambda oc, c, conn, ch: find_method_chains(c, conn, ch, "", ""),   0.7),
        ("dataset_coupling",  lambda oc, c, conn, ch: find_dataset_pairs(c, conn, ch, "", ""),   0.8),
    ],
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
