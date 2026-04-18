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
        for i, a in enumerate(unique[:10]):
            for b in unique[i + 1:10]:
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
            for m in _JOB_CLASS_RE.finditer(fname):
                listeners[m.group(1)].append(fname)
    edges = []
    for cls, dispatch_files in dispatchers.items():
        for listen_file in listeners.get(cls, []):
            for df in dispatch_files:
                if df != listen_file:
                    edges.append(WikiEdge(df, listen_file, 0.7, "event_dispatch"))
    return edges
