from __future__ import annotations
import copy
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.wiki_v2.models import WikiEdge


_TYPE_RE = re.compile(r'\b([A-Z][A-Za-z0-9]+(?:Service|Repository|Model|Manager|Handler|Controller|Helper))\b')
_DISPATCH_RE = re.compile(
    r'dispatch\s*\(|ShouldQueue|ProcessesJobs|implements\s+ShouldQueue|event\(|Event::dispatch'
)
_JOB_CLASS_RE = re.compile(r'\b([A-Z][A-Za-z0-9]+(?:Job|Event|Listener|Notification))\b')


def _build_class_index(overview_cache: dict) -> dict[str, str]:
    index = {}
    for fname in overview_cache:
        stem = Path(fname).stem.lower()
        index[stem] = fname
    return index


def _resolve_dep(dep_string: str, class_index: dict[str, str]) -> str | None:
    last = dep_string.split("\\")[-1].split("::")[0].lower()
    if last in class_index:
        return class_index[last]
    # Python convention: MemoryStore → store.py (suffix fallback)
    candidates = [(s, f) for s, f in class_index.items() if len(s) >= 3 and last.endswith(s)]
    return max(candidates, key=lambda x: len(x[0]))[1] if candidates else None


def _fn_field(fn: Any, key: str) -> list:
    if isinstance(fn, dict):
        v = fn.get(key, [])
        return v if isinstance(v, list) else []
    return []


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


class EdgeDetector:
    """Erkennt typisierte Edges zwischen Dateien für den Wiki-Graph."""

    def detect_from_overview(
        self,
        overview_cache: dict,
        conn: Any,
        workspace_id: str,
        repo_slug: str,
    ) -> list[WikiEdge]:
        """Runs import + call + shared_type + label_cooccurrence + event rules.
        Kein Ollama nötig. overview_cache: {filename → {dependencies, functions, file_summary}}"""
        edges: list[WikiEdge] = []
        edges += self._detect_imports(overview_cache, workspace_id, repo_slug)
        edges += self._detect_calls(overview_cache, workspace_id, repo_slug)
        edges += self._detect_shared_types(overview_cache, workspace_id, repo_slug)
        if conn is not None:
            edges += self._detect_label_cooccurrence(conn, overview_cache, workspace_id, repo_slug)
        edges += self._detect_event_dispatch(overview_cache, workspace_id, repo_slug)
        return self._deduplicate(edges)

    def detect_test_coverage(
        self,
        file_paths: list[str],
        workspace_id: str,
        repo_slug: str,
    ) -> list[WikiEdge]:
        """Heuristik: test_X.py → X.py über Namens-Match."""
        index = {Path(f).stem.lower(): f for f in file_paths}
        edges = []
        for fpath in file_paths:
            stem = Path(fpath).stem.lower()
            if stem.startswith("test_"):
                target_stem = stem[5:]
                target = index.get(target_stem)
                if target and target != fpath:
                    edges.append(WikiEdge(
                        source=fpath, target=target,
                        repo_slug=repo_slug, workspace_id=workspace_id,
                        type="test_covers", weight=1.0,
                        context="name_heuristic",
                    ))
        return edges

    def detect_concept_links(
        self,
        graph: Any,
        chroma: Any,
        threshold: float = 0.6,
    ) -> list[WikiEdge]:
        """Cosine-Similarity auf bestehende ChromaDB-Embeddings. Kein Re-Embed.
        Erwartet graph.workspace_id und graph.repo_slug."""
        if chroma is None:
            return []
        workspace_id = graph.workspace_id
        repo_slug = graph.repo_slug
        nodes = graph.all_nodes()
        if len(nodes) < 2:
            return []

        node_ids = [n.id for n in nodes]
        try:
            result = chroma.get(
                where={"workspace_id": {"$eq": workspace_id}},
                include=["embeddings", "metadatas"],
            )
            embeddings = result.get("embeddings") or []
            metadatas = result.get("metadatas") or []
        except Exception:
            return []

        emb_map: dict[str, list[float]] = {}
        for emb, meta in zip(embeddings, metadatas):
            if emb and meta:
                source_id = meta.get("source_id", "")
                path = source_id.split(":")[-1] if ":" in source_id else source_id
                for nid in node_ids:
                    if path.endswith(nid) or nid.endswith(path):
                        if nid not in emb_map:
                            emb_map[nid] = list(emb)

        edges = []
        nids = list(emb_map.keys())
        for i, a in enumerate(nids):
            for b in nids[i + 1:]:
                score = _cosine(emb_map[a], emb_map[b])
                if score >= threshold:
                    edges.append(WikiEdge(
                        source=a, target=b,
                        repo_slug=repo_slug, workspace_id=workspace_id,
                        type="concept_link", weight=round(score, 3),
                        context=f"cosine={score:.3f}",
                    ))
        return edges

    # --- private rule methods ---

    def _detect_imports(self, oc: dict, wid: str, slug: str) -> list[WikiEdge]:
        index = _build_class_index(oc)
        edges = []
        for fname, entry in oc.items():
            for dep in entry.get("dependencies", []):
                target = _resolve_dep(dep, index)
                if target and target != fname:
                    edges.append(WikiEdge(fname, target, slug, wid, "import", 1.0, dep))
        return edges

    def _detect_calls(self, oc: dict, wid: str, slug: str) -> list[WikiEdge]:
        index = _build_class_index(oc)
        edges = []
        for fname, entry in oc.items():
            for fn in entry.get("functions", []):
                calls = _fn_field(fn, "calls")
                fn_name = fn.get("name", "") if isinstance(fn, dict) else str(fn)
                for call in calls:
                    target = _resolve_dep(call, index)
                    if target and target != fname:
                        edges.append(WikiEdge(fname, target, slug, wid, "call", 0.9, fn_name))
        return edges

    def _detect_shared_types(self, oc: dict, wid: str, slug: str) -> list[WikiEdge]:
        type_files: dict[str, list[str]] = defaultdict(list)
        for fname, entry in oc.items():
            text = entry.get("file_summary", "")
            for fn in entry.get("functions", []):
                text += " ".join(str(x) for x in _fn_field(fn, "inputs") if x is not None)
                text += " ".join(str(x) for x in _fn_field(fn, "outputs") if x is not None)
                if isinstance(fn, str):
                    text += " " + fn
            for m in _TYPE_RE.finditer(text):
                type_files[m.group(1)].append(fname)
        edges = []
        for type_name, files in type_files.items():
            unique = list(dict.fromkeys(files))
            for i, a in enumerate(unique):
                for b in unique[i + 1:]:
                    edges.append(WikiEdge(a, b, slug, wid, "shared_type", 0.8, type_name))
        return edges

    def _detect_label_cooccurrence(self, conn: Any, oc: dict, wid: str, slug: str) -> list[WikiEdge]:
        try:
            rows = conn.execute(
                "SELECT source_id, category_labels FROM chunks WHERE is_active=1 AND category_labels != '' AND workspace_id=?",
                (wid,)
            ).fetchall()
        except Exception:
            try:
                rows = conn.execute(
                    "SELECT source_id, category_labels FROM chunks WHERE is_active=1 AND category_labels != ''"
                ).fetchall()
            except Exception:
                return []
        matrix: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            source_id = row[0] if isinstance(row, (list, tuple)) else row["source_id"]
            labels_str = row[1] if isinstance(row, (list, tuple)) else row["category_labels"]
            path = source_id.split(":")[-1] if ":" in source_id else source_id
            for label in labels_str.split(","):
                label = label.strip()
                if label:
                    matrix[label].append(path)
        edges = []
        for label, source_ids in matrix.items():
            unique = list(dict.fromkeys(source_ids))[:10]
            for i, a in enumerate(unique):
                for b in unique[i + 1:]:
                    edges.append(WikiEdge(a, b, slug, wid, "label_cooccurrence", 0.5, label))
        return edges

    def _detect_event_dispatch(self, oc: dict, wid: str, slug: str) -> list[WikiEdge]:
        dispatchers: dict[str, list[str]] = defaultdict(list)
        listeners: dict[str, list[str]] = defaultdict(list)
        for fname, entry in oc.items():
            text = entry.get("file_summary", "")
            if _DISPATCH_RE.search(text):
                for m in _JOB_CLASS_RE.finditer(text):
                    dispatchers[m.group(1)].append(fname)
            if "ShouldQueue" in text:
                for m in _JOB_CLASS_RE.finditer(text):
                    listeners[m.group(1)].append(fname)
        edges = []
        for cls, dispatch_files in dispatchers.items():
            for listen_file in listeners.get(cls, []):
                for df in dispatch_files:
                    if df != listen_file:
                        edges.append(WikiEdge(df, listen_file, slug, wid, "event_dispatch", 0.7, cls))
        return edges

    def _deduplicate(self, edges: list[WikiEdge]) -> list[WikiEdge]:
        """Merge duplicate (source, target, type) edges by summing weight."""
        merged: dict[tuple, WikiEdge] = {}
        for e in edges:
            key = (e.source, e.target, e.type, e.workspace_id)
            if key in merged:
                merged[key].weight = round(merged[key].weight + e.weight, 3)
                if e.context and e.context not in merged[key].context:
                    merged[key].context += f",{e.context}"
            else:
                merged[key] = copy.copy(e)
        return list(merged.values())
