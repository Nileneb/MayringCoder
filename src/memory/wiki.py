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
