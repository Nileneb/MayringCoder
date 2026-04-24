from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class WikiNode:
    id: str
    repo_slug: str
    workspace_id: str
    type: str = "file"
    cluster_id: str = ""
    labels: list[str] = field(default_factory=list)
    summary: str = ""
    turbulence_tier: str = ""
    loc: int = 0


@dataclass
class WikiEdge:
    source: str
    target: str
    repo_slug: str
    workspace_id: str
    type: str
    weight: float = 1.0
    context: str = ""
    validated: bool = False


@dataclass
class Cluster:
    cluster_id: str
    repo_slug: str
    workspace_id: str
    name: str
    description: str = ""
    rationale: str = ""
    strategy_used: str = "louvain"
    members: list[str] = field(default_factory=list)
