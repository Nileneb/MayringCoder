from __future__ import annotations
import json
import re
from src.ollama_client import generate as _ollama_gen
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.wiki_v2.graph import WikiGraph
    from src.wiki_v2.models import Cluster, WikiEdge


@dataclass
class ClusterVerdict:
    cluster_id: str
    action: str          # BESTÄTIGT | SPLIT_VORSCHLAG | MERGE_VORSCHLAG
    rationale: str = ""
    merge_target: str = ""
    split_groups: list[list[str]] = field(default_factory=list)


@dataclass
class EdgeVerdict:
    source: str
    target: str
    edge_type: str
    action: str          # BESTÄTIGT | ABGELEHNT | PRÄZISIERT
    rationale: str = ""
    corrected_type: str = ""


class WikiSecondOpinion:
    """Validates clusters and edges via a second LLM call (sync, Ollama-compatible)."""

    def validate_clusters(
        self,
        clusters: list["Cluster"],
        graph: "WikiGraph",
        validator_model: str,
        ollama_url: str,
    ) -> list[ClusterVerdict]:
        """Validate each cluster. Returns verdicts with BESTÄTIGT/SPLIT/MERGE."""
        verdicts = []
        for cluster in clusters:
            summaries = []
            for m in cluster.members[:15]:
                node = graph.get_node(m)
                s = node.summary if node and node.summary else "(keine Summary)"
                summaries.append(f"- {m}: {s}")

            prompt = (
                f'Cluster "{cluster.name}" ({len(cluster.members)} Dateien):\n'
                + "\n".join(summaries)
                + f"\n\nBeschreibung: {cluster.description or '(keine)'}\n\n"
                "Prüfe: Gehören ALLE Dateien logisch zusammen (eine Verantwortlichkeit)?\n"
                "Antworte mit EINEM der folgenden Wörter am Anfang: "
                "BESTÄTIGT, SPLIT_VORSCHLAG oder MERGE_VORSCHLAG — dann eine kurze Begründung.\n"
                "Nur Text, kein JSON."
            )

            raw = self._call_ollama(validator_model, ollama_url, prompt)
            action, rationale = self._parse_verdict(raw, ["BESTÄTIGT", "SPLIT_VORSCHLAG", "MERGE_VORSCHLAG"])
            verdicts.append(ClusterVerdict(
                cluster_id=cluster.cluster_id,
                action=action,
                rationale=rationale,
            ))
        return verdicts

    def validate_edges(
        self,
        edges: list["WikiEdge"],
        graph: "WikiGraph",
        validator_model: str,
        ollama_url: str,
        edge_types: list[str] | None = None,
    ) -> list[EdgeVerdict]:
        """Validate uncertain edges (default: concept_link, data_flow).
        Sets edge.validated = True in graph when confirmed.
        """
        if edge_types is None:
            edge_types = ["concept_link", "data_flow"]
        target_edges = [e for e in edges if e.type in edge_types]
        verdicts = []

        for edge in target_edges:
            src_node = graph.get_node(edge.source)
            tgt_node = graph.get_node(edge.target)
            src_sum = src_node.summary if src_node and src_node.summary else "(keine Summary)"
            tgt_sum = tgt_node.summary if tgt_node and tgt_node.summary else "(keine Summary)"

            prompt = (
                f"Besteht eine echte Beziehung zwischen diesen Dateien?\n\n"
                f"**{edge.source}**: {src_sum}\n"
                f"**{edge.target}**: {tgt_sum}\n"
                f"**Vermuteter Typ:** {edge.type}\n"
                f"**Kontext:** {edge.context or '(automatisch erkannt)'}\n\n"
                "Antworte mit EINEM der folgenden Wörter am Anfang: "
                "BESTÄTIGT, ABGELEHNT oder PRÄZISIERT — dann eine kurze Begründung.\n"
                "Falls PRÄZISIERT: Nenne den korrekten Edge-Typ.\nNur Text, kein JSON."
            )

            raw = self._call_ollama(validator_model, ollama_url, prompt)
            action, rationale = self._parse_verdict(raw, ["BESTÄTIGT", "ABGELEHNT", "PRÄZISIERT"])

            corrected = ""
            if action == "PRÄZISIERT":
                m = re.search(r'\b(import|call|test_covers|concept_link|issue_mentions|'
                              r'event_dispatch|label_cooccurrence|shared_type)\b', rationale)
                corrected = m.group(1) if m else ""

            if action == "BESTÄTIGT":
                try:
                    graph._conn.execute(
                        "UPDATE wiki_edges SET validated=1 WHERE source=? AND target=? "
                        "AND type=? AND workspace_id=?",
                        (edge.source, edge.target, edge.type, graph.workspace_id),
                    )
                    graph._conn.commit()
                except Exception:
                    pass

            verdicts.append(EdgeVerdict(
                source=edge.source,
                target=edge.target,
                edge_type=edge.type,
                action=action,
                rationale=rationale,
                corrected_type=corrected,
            ))
        return verdicts

    def apply_cluster_verdicts(
        self,
        clusters: list["Cluster"],
        verdicts: list[ClusterVerdict],
        graph: "WikiGraph",
    ) -> list["Cluster"]:
        """Apply SPLIT_VORSCHLAG verdicts by re-clustering the affected cluster's members.
        Returns the updated cluster list.
        """
        from src.wiki_v2.clustering import ClusterEngine
        result = []
        verdict_map = {v.cluster_id: v for v in verdicts}
        for cluster in clusters:
            verdict = verdict_map.get(cluster.cluster_id)
            if verdict and verdict.action == "SPLIT_VORSCHLAG" and len(cluster.members) >= 4:
                # Split: re-cluster this subset with Louvain
                try:
                    from src.wiki_v2.graph import WikiGraph as WG
                    from src.wiki_v2.models import WikiNode, WikiEdge
                    import tempfile, pathlib
                    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
                        sub_db = WG(graph.workspace_id, graph.repo_slug, pathlib.Path(tf.name))
                    for m in cluster.members:
                        n = graph.get_node(m)
                        sub_db.upsert_node(n or WikiNode(m, graph.repo_slug, graph.workspace_id))
                    for e in graph.get_edges():
                        if e.source in cluster.members and e.target in cluster.members:
                            sub_db.add_edge(e)
                    engine = ClusterEngine()
                    sub_clusters = engine.cluster(sub_db, strategy="louvain")
                    sub_db.close()
                    if len(sub_clusters) > 1:
                        result.extend(sub_clusters)
                        continue
                except Exception:
                    pass
            result.append(cluster)
        return result

    def second_opinion_report(self, cluster_verdicts: list[ClusterVerdict],
                               edge_verdicts: list[EdgeVerdict]) -> str:
        def _count(lst, field, val):
            return sum(1 for v in lst if getattr(v, field) == val)

        c_confirmed = _count(cluster_verdicts, "action", "BESTÄTIGT")
        c_split = _count(cluster_verdicts, "action", "SPLIT_VORSCHLAG")
        c_merge = _count(cluster_verdicts, "action", "MERGE_VORSCHLAG")

        e_confirmed = _count(edge_verdicts, "action", "BESTÄTIGT")
        e_rejected = _count(edge_verdicts, "action", "ABGELEHNT")
        e_refined = _count(edge_verdicts, "action", "PRÄZISIERT")

        lines = ["Second-Opinion Report:"]
        if cluster_verdicts:
            lines += [
                f"  Cluster — Bestätigt: {c_confirmed} | Split: {c_split} | Merge: {c_merge}",
            ]
        if edge_verdicts:
            lines += [
                f"  Edges   — Bestätigt: {e_confirmed} | Abgelehnt: {e_rejected} | Präzisiert: {e_refined}",
            ]
        return "\n".join(lines)

    def _call_ollama(self, model: str, ollama_url: str, prompt: str, timeout: int = 60) -> str:
        try:
            return _ollama_gen(
                ollama_url, model, prompt,
                stream=False,
                options={"temperature": 0.1},
                num_predict=300,
                timeout=float(timeout),
            )
        except Exception:
            return ""

    def _parse_verdict(self, text: str, keywords: list[str]) -> tuple[str, str]:
        text = text.strip()
        for kw in keywords:
            if kw in text.upper():
                rest = re.sub(re.escape(kw), "", text, count=1, flags=re.IGNORECASE).strip(" :—-\n")
                return kw, rest
        return keywords[0], text
