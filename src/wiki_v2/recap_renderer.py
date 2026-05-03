"""Recap renderer — Recap dataclass → markdown.

Output convention: `wiki/<workspace>/recap-<issue_id>.md`. Goal/Outcome blocks
get an explicit "(manuell ergänzen)" placeholder when no chunks exist on
those axes — that's the User's prompt to fill them in.
"""

from __future__ import annotations

from src.wiki_v2.recap_indexer import Recap, RecapChunk


def _format_chunk(c: RecapChunk) -> str:
    cats = ", ".join(c.category_labels) if c.category_labels else "—"
    preview = c.text_preview.replace("\n", " ").strip()
    return (
        f"- `{c.chunk_id[:16]}` [{cats}] (conf {c.igio_confidence:.2f})\n"
        f"  > {preview[:140]}"
    )


def _section(title: str, chunks: list[RecapChunk], placeholder: str = "") -> str:
    if not chunks:
        body = f"_{placeholder}_" if placeholder else "_(noch keine Daten)_"
        return f"## {title}\n\n{body}\n"
    body = "\n".join(_format_chunk(c) for c in chunks)
    return f"## {title}\n\n{body}\n"


def render_recap(recap: Recap) -> str:
    """Render a Recap to markdown."""
    parts: list[str] = []
    title = f"# Recap — Issue #{recap.issue_id}"
    if recap.workspace_id:
        title += f" (workspace `{recap.workspace_id}`)"
    parts.append(title + "\n")

    parts.append(_section("Issue", recap.issue_chunks,
                          placeholder="Keine Chunks mit igio_axis='issue' für dieses Issue gefunden."))

    parts.append(_section("Goal", recap.goal_chunks,
                          placeholder="(manuell ergänzen)"))

    if recap.plans:
        plans_md = "\n".join(
            f"- [{p.title}]({p.path}) — {p.mtime_iso}" for p in recap.plans
        )
        parts.append(f"## Intervention — Plans\n\n{plans_md}\n")
    else:
        parts.append("## Intervention — Plans\n\n_(kein Plan verlinkt)_\n")

    parts.append(_section("Intervention — Chunks", recap.intervention_chunks,
                          placeholder="(noch keine intervention-Chunks)"))

    if recap.commits:
        commits_md = "\n".join(
            f"- `{c.sha}` {c.iso_date} — {c.subject}" for c in recap.commits
        )
        parts.append(f"## Outcome — Commits\n\n{commits_md}\n")
    else:
        parts.append("## Outcome — Commits\n\n_(keine zugeordneten Commits)_\n")

    parts.append(_section("Outcome — Chunks", recap.outcome_chunks,
                          placeholder="(manuell ergänzen — z.B. Test-Ergebnisse, Review-Findings)"))

    return "\n".join(parts)


__all__ = ("render_recap",)
