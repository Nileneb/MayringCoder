#!/usr/bin/env python3
"""Generate Markdown knowledge graphs from the MCP memory store.

For each project (repo), one file is written to cache/knowledge_graph/<slug>.md.
The graph contains:
  - Kategorien: all discovered category_labels with their chunk sources
  - Quellen-Cluster: chunks grouped by source_type and file
  - Verbindungen: resolved related_chunk_ids from conversation_summary chunks

Called automatically at the end of ingestion runs. Also usable standalone:

    .venv/bin/python tools/generate_knowledge_graph.py
    .venv/bin/python tools/generate_knowledge_graph.py --project MayringCoder
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_GRAPH_DIR = Path("cache/knowledge_graph")
_RELATED_RE = re.compile(r"<!-- related_chunk_ids: (\[.*?\]) -->", re.DOTALL)

# Codebook per source_type (mirrors _INGEST_DEFAULTS in memory_ingest.py)
_SOURCE_TYPE_CODEBOOK = {
    "repo_file": "code",
    "note": "code",
    "github_issue": "social",
    "conversation_summary": "social",
    "session_knowledge": "social",
    "session_note": "social",
    "image": "code",
}

_CODEBOOKS_DIR = Path(__file__).parent.parent / "codebooks"


def _load_codebook_descriptions() -> dict[str, dict[str, str]]:
    """Return {codebook_name: {label: description}} from codebooks/*.yaml."""
    try:
        import yaml
    except ImportError:
        return {}
    result: dict[str, dict[str, str]] = {}
    for cb_file in _CODEBOOKS_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(cb_file.read_text(encoding="utf-8"))
            cats = data.get("categories", []) if isinstance(data, dict) else []
            descs: dict[str, str] = {}
            for c in cats:
                if isinstance(c, dict):
                    label = (c.get("label") or c.get("name", "")).lower()
                    desc = c.get("description", "") or c.get("desc", "")
                    if label:
                        descs[label] = desc
            result[cb_file.stem] = descs
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_chunks(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT c.chunk_id, c.source_id, c.category_labels, c.text,
               c.chunk_level, s.source_type, s.repo, s.path, s.captured_at
        FROM chunks c
        JOIN sources s ON c.source_id = s.source_id
        WHERE c.is_active = 1
        ORDER BY s.repo, s.source_type, c.source_id
    """).fetchall()
    return [
        {
            "chunk_id": r[0],
            "source_id": r[1],
            "category_labels": [l for l in r[2].split(",") if l] if r[2] else [],
            "text": r[3] or "",
            "chunk_level": r[4] or "",
            "source_type": r[5] or "",
            "repo": r[6] or "",
            "path": r[7] or "",
            "captured_at": r[8] or "",
        }
        for r in rows
    ]


def _parse_related_ids(text: str) -> list[str]:
    m = _RELATED_RE.search(text)
    if not m:
        return []
    try:
        return json.loads(m.group(1))
    except Exception:
        return []


def _chunk_id_to_source(chunks: list[dict]) -> dict[str, str]:
    return {c["chunk_id"]: c["source_id"] for c in chunks}


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def _repo_slug(repo: str) -> str:
    """'https://github.com/Nileneb/MayringCoder' → 'MayringCoder'"""
    repo = repo.rstrip("/")
    return repo.rsplit("/", 1)[-1] if repo else "unknown"


def _source_label(source_id: str, path: str) -> str:
    """Short human-readable label for a source."""
    if path:
        return path
    parts = source_id.split(":", 2)
    return parts[-1] if parts else source_id


def _session_date(text: str) -> str:
    m = re.search(r"Session (\d{4}-\d{2}-\d{2})", text)
    return m.group(1) if m else "?"


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def _build_project_graphs(chunks: list[dict]) -> dict[str, dict]:
    """Group chunks into per-project structures."""
    id_to_source = _chunk_id_to_source(chunks)

    # Normalise repo → project slug
    def _project(chunk: dict) -> str:
        if chunk["repo"]:
            return _repo_slug(chunk["repo"])
        # conversation:Slug:session_id → extract Slug
        sid = chunk["source_id"]
        if sid.startswith("conversation:"):
            parts = sid.split(":", 2)
            if len(parts) >= 2:
                return parts[1]
        return "misc"

    projects: dict[str, dict] = {}

    for c in chunks:
        proj = _project(c)
        if proj not in projects:
            projects[proj] = {
                "categories": defaultdict(list),     # label → [chunk]
                "by_type": defaultdict(list),         # source_type → [chunk]
                "connections": [],                    # (conv_chunk, [related_source_ids])
                "chunk_count": 0,
                "categorized_count": 0,
            }
        p = projects[proj]
        p["chunk_count"] += 1
        if c["category_labels"]:
            p["categorized_count"] += 1

        for label in c["category_labels"]:
            p["categories"][label].append(c)

        p["by_type"][c["source_type"]].append(c)

        if c["source_type"] == "conversation_summary":
            related_ids = _parse_related_ids(c["text"])
            if related_ids:
                related_sources = list({
                    id_to_source[rid] for rid in related_ids if rid in id_to_source
                })
                if related_sources:
                    p["connections"].append((c, related_sources))

    return projects


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_project(proj_name: str, data: dict, ts: str, cb_descs: dict | None = None) -> str:
    lines: list[str] = []
    total = data["chunk_count"]
    categorized = data.get("categorized_count", 0)
    cat_rate = f"{100 * categorized / total:.0f}%" if total else "0%"
    by_type = data["by_type"]

    type_summary = ", ".join(
        f"{len(v)} {k}" for k, v in sorted(by_type.items(), key=lambda x: -len(x[1]))
    )
    lines.append(f"# Knowledge Graph — {proj_name}")
    lines.append(f"*Stand: {ts} | {total} Chunks: {type_summary} | Kategorisiert: {cat_rate}*")
    lines.append("")

    # ── Kategorien ───────────────────────────────────────────────────────────
    lines.append("## Kategorien")
    cats = data["categories"]
    if not cats:
        lines.append("")
        lines.append(
            "> Noch keine Kategorien — Ingestion mit `categorize=True` ausführen "
            "(z.B. `ingest_conversations.py` neu starten oder `checker.py --populate-memory --memory-categorize`)."
        )
    else:
        # Build a merged description lookup across all codebooks
        all_descs: dict[str, str] = {}
        if cb_descs:
            for _book_descs in cb_descs.values():
                all_descs.update(_book_descs)

        for label, cat_chunks in sorted(cats.items(), key=lambda x: -len(x[1])):
            lines.append("")
            desc = all_descs.get(label.lower(), "")
            desc_str = f" — *{desc}*" if desc else ""
            lines.append(f"### {label} ({len(cat_chunks)} Chunks){desc_str}")
            seen_sources: set[str] = set()
            for c in cat_chunks:
                src = _source_label(c["source_id"], c["path"])
                if src not in seen_sources:
                    seen_sources.add(src)
                    lvl = f" `{c['chunk_level']}`" if c["chunk_level"] not in ("file", "") else ""
                    lines.append(f"- `{src}`{lvl}")
    lines.append("")

    # ── Quellen-Cluster ───────────────────────────────────────────────────────
    lines.append("## Quellen-Cluster")

    type_order = ["repo_file", "github_issue", "conversation_summary", "note",
                  "session_knowledge", "session_note", "image"]
    type_labels = {
        "repo_file": "Source-Dateien",
        "github_issue": "GitHub Issues",
        "conversation_summary": "Gesprächs-Sessions",
        "note": "Notes",
        "session_knowledge": "Session-Wissen",
        "session_note": "Session-Notizen",
        "image": "Bilder",
    }

    for stype in type_order:
        if stype not in by_type:
            continue
        type_chunks = by_type[stype]
        cb_name = _SOURCE_TYPE_CODEBOOK.get(stype, "code")
        label = type_labels.get(stype, stype)
        lines.append("")
        lines.append(f"### {label} ({len(type_chunks)}) · Codebook: {cb_name}")

        if stype == "repo_file":
            # Group by file path
            by_file: dict[str, list[dict]] = defaultdict(list)
            for c in type_chunks:
                by_file[c["path"] or c["source_id"]].append(c)
            for fpath, fchunks in sorted(by_file.items()):
                cat_str = ""
                all_labels = {l for c in fchunks for l in c["category_labels"]}
                if all_labels:
                    cat_str = f" → *{', '.join(sorted(all_labels))}*"
                lines.append(f"- `{fpath}` ({len(fchunks)} Chunks){cat_str}")

        elif stype == "github_issue":
            for c in type_chunks[:30]:
                # Extract issue number + title from source_id
                sid = c["source_id"]
                m = re.search(r"issue/(\d+)", sid)
                num = f"#{m.group(1)}" if m else ""
                title = c["text"][:80].replace("\n", " ").strip() if c["text"] else ""
                cat_str = ""
                if c["category_labels"]:
                    cat_str = f" *[{', '.join(c['category_labels'])}]*"
                lines.append(f"- {num} {title[:60]}…{cat_str}")
            if len(type_chunks) > 30:
                lines.append(f"- *… {len(type_chunks)-30} weitere*")

        elif stype == "conversation_summary":
            for c in type_chunks:
                date = _session_date(c["text"])
                turns_m = re.search(r"(\d+) turns", c["text"])
                turns = turns_m.group(0) if turns_m else ""
                thema_m = re.search(r"\*\*Thema:\*\*\s*(.+)", c["text"])
                thema = thema_m.group(1).strip()[:80] if thema_m else c["text"][:60]
                cat_str = ""
                if c["category_labels"]:
                    cat_str = f" *[{', '.join(c['category_labels'])}]*"
                lines.append(f"- {date} | {turns} | {thema}{cat_str}")

        else:
            for c in type_chunks[:20]:
                src = _source_label(c["source_id"], c["path"])
                lines.append(f"- `{src}`")

    lines.append("")

    # ── Verbindungen ─────────────────────────────────────────────────────────
    lines.append("## Verbindungen")
    connections = data["connections"]
    if not connections:
        lines.append("")
        lines.append("> Noch keine aufgelösten Verbindungen.")
    else:
        for conv_chunk, related_sources in connections:
            date = _session_date(conv_chunk["text"])
            thema_m = re.search(r"\*\*Thema:\*\*\s*(.+)", conv_chunk["text"])
            thema = thema_m.group(1).strip()[:60] if thema_m else "?"
            lines.append("")
            lines.append(f"### Session {date} — {thema}")
            for rsrc in sorted(related_sources)[:8]:
                # Shorten source_id for readability
                parts = rsrc.split(":", 3)
                short = parts[-1] if len(parts) >= 3 else rsrc
                lines.append(f"  ↔ `{short[:80]}`")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate(project_filter: str | None = None) -> dict[str, Path]:
    """Generate knowledge graph Markdown files. Returns {project: path}."""
    from dotenv import load_dotenv
    load_dotenv()

    from src.memory_store import init_memory_db
    conn = init_memory_db()

    chunks = _load_chunks(conn)
    conn.close()

    if not chunks:
        print("Keine Chunks in der Memory-DB.", file=sys.stderr)
        return {}

    projects = _build_project_graphs(chunks)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cb_descs = _load_codebook_descriptions()

    _GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for proj_name, data in sorted(projects.items()):
        if project_filter and project_filter.lower() not in proj_name.lower():
            continue
        md = _render_project(proj_name, data, ts, cb_descs=cb_descs)
        out = _GRAPH_DIR / f"{proj_name}.md"
        out.write_text(md, encoding="utf-8")
        n_cats = len(data["categories"])
        n_conn = len(data["connections"])
        categorized = data.get("categorized_count", 0)
        total = data["chunk_count"]
        cat_pct = f"{100 * categorized / total:.0f}%" if total else "0%"
        print(f"  ✓ {out}  ({total} Chunks, {n_cats} Kategorien, {n_conn} Verbindungen, {cat_pct} kategorisiert)")
        written[proj_name] = out

    return written


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Knowledge-Graph Markdown aus MCP Memory generieren")
    ap.add_argument("--project", metavar="NAME", help="Nur dieses Projekt generieren")
    args = ap.parse_args()
    generate(project_filter=args.project)
