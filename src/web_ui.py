"""Gradio WebUI for MayringCoder Memory Layer.

Usage:
    python -m src.web_ui [--port 7860] [--auth user:pass] [--ollama-url http://localhost:11434]

Tabs:
    1. Suche      — Hybrid memory search (symbolic + vector)
    2. Ingest     — Ingest text/file into memory
    3. Browser    — Browse all sources + chunk detail
    4. Feedback   — Label chunks with positive/negative/neutral signal
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Graceful imports — UI must start even if optional dependencies are missing
# ---------------------------------------------------------------------------

try:
    import gradio as gr
    _HAS_GRADIO = True
except ImportError as _e:
    raise SystemExit(
        "gradio is not installed. Run: pip install -r requirements-ui.txt"
    ) from _e

from src.ollama_status import check_ollama

# Memory modules — imported with graceful fallback
_MEMORY_READY = False
_IMPORT_ERROR: str = ""

try:
    from src.memory_store import (
        init_memory_db,
        add_feedback,
        get_chunks_by_source,
    )
    from src.memory_ingest import ingest, get_or_create_chroma_collection, ingest_conversation_summary
    from src.memory_retrieval import search
    from src.memory_schema import Source, Chunk
    from src.config import CACHE_DIR
    import hashlib
    from datetime import datetime, timezone

    _MEMORY_READY = True
except Exception as exc:
    _IMPORT_ERROR = str(exc)

# ---------------------------------------------------------------------------
# Module-level state (initialized lazily)
# ---------------------------------------------------------------------------

_conn: sqlite3.Connection | None = None
_chroma_collection: Any = None
_ollama_url: str = "http://localhost:11434"


def _get_conn() -> sqlite3.Connection | None:
    """Return SQLite connection, initializing on first call."""
    global _conn
    if _conn is not None:
        return _conn
    if not _MEMORY_READY:
        return None
    try:
        _conn = init_memory_db()
        return _conn
    except Exception:
        return None


def _get_chroma() -> Any:
    """Return ChromaDB collection, initializing on first call."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    if not _MEMORY_READY:
        return None
    try:
        _chroma_collection = get_or_create_chroma_collection()
        return _chroma_collection
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helper: Ollama status badge HTML
# ---------------------------------------------------------------------------

def _status_html(available: bool, models: list[str]) -> str:
    color = "#22c55e" if available else "#ef4444"
    label = "Ollama online" if available else "Ollama offline"
    model_hint = f" ({', '.join(models[:3])})" if models else ""
    return (
        f'<div style="display:inline-flex;align-items:center;gap:8px;'
        f'padding:4px 12px;border-radius:9999px;background:{color}20;'
        f'border:1px solid {color};font-size:0.85rem;">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{color};'
        f'display:inline-block;"></span>'
        f'<span style="color:{color};font-weight:600;">{label}{model_hint}</span>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Tab 1: Suche
# ---------------------------------------------------------------------------

def _do_search(query: str, top_k: int, ollama_available: bool) -> tuple[str, list[list]]:
    """Run hybrid memory search. Returns (status_msg, table_rows)."""
    if not _MEMORY_READY:
        return f"Memory-Module nicht geladen: {_IMPORT_ERROR}", []
    if not query.strip():
        return "Bitte Suchbegriff eingeben.", []

    conn = _get_conn()
    if conn is None:
        return "SQLite-Datenbank nicht verfugbar.", []

    chroma = _get_chroma() if ollama_available else None

    try:
        results = search(
            query=query,
            conn=conn,
            chroma_collection=chroma,
            ollama_url=_ollama_url,
            opts={"top_k": int(top_k)},
        )
    except Exception as exc:
        return f"Suchfehler: {exc}", []

    if not results:
        mode = "symbolisch" if not ollama_available else "hybrid"
        return f"Keine Ergebnisse ({mode} Suche).", []

    rows = []
    for r in results:
        cats = ", ".join(r.category_labels) if r.category_labels else "-"
        text_preview = r.text[:120].replace("\n", " ") + ("..." if len(r.text) > 120 else "")
        rows.append([
            r.chunk_id[:12],
            f"{r.score_final:.3f}",
            r.source_id,
            cats,
            text_preview,
            ", ".join(r.reasons),
        ])

    mode = "symbolisch" if not ollama_available else "hybrid"
    return f"{len(results)} Ergebnis(se) ({mode} Suche).", rows


# ---------------------------------------------------------------------------
# Tab 2: Ingest
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _do_ingest(
    text_input: str,
    file_upload,
    source_path: str,
    repo: str,
    categorize: bool,
    mode: str,
    codebook: str,
    model: str,
    ollama_available: bool,
) -> str:
    """Ingest text or file into memory. Returns JSON result."""
    if not _MEMORY_READY:
        return json.dumps({"error": f"Memory-Module nicht geladen: {_IMPORT_ERROR}"}, indent=2)

    if not ollama_available:
        warning = (
            "[Warnung] Ollama nicht erreichbar. "
            "Embedding-Indexierung wird ubersprungen, symbolische Suche bleibt aktiv."
        )
    else:
        warning = ""

    # Resolve content
    content = ""
    if file_upload is not None:
        try:
            content = Path(file_upload.name).read_text(encoding="utf-8", errors="replace")
            if not source_path.strip():
                source_path = Path(file_upload.name).name
        except Exception as exc:
            return json.dumps({"error": f"Datei konnte nicht gelesen werden: {exc}"}, indent=2)
    elif text_input.strip():
        content = text_input.strip()
    else:
        return json.dumps({"error": "Kein Inhalt angegeben (Text oder Datei)."}, indent=2)

    if not source_path.strip():
        source_path = "manual_input.txt"

    # Build Source
    try:
        content_hash = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
        source_id = Source.make_id(repo.strip() or "ui", source_path.strip())
        source = Source(
            source_id=source_id,
            source_type="repo_file" if repo.strip() else "note",
            repo=repo.strip(),
            path=source_path.strip(),
            content_hash=content_hash,
            captured_at=_now_iso(),
        )
    except Exception as exc:
        return json.dumps({"error": f"Source-Erstellung fehlgeschlagen: {exc}"}, indent=2)

    conn = _get_conn()
    if conn is None:
        return json.dumps({"error": "SQLite-Datenbank nicht verfugbar."}, indent=2)

    chroma = _get_chroma() if ollama_available else None

    try:
        result = ingest(
            source=source,
            content=content,
            conn=conn,
            chroma_collection=chroma,
            ollama_url=_ollama_url,
            model=model,
            opts={
                "categorize": categorize and ollama_available,
                "mode": mode,
                "codebook": codebook,
                "log": True,
            },
        )
    except Exception as exc:
        return json.dumps({"error": f"Ingest fehlgeschlagen: {exc}"}, indent=2)

    output = {k: v for k, v in result.items()}
    if warning:
        output["warning"] = warning
    return json.dumps(output, indent=2, ensure_ascii=False)


def _do_ingest_conversation(
    summary_text: str,
    session_id: str,
    run_id: str,
    model: str,
    ollama_available: bool,
) -> str:
    """Ingest a /compact summary as conversation_summary source."""
    if not _MEMORY_READY:
        return json.dumps({"error": f"Memory-Module nicht geladen: {_IMPORT_ERROR}"}, indent=2)
    if not summary_text.strip():
        return json.dumps({"error": "Kein Inhalt angegeben."}, indent=2)

    conn = _get_conn()
    if conn is None:
        return json.dumps({"error": "SQLite-Datenbank nicht verfügbar."}, indent=2)

    chroma = _get_chroma() if ollama_available else None

    try:
        result = ingest_conversation_summary(
            summary_text=summary_text.strip(),
            conn=conn,
            chroma_collection=chroma,
            ollama_url=_ollama_url,
            model=model if ollama_available else "",
            session_id=session_id.strip() or None,
            run_id=run_id.strip() or None,
        )
    except Exception as exc:
        return json.dumps({"error": f"Conversation-Ingest fehlgeschlagen: {exc}"}, indent=2)

    return json.dumps(result, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tab 3: Memory Browser
# ---------------------------------------------------------------------------

def _load_sources(repo_filter: str, category_filter: str) -> list[list]:
    """Load all sources from SQLite, applying optional filters."""
    if not _MEMORY_READY:
        return []
    conn = _get_conn()
    if conn is None:
        return []

    try:
        rows = conn.execute(
            "SELECT source_id, source_type, repo, path, captured_at FROM sources ORDER BY captured_at DESC"
        ).fetchall()
    except Exception:
        return []

    result = []
    for r in rows:
        sid, stype, srepo, spath, cat = r
        if repo_filter and repo_filter.lower() not in srepo.lower():
            continue
        result.append([sid, stype, srepo, spath, cat])
    return result


def _load_chunks_for_source(source_id: str) -> str:
    """Return chunk details for a given source_id as Markdown."""
    if not source_id or not _MEMORY_READY:
        return "_Keine Source ausgewahlt._"
    conn = _get_conn()
    if conn is None:
        return "_Datenbank nicht verfugbar._"

    try:
        chunks = get_chunks_by_source(conn, source_id, active_only=True)
    except Exception as exc:
        return f"_Fehler: {exc}_"

    if not chunks:
        return f"_Keine aktiven Chunks fur `{source_id}`._"

    lines = [f"## Chunks fur `{source_id}`", f"Gesamt: {len(chunks)}", ""]
    for c in chunks[:20]:  # show max 20
        cats = ", ".join(c.category_labels) or "-"
        preview = c.text[:200].replace("\n", " ")
        lines.append(f"**{c.chunk_id[:12]}** | Level: `{c.chunk_level}` | Kategorien: {cats}")
        lines.append(f"> {preview}{'...' if len(c.text) > 200 else ''}")
        lines.append("")

    if len(chunks) > 20:
        lines.append(f"_... und {len(chunks) - 20} weitere Chunks._")

    return "\n".join(lines)


def _get_repo_choices() -> list[str]:
    """Return distinct repo values from sources table."""
    if not _MEMORY_READY:
        return [""]
    conn = _get_conn()
    if conn is None:
        return [""]
    try:
        rows = conn.execute("SELECT DISTINCT repo FROM sources ORDER BY repo").fetchall()
        return [""] + [r[0] for r in rows if r[0]]
    except Exception:
        return [""]


# ---------------------------------------------------------------------------
# Tab 4: Feedback & Labels
# ---------------------------------------------------------------------------

def _do_feedback(chunk_id: str, signal: str, label: str) -> str:
    """Write feedback for a chunk. Returns status message."""
    if not _MEMORY_READY:
        return f"Memory-Module nicht geladen: {_IMPORT_ERROR}"
    if not chunk_id.strip():
        return "Bitte Chunk-ID eingeben."

    conn = _get_conn()
    if conn is None:
        return "SQLite-Datenbank nicht verfugbar."

    metadata: dict = {}
    if label.strip():
        metadata["label"] = label.strip()

    try:
        add_feedback(conn, chunk_id.strip(), signal, metadata)
        return f"Feedback gespeichert: chunk_id={chunk_id.strip()}, signal={signal}"
    except Exception as exc:
        return f"Fehler beim Speichern: {exc}"


# ---------------------------------------------------------------------------
# App Builder
# ---------------------------------------------------------------------------

def build_app(ollama_url: str) -> gr.Blocks:
    """Construct and return the Gradio Blocks app."""
    global _ollama_url
    _ollama_url = ollama_url

    ollama_available, ollama_models = check_ollama(ollama_url)

    with gr.Blocks(title="MayringCoder Memory UI", theme=gr.themes.Soft()) as app:

        # --- Persistent status badge + model selector ---
        with gr.Row():
            gr.HTML(_status_html(ollama_available, ollama_models))
            model_selector = gr.Dropdown(
                choices=ollama_models,
                value=ollama_models[0] if ollama_models else None,
                label="Modell",
                interactive=ollama_available,
                scale=1,
                min_width=200,
            )

        if not _MEMORY_READY:
            gr.Markdown(
                f"> **Warnung:** Memory-Module konnten nicht geladen werden.\n>\n"
                f"> `{_IMPORT_ERROR}`\n>\n"
                f"> Die UI ist eingeschrankt funktionsfahig."
            )

        # -----------------------------------------------------------------------
        # Tab 1: Suche
        # -----------------------------------------------------------------------
        with gr.Tab("Suche"):
            gr.Markdown("### Hybrid Memory Suche")
            with gr.Row():
                search_input = gr.Textbox(
                    label="Suchanfrage",
                    placeholder="z.B. 'error handling authentication'",
                    scale=4,
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=20, value=8, step=1,
                    label="Top-K",
                    scale=1,
                )
            search_btn = gr.Button("Suchen", variant="primary")
            search_status = gr.Markdown("")
            search_table = gr.Dataframe(
                headers=["chunk_id", "score", "source_id", "kategorien", "text_preview", "grunde"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
            )

            def _search_handler(q, k):
                status, rows = _do_search(q, k, ollama_available)
                return status, rows

            search_btn.click(
                fn=_search_handler,
                inputs=[search_input, top_k_slider],
                outputs=[search_status, search_table],
            )
            search_input.submit(
                fn=_search_handler,
                inputs=[search_input, top_k_slider],
                outputs=[search_status, search_table],
            )

        # -----------------------------------------------------------------------
        # Tab 2: Ingest
        # -----------------------------------------------------------------------
        with gr.Tab("Ingest"):
            gr.Markdown("### Inhalt in Memory aufnehmen")

            if not ollama_available:
                gr.Markdown(
                    "> **Ollama nicht erreichbar.** "
                    "Embeddings werden nicht erstellt. "
                    "Nur symbolische Suche verfugbar nach dem Ingest."
                )

            with gr.Row():
                ingest_repo = gr.Textbox(
                    label="Repo (optional)",
                    placeholder="owner/repo-name",
                    scale=2,
                )
                ingest_path = gr.Textbox(
                    label="Quellpfad / Label",
                    placeholder="src/module.py",
                    scale=3,
                )
            ingest_text = gr.Textbox(
                label="Text (direkt eingeben oder Datei hochladen)",
                placeholder="Code oder Text hier einfugen...",
                lines=8,
            )
            ingest_file = gr.File(
                label="Oder Datei hochladen",
                file_types=[".py", ".md", ".txt", ".js", ".ts", ".yaml", ".yml", ".json"],
            )
            ingest_categorize = gr.Checkbox(
                label="Mayring-Kategorisierung via LLM",
                value=False,
                info="Nur aktiv wenn Ollama erreichbar ist.",
            )
            with gr.Column(visible=False) as mayring_opts_col:
                ingest_mode = gr.Radio(
                    choices=["hybrid", "deductive", "inductive"],
                    value="hybrid",
                    label="Mayring-Modus",
                    info="hybrid = Anker + neue Kategorien | deductive = nur Codebook | inductive = frei",
                )
                ingest_codebook = gr.Dropdown(
                    choices=["auto", "code", "social", "original"],
                    value="auto",
                    label="Codebook",
                    info="auto = anhand source_type | code = Code-Analyse | social = Sozialforschung | original = Mayring-Basiskategorien",
                )

            ingest_categorize.change(
                fn=lambda v: gr.Column(visible=v),
                inputs=[ingest_categorize],
                outputs=[mayring_opts_col],
            )
            ingest_btn = gr.Button("Ingest starten", variant="primary")
            ingest_output = gr.Code(language="json", label="Ergebnis")

            def _ingest_handler(text, file, path, repo, cat, mode, codebook, model):
                return _do_ingest(text, file, path, repo, cat, mode, codebook, model, ollama_available)

            ingest_btn.click(
                fn=_ingest_handler,
                inputs=[ingest_text, ingest_file, ingest_path, ingest_repo,
                        ingest_categorize, ingest_mode, ingest_codebook, model_selector],
                outputs=[ingest_output],
            )

        # -----------------------------------------------------------------------
        # Tab 3: Memory Browser
        # -----------------------------------------------------------------------
        with gr.Tab("Memory Browser"):
            gr.Markdown("### Alle gespeicherten Sources")

            with gr.Row():
                browser_repo_filter = gr.Textbox(
                    label="Repo-Filter (Freitext)",
                    placeholder="z.B. owner/repo",
                    scale=3,
                )
                browser_load_btn = gr.Button("Laden / Aktualisieren", scale=1)

            browser_table = gr.Dataframe(
                headers=["source_id", "typ", "repo", "pfad", "erfasst_am"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False,
            )

            gr.Markdown("#### Chunk-Detail")
            browser_source_id_input = gr.Textbox(
                label="source_id (aus Tabelle kopieren)",
                placeholder="repo:owner/name:pfad/datei.py",
            )
            browser_chunk_btn = gr.Button("Chunks anzeigen")
            browser_chunk_detail = gr.Markdown("_Noch keine Source ausgewahlt._")

            def _browser_load(repo_f):
                return _load_sources(repo_f, "")

            def _browser_chunks(sid):
                return _load_chunks_for_source(sid)

            browser_load_btn.click(
                fn=_browser_load,
                inputs=[browser_repo_filter],
                outputs=[browser_table],
            )
            browser_chunk_btn.click(
                fn=_browser_chunks,
                inputs=[browser_source_id_input],
                outputs=[browser_chunk_detail],
            )

            # Auto-load on tab open
            app.load(fn=lambda: _load_sources("", ""), outputs=[browser_table])

        # -----------------------------------------------------------------------
        # Tab 4: Feedback & Labels
        # -----------------------------------------------------------------------
        with gr.Tab("Feedback & Labels"):
            gr.Markdown("### Chunk-Feedback erfassen")

            feedback_chunk_id = gr.Textbox(
                label="Chunk-ID",
                placeholder="chk_a1b2c3d4e5f6...",
            )
            feedback_signal = gr.Radio(
                choices=["positive", "negative", "neutral"],
                value="positive",
                label="Signal",
            )
            feedback_label = gr.Textbox(
                label="Freies Label (optional)",
                placeholder="z.B. 'relevant fur Auth', 'Duplikat', ...",
            )
            feedback_btn = gr.Button("Feedback speichern", variant="primary")
            feedback_status = gr.Textbox(label="Status", interactive=False)

            feedback_btn.click(
                fn=_do_feedback,
                inputs=[feedback_chunk_id, feedback_signal, feedback_label],
                outputs=[feedback_status],
            )

        # -----------------------------------------------------------------------
        # Tab 5: Conversation (Task X)
        # -----------------------------------------------------------------------
        with gr.Tab("Conversation"):
            gr.Markdown(
                "### /compact-Output als Memory speichern\n\n"
                "Füge hier den Output von Claude's `/compact`-Befehl ein. "
                "Er wird als `conversation_summary`-Quelle gespeichert und ist via Suche abrufbar."
            )

            if not ollama_available:
                gr.Markdown(
                    "> **Ollama nicht erreichbar.** Embedding wird übersprungen. "
                    "Nur symbolische Suche nach dem Ingest."
                )

            conv_text = gr.Textbox(
                label="Zusammenfassung (/compact Output)",
                placeholder="## Session Summary\n\n...",
                lines=12,
            )
            with gr.Row():
                conv_session_id = gr.Textbox(
                    label="session_id (optional)",
                    placeholder="z.B. sess-2026-04-08",
                    scale=1,
                )
                conv_run_id = gr.Textbox(
                    label="run_id (optional)",
                    placeholder="z.B. run-001",
                    scale=1,
                )
            conv_btn = gr.Button("Als Memory speichern", variant="primary")
            conv_output = gr.Code(language="json", label="Ergebnis")

            def _conv_handler(text, session_id, run_id, model):
                return _do_ingest_conversation(text, session_id, run_id, model, ollama_available)

            conv_btn.click(
                fn=_conv_handler,
                inputs=[conv_text, conv_session_id, conv_run_id, model_selector],
                outputs=[conv_output],
            )

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MayringCoder Memory WebUI (Gradio)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=7860, help="HTTP-Port")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind-Adresse (127.0.0.1 fur lokal, 0.0.0.0 fur Docker)",
    )
    parser.add_argument(
        "--auth",
        default=None,
        help="Authentifizierung im Format user:pass",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama-Basis-URL",
    )
    args = parser.parse_args()

    # Parse auth
    auth = None
    if args.auth:
        parts = args.auth.split(":", 1)
        if len(parts) == 2:
            auth = (parts[0], parts[1])
        else:
            parser.error("--auth muss das Format 'user:pass' haben.")

    app = build_app(ollama_url=args.ollama_url)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        auth=auth,
        show_error=True,
    )


if __name__ == "__main__":
    main()
