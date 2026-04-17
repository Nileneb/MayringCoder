"""Gradio WebUI for MayringCoder Memory Layer.

Usage:
    python -m src.api.web_ui [--port 7860] [--ollama-url http://localhost:11434] [--api-url http://localhost:8080]

Tabs:
    Memory   — Suche + Source-Browser in einem View
    Ingest   — Repo / Text-Datei / Compact-Paste in einem Formular
    Analyse  — Vollanalyse / Overview / Turbulenz mit Issues default-an
    Feedback — Feedback + Benchmark + Training-Export
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

try:
    import gradio as gr
    _HAS_GRADIO = True
except ImportError as _e:
    raise SystemExit(
        "gradio is not installed. Run: pip install -r requirements-ui.txt"
    ) from _e

from src.ollama_client import check_ollama

try:
    import httpx as _httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

_MEMORY_READY = False
_IMPORT_ERROR: str = ""

try:
    from src.memory.store import (
        init_memory_db,
        add_feedback,
        get_chunks_by_source,
    )
    from src.memory.ingest import ingest, get_or_create_chroma_collection, ingest_conversation_summary
    from src.memory.retrieval import search
    from src.memory.schema import Source, Chunk
    from src.config import CACHE_DIR
    import hashlib
    from datetime import datetime, timezone

    _MEMORY_READY = True
except Exception as exc:
    _IMPORT_ERROR = str(exc)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_conn: sqlite3.Connection | None = None
_chroma_collection: Any = None
_ollama_url: str = "http://localhost:11434"
_api_url: str = "http://localhost:8080"


def _get_conn() -> sqlite3.Connection | None:
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
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    if not _MEMORY_READY:
        return None
    try:
        from src.memory.store import get_chroma_collection as get_collection
        _chroma_collection = get_collection()
        return _chroma_collection
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
# Memory: Search
# ---------------------------------------------------------------------------

def _do_search(query: str, top_k: int, ollama_available: bool) -> tuple[str, list[list]]:
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
# Memory: Source Browser
# ---------------------------------------------------------------------------

def _load_sources(repo_filter: str, category_filter: str) -> list[list]:
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
    for c in chunks[:20]:
        cats = ", ".join(c.category_labels) or "-"
        preview = c.text[:200].replace("\n", " ")
        lines.append(f"**{c.chunk_id[:12]}** | Level: `{c.chunk_level}` | Kategorien: {cats}")
        lines.append(f"> {preview}{'...' if len(c.text) > 200 else ''}")
        lines.append("")
    if len(chunks) > 20:
        lines.append(f"_... und {len(chunks) - 20} weitere Chunks._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ingest: Text/Datei
# ---------------------------------------------------------------------------

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
    if not _MEMORY_READY:
        return json.dumps({"error": f"Memory-Module nicht geladen: {_IMPORT_ERROR}"}, indent=2)

    warning = (
        "[Warnung] Ollama nicht erreichbar. Embedding wird ubersprungen."
        if not ollama_available else ""
    )

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

    output = dict(result)
    if warning:
        output["warning"] = warning
    return json.dumps(output, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Ingest: Conversation (/compact)
# ---------------------------------------------------------------------------

def _do_ingest_conversation(
    summary_text: str,
    session_id: str,
    run_id: str,
    model: str,
    ollama_available: bool,
) -> str:
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
# Ingest: Auto-scan Claude compact files
# ---------------------------------------------------------------------------

def _scan_compact_files(model: str, ollama_available: bool) -> str:
    """Scan ~/.claude/projects/*/memory/*.md and ingest new ones."""
    if not _MEMORY_READY:
        return f"Memory nicht verfügbar: {_IMPORT_ERROR}"

    claude_base = Path.home() / ".claude" / "projects"
    if not claude_base.exists():
        return "Kein ~/.claude/projects/ Verzeichnis gefunden."

    md_files = list(claude_base.rglob("memory/*.md"))
    if not md_files:
        return "Keine Compact-Dateien gefunden."

    conn = _get_conn()
    if conn is None:
        return "SQLite nicht verfügbar."

    chroma = _get_chroma() if ollama_available else None
    ingested, skipped, errors = 0, 0, 0

    for f in md_files:
        try:
            content = f.read_text(encoding="utf-8")
            if not content.strip():
                skipped += 1
                continue
            content_hash = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
            source_id = f"note:claude_memory:{f.stem}"
            source = Source(
                source_id=source_id,
                source_type="note",
                repo="claude_memory",
                path=str(f.relative_to(Path.home())),
                content_hash=content_hash,
                captured_at=_now_iso(),
            )
            result = ingest_conversation_summary(
                summary_text=content,
                conn=conn,
                chroma_collection=chroma,
                ollama_url=_ollama_url,
                model=model if ollama_available else "",
            )
            if result.get("skipped"):
                skipped += 1
            else:
                ingested += 1
        except Exception:
            errors += 1

    parts = []
    if ingested:
        parts.append(f"✅ {ingested} neu ingested")
    if skipped:
        parts.append(f"⏭ {skipped} unverändert übersprungen")
    if errors:
        parts.append(f"❌ {errors} Fehler")
    return " · ".join(parts) if parts else "Nichts zu tun."


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def _do_feedback(chunk_id: str, signal: str, label: str) -> str:
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
# Auth + API Helpers
# ---------------------------------------------------------------------------

def _validate_token(token: str) -> tuple[bool, str]:
    if not token.strip():
        return False, "Kein Token eingegeben."
    try:
        from src.api.sanctum_auth import validate_sanctum_token
        ws = validate_sanctum_token(token.strip())
        if ws:
            return True, ws
        return False, "Token ungültig oder abgelaufen."
    except Exception as exc:
        return False, f"Auth-Fehler: {exc}"


def _api_post(endpoint: str, payload: dict, token: str) -> dict:
    if not _HAS_HTTPX:
        return {"error": "httpx nicht installiert"}
    if not token:
        return {"error": "Nicht eingeloggt."}
    try:
        r = _httpx.post(
            f"{_api_url}/{endpoint.lstrip('/')}",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def _api_get_job(job_id: str, token: str) -> dict:
    if not _HAS_HTTPX or not token:
        return {"error": "Nicht verbunden."}
    try:
        r = _httpx.get(
            f"{_api_url}/jobs/{job_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def _poll_job(job_id: str, token: str) -> str:
    if not job_id or not token:
        return "Job-ID oder Token fehlt."
    r = _api_get_job(job_id, token)
    if "error" in r:
        return f"❌ {r['error']}"
    status = r.get("status", "?")
    output = r.get("output", "")
    return f"**Status:** {status}\n\n```\n{output[-3000:] if output else ''}\n```"


def _training_files() -> list[Path]:
    if not _MEMORY_READY:
        return []
    try:
        from src.config import CACHE_DIR
        return sorted(CACHE_DIR.glob("*_training_log.jsonl"))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# App Builder
# ---------------------------------------------------------------------------

def build_app(ollama_url: str, api_url: str = "http://localhost:8080") -> gr.Blocks:
    global _ollama_url, _api_url
    _ollama_url = ollama_url
    _api_url = api_url

    ollama_available, ollama_models = check_ollama(ollama_url)

    with gr.Blocks(title="MayringCoder", theme=gr.themes.Soft()) as app:

        _token_state = gr.State("")
        _workspace_state = gr.State("")

        # --- Header: Login + Status ---
        with gr.Accordion("Einloggen (Sanctum Token)", open=True) as login_accordion:
            with gr.Row():
                login_token_input = gr.Textbox(
                    label="Personal Access Token (aus app.linn.games)",
                    placeholder="1|abc123...",
                    type="password",
                    scale=4,
                )
                login_btn = gr.Button("Einloggen", variant="primary", scale=1)
            login_status = gr.Markdown("_Nicht eingeloggt._")

        def _login(token):
            ok, result = _validate_token(token)
            if ok:
                return (
                    token, result,
                    f"Eingeloggt als Workspace: `{result}`",
                    gr.Accordion(open=False),
                )
            return "", "", f"Fehler: {result}", gr.Accordion(open=True)

        login_btn.click(
            fn=_login,
            inputs=[login_token_input],
            outputs=[_token_state, _workspace_state, login_status, login_accordion],
        )

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
                f"> **Warnung:** Memory-Module konnten nicht geladen werden.\n> `{_IMPORT_ERROR}`"
            )

        # =======================================================================
        # Tab 1: Memory — Suche + Browser
        # =======================================================================
        with gr.Tab("Memory"):

            # --- Suche (prominent oben) ---
            gr.Markdown("### Suche")
            with gr.Row():
                search_input = gr.Textbox(
                    label="Suchanfrage",
                    placeholder="z.B. 'error handling authentication'",
                    scale=4,
                )
                top_k_slider = gr.Slider(minimum=1, maximum=20, value=8, step=1, label="Top-K", scale=1)
            search_btn = gr.Button("Suchen", variant="primary")
            search_status = gr.Markdown("")
            search_table = gr.Dataframe(
                headers=["chunk_id", "score", "source_id", "kategorien", "vorschau", "grunde"],
                datatype=["str"] * 6,
                interactive=False,
            )

            def _search_handler(q, k):
                return _do_search(q, k, ollama_available)

            search_btn.click(fn=_search_handler, inputs=[search_input, top_k_slider],
                             outputs=[search_status, search_table])
            search_input.submit(fn=_search_handler, inputs=[search_input, top_k_slider],
                                outputs=[search_status, search_table])

            gr.Markdown("---")

            # --- Source-Browser (unten) ---
            with gr.Accordion("Alle Sources", open=False):
                with gr.Row():
                    browser_repo_filter = gr.Textbox(
                        label="Repo-Filter", placeholder="owner/repo", scale=3
                    )
                    browser_load_btn = gr.Button("Laden", scale=1)

                browser_table = gr.Dataframe(
                    headers=["source_id", "typ", "repo", "pfad", "erfasst_am"],
                    datatype=["str"] * 5,
                    interactive=False,
                )
                browser_source_id_input = gr.Textbox(
                    label="source_id eingeben für Chunk-Detail",
                    placeholder="repo:owner/name:pfad/datei.py",
                )
                browser_chunk_btn = gr.Button("Chunks anzeigen")
                browser_chunk_detail = gr.Markdown("_Noch keine Source ausgewählt._")

                browser_load_btn.click(
                    fn=lambda r: _load_sources(r, ""),
                    inputs=[browser_repo_filter],
                    outputs=[browser_table],
                )
                browser_chunk_btn.click(
                    fn=_load_chunks_for_source,
                    inputs=[browser_source_id_input],
                    outputs=[browser_chunk_detail],
                )
                app.load(fn=lambda: _load_sources("", ""), outputs=[browser_table])

        # =======================================================================
        # Tab 2: Ingest — Repo / Text-Datei / Compact
        # =======================================================================
        with gr.Tab("Ingest"):
            gr.Markdown("### Inhalt in Memory aufnehmen")

            ingest_mode_radio = gr.Radio(
                choices=["Text / Datei", "Conversation (/compact)"],
                value="Text / Datei",
                label="Quelle",
            )

            # -- Panel A: Text/Datei --
            with gr.Column(visible=True) as panel_file:
                if not ollama_available:
                    gr.Markdown("> **Ollama offline.** Kein Embedding, nur symbolische Suche.")

                with gr.Row():
                    ingest_repo = gr.Textbox(label="Repo (optional)", placeholder="owner/repo", scale=2)
                    ingest_path = gr.Textbox(label="Quellpfad / Label", placeholder="src/module.py", scale=3)
                ingest_text = gr.Textbox(
                    label="Text",
                    placeholder="Code oder Text hier einfügen...",
                    lines=8,
                )
                ingest_file = gr.File(
                    label="Oder Datei hochladen",
                    file_types=[".py", ".md", ".txt", ".js", ".ts", ".yaml", ".yml", ".json"],
                )
                ingest_categorize = gr.Checkbox(
                    label="Mayring-Kategorisierung via LLM",
                    value=False,
                    info="Nur aktiv wenn Ollama erreichbar.",
                )
                with gr.Column(visible=False) as mayring_opts_col:
                    ingest_mode = gr.Radio(
                        choices=["hybrid", "deductive", "inductive"],
                        value="hybrid",
                        label="Mayring-Modus",
                    )
                    ingest_codebook = gr.Dropdown(
                        choices=["auto", "code", "social", "original"],
                        value="auto",
                        label="Codebook",
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

            # -- Panel B: Conversation --
            with gr.Column(visible=False) as panel_conv:
                gr.Markdown(
                    "Füge den Output von `/compact` ein **oder** scanne automatisch alle "
                    "Claude-Memory-Dateien aus `~/.claude/projects/`."
                )
                if not ollama_available:
                    gr.Markdown("> **Ollama offline.** Kein Embedding.")

                conv_text = gr.Textbox(
                    label="Zusammenfassung (/compact Output)",
                    placeholder="## Session Summary\n\n...",
                    lines=10,
                )
                with gr.Row():
                    conv_session_id = gr.Textbox(
                        label="session_id (optional)", placeholder="sess-2026-04-08", scale=1
                    )
                    conv_run_id = gr.Textbox(
                        label="run_id (optional)", placeholder="run-001", scale=1
                    )
                with gr.Row():
                    conv_btn = gr.Button("Manuell speichern", variant="primary", scale=2)
                    conv_scan_btn = gr.Button("Claude-Dateien auto-scannen", variant="secondary", scale=2)
                conv_output = gr.Code(language="json", label="Ergebnis (manuell)")
                conv_scan_output = gr.Textbox(label="Auto-Scan Ergebnis", interactive=False)

                def _conv_handler(text, session_id, run_id, model):
                    return _do_ingest_conversation(text, session_id, run_id, model, ollama_available)

                def _scan_handler(model):
                    return _scan_compact_files(model, ollama_available)

                conv_btn.click(
                    fn=_conv_handler,
                    inputs=[conv_text, conv_session_id, conv_run_id, model_selector],
                    outputs=[conv_output],
                )
                conv_scan_btn.click(
                    fn=_scan_handler,
                    inputs=[model_selector],
                    outputs=[conv_scan_output],
                )

            def _switch_panel(choice):
                return (
                    gr.Column(visible=(choice == "Text / Datei")),
                    gr.Column(visible=(choice == "Conversation (/compact)")),
                )

            ingest_mode_radio.change(
                fn=_switch_panel,
                inputs=[ingest_mode_radio],
                outputs=[panel_file, panel_conv],
            )

        # =======================================================================
        # Tab 3: Analyse — alle Pipeline-Modi + Issues default-an
        # =======================================================================
        with gr.Tab("Analyse"):
            gr.Markdown(
                "### Repo-Analyse\n"
                "_Vollanalyse_ = Fehlersuche per LLM · "
                "_Inventar_ = Summary aller Dateien · "
                "_Hot-Zones_ = Turbulenz-Analyse (kein LLM nötig)"
            )

            analyse_repo = gr.Textbox(
                label="Repository URL",
                placeholder="https://github.com/owner/repo",
            )
            analyse_mode = gr.Radio(
                choices=["Vollanalyse (Fehlersuche)", "Inventar (Overview)", "Hot-Zones (Turbulenz)"],
                value="Vollanalyse (Fehlersuche)",
                label="Modus",
            )
            with gr.Row():
                analyse_issues = gr.Checkbox(
                    label="Issues miterfassen",
                    value=True,
                    info="GitHub Issues werden automatisch in Memory ingested.",
                )
                analyse_memory = gr.Checkbox(
                    label="Findings in Memory speichern",
                    value=True,
                )
                analyse_adversarial = gr.Checkbox(
                    label="Adversarial (False-Positive-Filter)",
                    value=False,
                )
                analyse_llm = gr.Checkbox(
                    label="LLM-Modus (nur Hot-Zones)",
                    value=False,
                )

            analyse_btn = gr.Button("Analyse starten", variant="primary")
            analyse_job_id = gr.Textbox(label="Job-ID", interactive=False)
            analyse_poll_btn = gr.Button("Status abfragen")
            analyse_output = gr.Markdown("")

            def _do_analyse(repo, mode, issues, memory, adversarial, llm_mode, token):
                if not token:
                    return "", "Erst einloggen."
                mode_map = {
                    "Vollanalyse (Fehlersuche)": "analyze",
                    "Inventar (Overview)": "overview",
                    "Hot-Zones (Turbulenz)": "turbulence",
                }
                endpoint = mode_map.get(mode, "analyze")
                payload: dict = {
                    "repo": repo,
                    "adversarial": adversarial,
                    "populate_memory": memory,
                    "ingest_issues": issues,
                    "llm": llm_mode,
                }
                r = _api_post(endpoint, payload, token)
                if "error" in r:
                    return "", f"Fehler: {r['error']}"
                job_id = r.get("job_id") or r.get("pid", "")
                return str(job_id), f"Gestartet (Job: {job_id or '?'})"

            analyse_btn.click(
                fn=_do_analyse,
                inputs=[analyse_repo, analyse_mode, analyse_issues, analyse_memory,
                        analyse_adversarial, analyse_llm, _token_state],
                outputs=[analyse_job_id, analyse_output],
            )
            analyse_poll_btn.click(
                fn=_poll_job,
                inputs=[analyse_job_id, _token_state],
                outputs=[analyse_output],
            )

        # =======================================================================
        # Tab 4: Feedback & Qualität
        # =======================================================================
        with gr.Tab("Feedback & Qualität"):

            with gr.Accordion("Chunk-Feedback", open=True):
                gr.Markdown("Chunks nach Relevanz labeln — verbessert zukünftige Retrieval-Rankings.")
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
                    placeholder="z.B. 'relevant fur Auth', 'Duplikat'",
                )
                feedback_btn = gr.Button("Feedback speichern", variant="primary")
                feedback_status = gr.Textbox(label="Status", interactive=False)

                feedback_btn.click(
                    fn=_do_feedback,
                    inputs=[feedback_chunk_id, feedback_signal, feedback_label],
                    outputs=[feedback_status],
                )

            with gr.Accordion("Benchmark (Retrieval-Qualität)", open=False):
                gr.Markdown("MRR, Recall@1, Recall@K über definierte Test-Queries.")
                bench_top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top-K")
                bench_repo = gr.Textbox(label="Repo-Filter (optional)", placeholder="Nileneb/MayringCoder")
                bench_btn = gr.Button("Benchmark starten", variant="primary")
                bench_job_id = gr.Textbox(label="Job-ID", interactive=False)
                bench_poll_btn = gr.Button("Status abfragen")
                bench_output = gr.Markdown("")

                def _do_benchmark(top_k, repo, token):
                    if not token:
                        return "", "Erst einloggen."
                    payload: dict = {"top_k": int(top_k)}
                    if repo.strip():
                        payload["repo"] = repo.strip()
                    r = _api_post("benchmark", payload, token)
                    if "error" in r:
                        return "", f"Fehler: {r['error']}"
                    return r.get("job_id", ""), f"Gestartet (Job: {r.get('job_id', '?')})"

                bench_btn.click(
                    fn=_do_benchmark,
                    inputs=[bench_top_k, bench_repo, _token_state],
                    outputs=[bench_job_id, bench_output],
                )
                bench_poll_btn.click(
                    fn=_poll_job,
                    inputs=[bench_job_id, _token_state],
                    outputs=[bench_output],
                )

            with gr.Accordion("Training Data Export", open=False):
                gr.Markdown(
                    "Prompt/Response-Logs aus `cache/*_training_log.jsonl` für Unsloth/LoRA-Finetuning."
                )
                training_files_list = gr.Markdown("_Lade Dateiliste..._")
                training_refresh_btn = gr.Button("Aktualisieren")
                training_download = gr.File(label="JSONL-Datei herunterladen", interactive=False)
                training_file_select = gr.Dropdown(label="Datei auswählen", choices=[], interactive=True)

                def _load_training_list():
                    files = _training_files()
                    if not files:
                        return "_Keine Training-Logs gefunden._", [], None
                    lines = [f"- `{f.name}` ({f.stat().st_size // 1024} KB)" for f in files]
                    return "\n".join(lines), [str(f) for f in files], None

                training_refresh_btn.click(
                    fn=_load_training_list,
                    outputs=[training_files_list, training_file_select, training_download],
                )
                training_file_select.change(
                    fn=lambda p: p or None,
                    inputs=[training_file_select],
                    outputs=[training_download],
                )
                app.load(
                    fn=_load_training_list,
                    outputs=[training_files_list, training_file_select, training_download],
                )

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MayringCoder WebUI (Gradio)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("MAYRING_API_URL", "http://localhost:8080"),
    )
    args = parser.parse_args()

    app = build_app(ollama_url=args.ollama_url, api_url=args.api_url)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        root_path=os.environ.get("GRADIO_ROOT_PATH", ""),
        show_error=True,
    )


if __name__ == "__main__":
    main()
