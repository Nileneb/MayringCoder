from __future__ import annotations

import gradio as gr

from src.api.web_ui_helpers import (
    _MEMORY_READY,
    _IMPORT_ERROR,
    _HAS_HTTPX,
    _api_url,
    _do_search,
    _load_sources,
    _load_chunks_for_source,
    _do_ingest,
    _do_ingest_conversation,
    _scan_compact_files,
    _api_post,
    _poll_job,
    _training_files,
)

try:
    from src.github import parse_github_input, GitHubInputError
    import httpx as _httpx
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Training-Tab Helpers (call /api/training/*)
# ---------------------------------------------------------------------------

def _get_training_status() -> str:
    try:
        import httpx
        import src.api.web_ui_helpers as _h
        r = httpx.get(f"{_h._api_url}/api/training/status", timeout=5)
        d = r.json()
        job = d.get("finetune_job", {})
        job_str = f" | Job: {job.get('status', 'idle')}" if job.get("status") != "idle" else ""
        return (
            f"**Trainingsdaten:** {d.get('sample_count', 0)} Samples (Val: {d.get('val_count', 0)}) | "
            f"**Annotiert:** {d.get('annotated', 0)} (Haiku: {d.get('haiku_annotations', 0)}, "
            f"Langdock: {d.get('langdock_batches', 0)}){job_str}"
        )
    except Exception as e:
        return f"Fehler beim Laden: {e}"


def _export_training_batch() -> str:
    try:
        import httpx
        import src.api.web_ui_helpers as _h
        r = httpx.post(f"{_h._api_url}/api/training/batches/export", timeout=30)
        d = r.json()
        if d.get("sample_count", 0) == 0:
            return d.get("message", "Keine neuen Samples gefunden.")
        return f"✅ Batch `{d['batch_id']}` exportiert — **{d['sample_count']} Samples** in `{d.get('batch_file', '')}`"
    except Exception as e:
        return f"Fehler: {e}"


def _merge_training_annotations() -> str:
    try:
        import httpx
        import src.api.web_ui_helpers as _h
        r = httpx.post(f"{_h._api_url}/api/training/merge", timeout=60)
        d = r.json()
        return (
            f"✅ Merge fertig: **{d.get('total', 0)} Samples** in train.jsonl | "
            f"+{d.get('added', 0)} neu | {d.get('skipped', 0)} übersprungen"
        )
    except Exception as e:
        return f"Fehler: {e}"


def _trigger_finetune() -> str:
    try:
        import httpx
        import src.api.web_ui_helpers as _h
        r = httpx.post(f"{_h._api_url}/api/training/finetune", timeout=10)
        if r.status_code == 409:
            return "⚠️ Fine-tuning läuft bereits."
        if r.status_code == 400:
            return f"❌ {r.json().get('detail', 'Fehler')}"
        d = r.json()
        return f"🚀 Fine-tuning gestartet (Job: `{d.get('job_id')}`, {d.get('train_samples')} Samples)"
    except Exception as e:
        return f"Fehler: {e}"


def _get_finetune_status() -> str:
    try:
        import httpx
        import src.api.web_ui_helpers as _h
        r = httpx.get(f"{_h._api_url}/api/training/finetune/status", timeout=5)
        d = r.json()
        status = d.get("status", "idle")
        if status == "idle":
            return "Kein laufender Fine-tune-Job."
        elif status == "running":
            return f"⏳ Läuft... (PID: {d.get('pid')}, gestartet: {d.get('started_at', '?')})"
        elif status == "done":
            return f"✅ Fertig (beendet: {d.get('finished_at', '?')})"
        else:
            return f"❌ Fehler: {d.get('error', 'Unbekannt')}"
    except Exception as e:
        return f"Fehler: {e}"


# ---------------------------------------------------------------------------
# Tab Builders
# ---------------------------------------------------------------------------

def build_memory_tab(
    app: gr.Blocks,
    *,
    ollama_available: bool,
    token_state: gr.State,
) -> None:
    with gr.Tab("Memory"):
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


def build_ingest_tab(
    app: gr.Blocks,
    *,
    ollama_available: bool,
    model_selector: gr.Dropdown,
    token_state: gr.State,
) -> None:
    with gr.Tab("Ingest"):
        gr.Markdown("### Inhalt in Memory aufnehmen")

        ingest_mode_radio = gr.Radio(
            choices=["Text / Datei", "GitHub-Repo", "Conversation (/compact)"],
            value="Text / Datei",
            label="Quelle",
        )

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

            def _ingest_handler(text, file, path, repo, cat, mode, codebook, model,
                                progress=gr.Progress(track_tqdm=True)):
                return _do_ingest(text, file, path, repo, cat, mode, codebook, model, ollama_available)

            ingest_btn.click(
                fn=_ingest_handler,
                inputs=[ingest_text, ingest_file, ingest_path, ingest_repo,
                        ingest_categorize, ingest_mode, ingest_codebook, model_selector],
                outputs=[ingest_output],
            )

        with gr.Column(visible=False) as panel_github:
            gr.Markdown(
                "Repository als Datenquelle in Memory ingesten — Sources "
                "(via `gitingest`) und optional GitHub Issues. Läuft als "
                "Hintergrund-Job, Poll-Button für Status."
            )
            gh_repo_input = gr.Textbox(
                label="Repo",
                placeholder="owner/repo oder https://github.com/owner/repo",
            )
            with gr.Row():
                gh_include_issues = gr.Checkbox(
                    label="Issues miterfassen",
                    value=False,
                    info="Ingest GitHub Issues via gh CLI + multiview chunking.",
                )
                gh_force = gr.Checkbox(
                    label="Force re-ingest",
                    value=False,
                    info="Überschreibt bestehende Chunks für diese Sources.",
                )
            with gr.Row():
                gh_ingest_btn = gr.Button("GitHub-Repo ingesten", variant="primary", scale=2)
                gh_poll_btn = gr.Button("Status abfragen", scale=1)
            gh_status = gr.Markdown("")
            gh_job_id = gr.Textbox(label="Job-ID (Sources)", interactive=False)
            gh_issues_job_id = gr.Textbox(label="Job-ID (Issues)", interactive=False, visible=False)
            gh_timer = gr.Timer(value=4, active=False)

            def _do_github_ingest(raw: str, include_issues: bool, force: bool, token: str):
                if not token:
                    return "", "", "Erst via app.linn.games/mayring/dashboard einloggen.", gr.Timer(active=False)
                try:
                    gh = parse_github_input(raw)
                except GitHubInputError as exc:
                    return "", "", f"❌ {exc}", gr.Timer(active=False)

                payload = {"repo": gh.url, "force_reingest": force}
                r = _api_post("populate", payload, token)
                if "error" in r:
                    return "", "", f"❌ /populate: {r['error']}", gr.Timer(active=False)
                job = str(r.get("job_id") or "")
                status = f"⏳ Sources-Ingest ({gh.slug}) gestartet · Job `{job}`"

                issue_job = ""
                if include_issues:
                    r2 = _api_post("issues/ingest", {"repo": gh.slug, "force_reingest": force}, token)
                    if "error" in r2:
                        status += f"\n⚠️ /issues/ingest: {r2['error']}"
                    else:
                        issue_job = str(r2.get("job_id") or "")
                        status += f"\n⏳ Issues-Ingest Job `{issue_job}`"

                return job, issue_job, status, gr.Timer(active=bool(job))

            def _poll_github(job_id: str, issues_job_id: str, token: str):
                if not token:
                    return gr.update(), gr.Timer(active=False)
                parts: list[str] = []
                done_flags: list[bool] = []
                for label, jid in [("Sources", job_id), ("Issues", issues_job_id)]:
                    if not jid:
                        continue
                    txt = _poll_job(jid, token)
                    parts.append(f"### {label} — Job `{jid}`\n{txt}")
                    done_flags.append(any(mark in txt for mark in ("✅", "❌", "finished", "failed")))
                if not parts:
                    return "Keine aktive Ingestion.", gr.Timer(active=False)
                return "\n\n".join(parts), gr.Timer(active=not all(done_flags))

            gh_ingest_btn.click(
                fn=_do_github_ingest,
                inputs=[gh_repo_input, gh_include_issues, gh_force, token_state],
                outputs=[gh_job_id, gh_issues_job_id, gh_status, gh_timer],
            )
            gh_poll_btn.click(
                fn=_poll_github,
                inputs=[gh_job_id, gh_issues_job_id, token_state],
                outputs=[gh_status, gh_timer],
            )
            gh_timer.tick(
                fn=_poll_github,
                inputs=[gh_job_id, gh_issues_job_id, token_state],
                outputs=[gh_status, gh_timer],
            )

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

            def _conv_handler(text, session_id, run_id, model,
                              progress=gr.Progress(track_tqdm=True)):
                return _do_ingest_conversation(text, session_id, run_id, model, ollama_available)

            def _scan_handler(model, progress=gr.Progress(track_tqdm=True)):
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
                gr.Column(visible=(choice == "GitHub-Repo")),
                gr.Column(visible=(choice == "Conversation (/compact)")),
            )

        ingest_mode_radio.change(
            fn=_switch_panel,
            inputs=[ingest_mode_radio],
            outputs=[panel_file, panel_github, panel_conv],
        )


def build_training_tab(app: gr.Blocks) -> None:
    with gr.Tab("Training"):
        gr.Markdown("## Trainingspipeline")

        with gr.Accordion("Status", open=True):
            train_status_md = gr.Markdown(_get_training_status())
            refresh_status_btn = gr.Button("Aktualisieren", size="sm")
            refresh_status_btn.click(fn=_get_training_status, outputs=[train_status_md])

        with gr.Accordion("Langdock Annotation", open=True):
            gr.Markdown(
                "Exportiert nicht-annotierte Samples aus den Training-Logs als Batch. "
                "Langdock annotiert sie und schickt sie per Webhook zurück "
                f"(Endpoint: `POST /api/training/langdock/webhook`)."
            )
            export_btn = gr.Button("Batch exportieren → Langdock", variant="secondary")
            export_out = gr.Markdown("")
            export_btn.click(fn=_export_training_batch, outputs=[export_out])

        with gr.Accordion("Merge + Fine-tune", open=False):
            gr.Markdown(
                "Merged alle annotierten Batches (Langdock + Haiku) in `cache/finetuning/train.jsonl`. "
                "Danach Fine-tuning starten."
            )
            merge_btn = gr.Button("Annotierte Batches mergen", variant="secondary")
            merge_out = gr.Markdown("")
            merge_btn.click(fn=_merge_training_annotations, outputs=[merge_out])

            gr.Markdown("---")
            finetune_btn = gr.Button("Fine-tuning starten", variant="primary")
            finetune_out = gr.Markdown("")
            finetune_btn.click(fn=_trigger_finetune, outputs=[finetune_out])

            gr.Markdown("### Job-Status")
            finetune_status_md = gr.Markdown(_get_finetune_status())
            finetune_timer = gr.Timer(value=5, active=False)
            finetune_timer.tick(fn=_get_finetune_status, outputs=[finetune_status_md])
            finetune_btn.click(
                fn=lambda: gr.Timer(active=True),
                outputs=[finetune_timer],
            )
