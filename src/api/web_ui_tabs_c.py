from __future__ import annotations

import gradio as gr

from src.api.web_ui_helpers import (
    _HAS_HTTPX,
    _api_post,
    _poll_job,
    _do_feedback,
    _training_files,
)

try:
    import httpx as _httpx
except ImportError:
    pass


def build_feedback_tab(
    app: gr.Blocks,
    *,
    token_state: gr.State,
) -> None:
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
                inputs=[feedback_chunk_id, feedback_signal, feedback_label, token_state],
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
                inputs=[bench_top_k, bench_repo, token_state],
                outputs=[bench_job_id, bench_output],
            )
            bench_poll_btn.click(
                fn=_poll_job,
                inputs=[bench_job_id, token_state],
                outputs=[bench_output],
            )

        with gr.Accordion("Chunk-Inspector", open=False):
            gr.Markdown(
                "Chunk-Details anzeigen — nutzt `GET /memory/explain/{id}` und `GET /memory/chunks/{source_id}`. "
                "Kein Black-Box-Feeling: du siehst genau was, woher und wie es bewertet wurde."
            )
            with gr.Row():
                inspect_chunk_id = gr.Textbox(label="Chunk-ID", scale=3)
                inspect_btn = gr.Button("Details laden", scale=1)
            inspect_output = gr.Markdown("")

            with gr.Row():
                inspect_source_id = gr.Textbox(label="Source-ID (alternativ: alle Chunks einer Quelle)", scale=3)
                inspect_source_btn = gr.Button("Chunks der Quelle", scale=1)
            inspect_source_table = gr.Dataframe(
                headers=["chunk_id", "chunk_level", "ordinal", "is_active", "vorschau"],
                datatype=["str"] * 5,
                interactive=False,
            )

            with gr.Row():
                reindex_source_id = gr.Textbox(
                    label="Source-ID (leer = alle Chunks) — ACHTUNG: kann lange dauern",
                    scale=3,
                )
                reindex_btn = gr.Button("Reindex starten", variant="secondary", scale=1)
            reindex_status = gr.Markdown("")

            with gr.Row():
                invalidate_source_id = gr.Textbox(label="Source-ID (Chunks deaktivieren)", scale=3)
                invalidate_btn = gr.Button("Invalidieren", variant="secondary", scale=1)
            invalidate_status = gr.Markdown("")

            def _do_inspect_chunk(chunk_id, token):
                if not token:
                    return "Erst einloggen."
                if not chunk_id.strip():
                    return "Chunk-ID fehlt."
                if not _HAS_HTTPX:
                    return "httpx nicht verfügbar."
                try:
                    import src.api.web_ui_helpers as _h
                    resp = _httpx.get(
                        f"{_h._api_url}/memory/explain/{chunk_id.strip()}",
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10.0,
                    )
                    if resp.status_code == 404:
                        return "Chunk nicht gefunden."
                    if resp.status_code != 200:
                        return f"Fehler: HTTP {resp.status_code}"
                    data = resp.json()
                except Exception as exc:
                    return f"Fehler: {exc}"
                lines = [
                    f"**Chunk:** `{data.get('chunk_id')}`",
                    f"**Memory-Key:** `{data.get('memory_key')}`",
                    f"**Source-ID:** `{data.get('source_id')}`",
                    f"**Level / Ordinal:** `{data.get('chunk_level')}` / `{data.get('ordinal')}`",
                    f"**Kategorien:** `{', '.join(data.get('category_labels', [])) or '—'}`",
                    f"**Aktiv:** `{data.get('is_active')}`",
                    f"**Erstellt:** `{data.get('created_at')}`",
                    f"**Quality-Score:** `{data.get('quality_score')}`",
                ]
                src = data.get("source", {})
                if src:
                    lines.append(
                        f"\n**Source:** `{src.get('path')}` @ `{src.get('branch') or '-'}` · "
                        f"`{src.get('commit') or '-'}` · `{src.get('content_hash', '')[:20]}`"
                    )
                return "\n\n".join(lines)

            def _do_list_chunks(source_id, token):
                if not token or not source_id.strip() or not _HAS_HTTPX:
                    return []
                try:
                    import src.api.web_ui_helpers as _h
                    resp = _httpx.get(
                        f"{_h._api_url}/memory/chunks/{source_id.strip()}",
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10.0,
                    )
                    if resp.status_code != 200:
                        return []
                    data = resp.json()
                except Exception:
                    return []
                return [
                    [
                        c.get("chunk_id", ""),
                        c.get("chunk_level", ""),
                        str(c.get("ordinal", "")),
                        str(c.get("is_active", "")),
                        (c.get("text", "") or "")[:120],
                    ]
                    for c in data.get("chunks", [])
                ]

            def _do_reindex(source_id, token):
                if not token or not _HAS_HTTPX:
                    return "Erst einloggen oder httpx fehlt."
                payload: dict = {"source_id": source_id.strip() or None}
                try:
                    import src.api.web_ui_helpers as _h
                    resp = _httpx.post(
                        f"{_h._api_url}/memory/reindex",
                        json=payload,
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=600.0,
                    )
                    data = resp.json()
                except Exception as exc:
                    return f"Fehler: {exc}"
                if resp.status_code != 200 or "error" in data:
                    return f"Fehler: {data.get('error') or resp.status_code}"
                return f"Reindexed: {data.get('reindexed_count')} chunks, {data.get('errors')} errors"

            def _do_invalidate(source_id, token):
                if not token or not _HAS_HTTPX or not source_id.strip():
                    return "Erst einloggen oder Source-ID fehlt."
                try:
                    import src.api.web_ui_helpers as _h
                    resp = _httpx.post(
                        f"{_h._api_url}/memory/invalidate",
                        json={"source_id": source_id.strip()},
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10.0,
                    )
                    data = resp.json()
                except Exception as exc:
                    return f"Fehler: {exc}"
                if resp.status_code != 200 or "error" in data:
                    return f"Fehler: {data.get('error') or resp.status_code}"
                return f"Deaktiviert: {data.get('deactivated_count')} chunks fur `{data.get('source_id')}`"

            inspect_btn.click(
                fn=_do_inspect_chunk,
                inputs=[inspect_chunk_id, token_state],
                outputs=[inspect_output],
            )
            inspect_source_btn.click(
                fn=_do_list_chunks,
                inputs=[inspect_source_id, token_state],
                outputs=[inspect_source_table],
            )
            reindex_btn.click(
                fn=_do_reindex,
                inputs=[reindex_source_id, token_state],
                outputs=[reindex_status],
            )
            invalidate_btn.click(
                fn=_do_invalidate,
                inputs=[invalidate_source_id, token_state],
                outputs=[invalidate_status],
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
