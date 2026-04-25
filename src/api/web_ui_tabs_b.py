from __future__ import annotations

import gradio as gr

from src.api.web_ui_helpers import (
    _HAS_HTTPX,
    _api_post,
    _poll_job,
)

try:
    from src.github import parse_github_input, GitHubInputError
    import httpx as _httpx
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Router-Config Helper
# ---------------------------------------------------------------------------

def _save_router_config(*model_values) -> str:
    try:
        from src.model_router import ModelRouter
        from src.api.server import _router

        tasks = ModelRouter.TASKS
        for task, model_value in zip(tasks, model_values):
            _router.set_route(task, str(model_value).strip())

        from pathlib import Path
        _router.save_config(Path("config/model_routes.yaml"))
        return "✅ Router-Konfiguration gespeichert in `config/model_routes.yaml`"
    except Exception as e:
        return f"Fehler beim Speichern: {e}"


# ---------------------------------------------------------------------------
# Tab Builders
# ---------------------------------------------------------------------------

def build_analyse_tab(
    app: gr.Blocks,
    *,
    token_state: gr.State,
) -> None:
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
            analyse_second_opinion = gr.Textbox(
                label="Second Opinion (Modell, optional)",
                placeholder="z.B. claude-haiku-4-5-20251001",
                value="",
            )

        analyse_btn = gr.Button("Analyse starten", variant="primary")
        analyse_job_id = gr.Textbox(label="Job-ID", interactive=False)
        analyse_poll_btn = gr.Button("Status abfragen")
        analyse_output = gr.Markdown("")
        analyse_timer = gr.Timer(value=4, active=False)

        def _do_analyse(repo, mode, issues, memory, adversarial, llm_mode, second_opinion, token):
            if not token:
                return "", "Erst einloggen.", gr.Timer(active=False)
            try:
                gh = parse_github_input(repo)
            except GitHubInputError as exc:
                return "", f"❌ {exc}", gr.Timer(active=False)
            mode_map = {
                "Vollanalyse (Fehlersuche)": "analyze",
                "Inventar (Overview)": "overview",
                "Hot-Zones (Turbulenz)": "turbulence",
            }
            endpoint = mode_map.get(mode, "analyze")
            payload: dict = {
                "repo": gh.url,
                "adversarial": adversarial,
                "populate_memory": memory,
                "ingest_issues": issues,
                "llm": llm_mode,
            }
            if second_opinion.strip():
                payload["second_opinion"] = second_opinion.strip()
            r = _api_post(endpoint, payload, token)
            if "error" in r:
                return "", f"Fehler: {r['error']}", gr.Timer(active=False)
            job_id = r.get("job_id") or r.get("pid", "")
            return str(job_id), f"⏳ Gestartet (Job: {job_id or '?'})", gr.Timer(active=True)

        def _poll_and_maybe_stop(job_id, token):
            result = _poll_job(job_id, token)
            done = any(k in result for k in ("✅", "❌", "Fehler"))
            return result, gr.Timer(active=not done)

        analyse_btn.click(
            fn=_do_analyse,
            inputs=[analyse_repo, analyse_mode, analyse_issues, analyse_memory,
                    analyse_adversarial, analyse_llm, analyse_second_opinion, token_state],
            outputs=[analyse_job_id, analyse_output, analyse_timer],
        )
        analyse_poll_btn.click(
            fn=_poll_job,
            inputs=[analyse_job_id, token_state],
            outputs=[analyse_output],
        )
        analyse_timer.tick(
            fn=_poll_and_maybe_stop,
            inputs=[analyse_job_id, token_state],
            outputs=[analyse_output, analyse_timer],
        )


def build_duel_tab(
    app: gr.Blocks,
    *,
    ollama_available: bool,
    ollama_models: list[str],
    token_state: gr.State,
) -> None:
    with gr.Tab("Model-Duell"):
        gr.Markdown(
            "### Zwei Modelle vergleichen\n"
            "Derselbe Task läuft sequenziell auf beiden Modellen. "
            "Ergebnisse erscheinen side-by-side mit Laufzeit-Metriken."
        )
        with gr.Row():
            duel_model_a = gr.Dropdown(
                choices=ollama_models,
                value=ollama_models[0] if ollama_models else None,
                label="Modell A",
                interactive=ollama_available,
                scale=1,
            )
            duel_model_b = gr.Dropdown(
                choices=ollama_models,
                value=ollama_models[1] if len(ollama_models) >= 2 else (ollama_models[0] if ollama_models else None),
                label="Modell B",
                interactive=ollama_available,
                scale=1,
            )
        duel_repo_slug = gr.Textbox(
            label="Repo-Slug (optional, für Memory-Scope)",
            placeholder="z.B. Nileneb/MayringCoder",
        )
        duel_task = gr.Textbox(
            label="Task / Prompt",
            placeholder="z.B. 'Erkläre die Zusammenhänge zwischen CreditService und PhaseChainService'",
            lines=6,
        )
        duel_btn = gr.Button("Duell starten", variant="primary")
        duel_status = gr.Markdown("")
        duel_job_id = gr.Textbox(label="Job-ID", interactive=False, visible=True)

        gr.Markdown("### Ergebnisse")
        with gr.Row():
            with gr.Column():
                duel_label_a = gr.Markdown("**Modell A** — _noch kein Ergebnis_")
                duel_result_a = gr.Markdown("")
            with gr.Column():
                duel_label_b = gr.Markdown("**Modell B** — _noch kein Ergebnis_")
                duel_result_b = gr.Markdown("")

        duel_timer = gr.Timer(value=3, active=False)

        def _start_duel(model_a, model_b, task, repo_slug, token):
            if not token:
                return "", "Erst einloggen.", gr.Timer(active=False), "**Modell A**", "", "**Modell B**", ""
            if not task.strip():
                return "", "Task darf nicht leer sein.", gr.Timer(active=False), "**Modell A**", "", "**Modell B**", ""
            if not model_a or not model_b:
                return "", "Beide Modelle auswählen.", gr.Timer(active=False), "**Modell A**", "", "**Modell B**", ""
            normalized_slug: str | None = None
            if repo_slug.strip():
                try:
                    normalized_slug = parse_github_input(repo_slug).slug
                except GitHubInputError as exc:
                    return "", f"❌ {exc}", gr.Timer(active=False), "**Modell A**", "", "**Modell B**", ""
            payload = {
                "task": task,
                "model_a": model_a,
                "model_b": model_b,
                "repo_slug": normalized_slug,
            }
            r = _api_post("duel", payload, token)
            if "error" in r:
                return "", f"Fehler: {r['error']}", gr.Timer(active=False), "**Modell A**", "", "**Modell B**", ""
            job_id = str(r.get("job_id", ""))
            return (
                job_id,
                f"Gestartet (Job: `{job_id}`) — Modell A: {model_a} → danach Modell B: {model_b}",
                gr.Timer(active=True),
                f"**Modell A** · _{model_a}_ · läuft…",
                "",
                f"**Modell B** · _{model_b}_ · wartet…",
                "",
            )

        def _poll_duel(job_id, token):
            if not job_id:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.Timer(active=False)
            if not token or not _HAS_HTTPX:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.Timer(active=False)
            try:
                import src.api.web_ui_helpers as _h
                resp = _httpx.get(f"{_h._api_url}/jobs/{job_id}",
                                   headers={"Authorization": f"Bearer {token}"}, timeout=10.0)
                if resp.status_code != 200:
                    return (f"Fehler beim Polling: {resp.status_code}",
                            gr.update(), gr.update(), gr.update(), gr.update(), gr.Timer(active=False))
                job = resp.json()
            except Exception as exc:
                return f"Polling-Fehler: {exc}", gr.update(), gr.update(), gr.update(), gr.update(), gr.Timer(active=False)

            progress = job.get("progress", "")
            model_a = job.get("model_a", "Modell A")
            model_b = job.get("model_b", "Modell B")
            result_a = job.get("result_a", "")
            result_b = job.get("result_b", "")
            time_a = job.get("time_a_ms")
            time_b = job.get("time_b_ms")

            label_a = f"**Modell A** · _{model_a}_"
            if time_a is not None:
                label_a += f" · {time_a} ms · {len(result_a)} chars"
            elif progress == "running_a":
                label_a += " · läuft…"

            label_b = f"**Modell B** · _{model_b}_"
            if time_b is not None:
                label_b += f" · {time_b} ms · {len(result_b)} chars"
            elif progress == "running_b":
                label_b += " · läuft…"
            elif progress == "running_a":
                label_b += " · wartet…"

            status_line = {
                "running_a": f"Modell A läuft ({model_a})…",
                "running_b": f"Modell B läuft ({model_b})…",
                "done": f"Duell abgeschlossen (A: {time_a}ms, B: {time_b}ms)",
            }.get(progress, f"Status: {job.get('status', '')}")

            done = progress == "done" or job.get("status") == "finished"
            return status_line, label_a, result_a or "_(noch kein Output)_", label_b, result_b or "_(noch kein Output)_", gr.Timer(active=not done)

        duel_btn.click(
            fn=_start_duel,
            inputs=[duel_model_a, duel_model_b, duel_task, duel_repo_slug, token_state],
            outputs=[duel_job_id, duel_status, duel_timer, duel_label_a, duel_result_a, duel_label_b, duel_result_b],
        )
        duel_timer.tick(
            fn=_poll_duel,
            inputs=[duel_job_id, token_state],
            outputs=[duel_status, duel_label_a, duel_result_a, duel_label_b, duel_result_b, duel_timer],
        )


def build_settings_tab(app: gr.Blocks) -> None:
    with gr.Tab("Einstellungen"):
        gr.Markdown("## Modell-Router Konfiguration")
        gr.Markdown(
            "Weise jeder Aufgabe ein spezifisches Ollama-Modell zu. "
            "Änderungen sofort aktiv, via **Speichern** dauerhaft."
        )

        try:
            from src.model_router import ModelRouter
            from src.api.server import _router as _active_router
            _router_available = True
        except Exception:
            _active_router = None
            _router_available = False

        route_inputs = []
        task_labels = {
            "mayring_code": "Code-Analyse (mayring_code)",
            "mayring_social": "Sozialforschung (mayring_social)",
            "mayring_hybrid": "Hybrid-Modus (mayring_hybrid)",
            "vision": "Bild-Captioning (vision)",
            "analysis": "Allgemeine Analyse (analysis)",
            "embedding": "Embeddings (embedding)",
        }

        for task in (ModelRouter.TASKS if _router_available else []):
            current_model = _active_router.resolve(task) if _active_router else ""
            with gr.Row():
                gr.Markdown(f"**{task_labels.get(task, task)}**")
                inp = gr.Textbox(
                    value=current_model,
                    placeholder="model:tag oder leer (= ENV)",
                    show_label=False,
                    scale=3,
                )
                route_inputs.append(inp)

        if not _router_available:
            gr.Markdown("> Router nicht verfügbar.")

        with gr.Row():
            save_routes_btn = gr.Button("Speichern", variant="primary")
            routes_out = gr.Markdown("")

        if route_inputs:
            save_routes_btn.click(
                fn=_save_router_config,
                inputs=route_inputs,
                outputs=[routes_out],
            )
