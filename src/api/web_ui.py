"""Gradio WebUI for MayringCoder Memory Layer.

Usage:
    python -m src.api.web_ui [--port 7860] [--ollama-url http://localhost:11434] [--api-url http://localhost:8080]

Tabs:
    Memory   — Suche + Source-Browser in einem View
    Ingest   — Repo / Text-Datei / Compact-Paste in einem Formular
    Training — Trainingspipeline + Langdock-Annotation
    Analyse  — Vollanalyse / Overview / Turbulenz mit Issues default-an
    Duell    — Zwei Modelle side-by-side vergleichen
    Feedback — Feedback + Benchmark + Chunk-Inspector + Training-Export
    Einstellungen — Modell-Router Konfiguration
"""

from __future__ import annotations

import argparse
import os

try:
    import gradio as gr
    _HAS_GRADIO = True
except ImportError as _e:
    raise SystemExit(
        "gradio is not installed. Run: pip install -r requirements-ui.txt"
    ) from _e

from src.ollama_client import check_ollama
from src.api.web_ui_helpers import (
    _MEMORY_READY,
    _IMPORT_ERROR,
    _status_html,
    _validate_token,
    _laravel_base,
    _laravel_headers,
    set_runtime_urls,
)
from src.api.web_ui_tabs import (
    build_memory_tab,
    build_ingest_tab,
    build_training_tab,
)
from src.api.web_ui_tabs_b import (
    build_analyse_tab,
    build_duel_tab,
    build_settings_tab,
)
from src.api.web_ui_tabs_c import build_feedback_tab

try:
    import httpx as _httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False


def build_app(ollama_url: str, api_url: str = "http://localhost:8080") -> gr.Blocks:
    set_runtime_urls(ollama_url, api_url)

    ollama_available, ollama_models = check_ollama(ollama_url)

    with gr.Blocks(title="MayringCoder") as app:

        _token_state = gr.State("")
        _workspace_state = gr.State("")

        _LOGGED_OUT_MD = (
            "🔒 **Nicht eingeloggt.** "
            "&nbsp;[→ Jetzt via app.linn.games einloggen]"
            "(https://app.linn.games/mayring/dashboard)"
        )
        login_status = gr.Markdown(_LOGGED_OUT_MD)

        def _auto_login(request: gr.Request):
            params = dict(request.query_params) if request.query_params else {}
            code = params.get("code", "").strip()
            if not code or not _HAS_HTTPX:
                return "", "", _LOGGED_OUT_MD

            try:
                base = os.environ.get("LARAVEL_INTERNAL_URL", "https://app.linn.games").rstrip("/")
                headers = {"Host": "app.linn.games"} if base.startswith("http://") else {}
                resp = _httpx.get(
                    f"{base}/api/mayring/token-exchange",
                    params={"code": code},
                    headers=headers,
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    token = resp.json().get("token", "").strip()
                    if token:
                        ok, result = _validate_token(token)
                        if ok:
                            return token, result, f"✅ Eingeloggt als Workspace `{result}`"
                        return "", "", f"⚠️ Token ungültig: {result}. {_LOGGED_OUT_MD}"
            except Exception as exc:
                return "", "", f"⚠️ Login-Fehler: {exc}. {_LOGGED_OUT_MD}"

            return "", "", _LOGGED_OUT_MD

        app.load(
            fn=_auto_login,
            outputs=[_token_state, _workspace_state, login_status],
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

        build_memory_tab(app, ollama_available=ollama_available, token_state=_token_state)
        build_ingest_tab(app, ollama_available=ollama_available, model_selector=model_selector, token_state=_token_state)
        build_training_tab(app)
        build_analyse_tab(app, token_state=_token_state)
        build_duel_tab(app, ollama_available=ollama_available, ollama_models=ollama_models, token_state=_token_state)
        build_feedback_tab(app, token_state=_token_state)
        build_settings_tab(app)

    return app


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
        pwa=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
