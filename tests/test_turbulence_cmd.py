"""Tests for --mode turbulence --model flag behaviour (Issue #16)."""
from __future__ import annotations

import argparse
from unittest.mock import patch, MagicMock


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        model=None, llm=False, full=False, use_overview_cache=False,
        workspace_id="default",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_model_flag_implies_llm():
    """--model without --llm must implicitly enable LLM (args.llm=True)."""
    from src.cli import _cmd_turbulence

    args = _make_args(model="qwen3:latest", llm=False)

    with (
        patch("src.cli.resolve_model", return_value="qwen3:latest"),
        patch("src.cli.run_turbulence") as mock_run,
    ):
        _cmd_turbulence(args, "https://github.com/x/y", "http://localhost:11434")

    assert args.llm is True
    mock_run.assert_called_once()
    _, _, _, passed_model = mock_run.call_args.args
    assert passed_model == "qwen3:latest"


def test_explicit_llm_model_passes_through():
    """--llm --model must pass the model to run_turbulence unchanged."""
    from src.cli import _cmd_turbulence

    args = _make_args(model="mistral:7b", llm=True)

    with (
        patch("src.cli.resolve_model", return_value="mistral:7b"),
        patch("src.cli.run_turbulence") as mock_run,
    ):
        _cmd_turbulence(args, "https://github.com/x/y", "http://localhost:11434")

    _, _, _, passed_model = mock_run.call_args.args
    assert passed_model == "mistral:7b"


def test_no_model_no_llm_uses_env_fallback(monkeypatch):
    """Without --model and without --llm the TURB_MODEL env var is used."""
    from src.cli import _cmd_turbulence

    monkeypatch.setenv("TURB_MODEL", "env-model:1b")
    args = _make_args(model=None, llm=False)

    with patch("src.cli.run_turbulence") as mock_run:
        _cmd_turbulence(args, "https://github.com/x/y", "http://localhost:11434")

    _, _, _, passed_model = mock_run.call_args.args
    assert passed_model == "env-model:1b"
