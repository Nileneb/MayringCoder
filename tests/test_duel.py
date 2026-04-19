"""Tests for model duel endpoint."""
from __future__ import annotations
import asyncio
import pytest

from src.api.server import DuelRequest, _run_duel
from src.api.job_queue import make_job as _make_job, _JOBS


@pytest.fixture(autouse=True)
def _clear_jobs():
    _JOBS.clear()
    yield
    _JOBS.clear()


def test_duel_request_validation():
    """Request model enforces required fields."""
    req = DuelRequest(task="hello", model_a="a", model_b="b")
    assert req.task == "hello"
    assert req.timeout == 180.0


def test_run_duel_executes_both_models(monkeypatch):
    """_run_duel runs both models sequentially and fills job fields."""
    calls: list[str] = []

    def _fake_task(**kw):
        calls.append(kw["model"])
        return f"response from {kw['model']}"

    monkeypatch.setattr("src.agents.pi.run_task_with_memory", _fake_task)

    job_id = _make_job("ws1")
    req = DuelRequest(task="test", model_a="mA", model_b="mB")
    asyncio.run(_run_duel(job_id, req, "ws1"))

    job = _JOBS[job_id]
    assert job["progress"] == "done"
    assert job["result_a"] == "response from mA"
    assert job["result_b"] == "response from mB"
    assert job["time_a_ms"] >= 0
    assert job["time_b_ms"] >= 0
    assert calls == ["mA", "mB"]


def test_run_duel_captures_exception_as_text(monkeypatch):
    """If a model raises, result contains the error string instead of crashing."""
    def _raising(**kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.agents.pi.run_task_with_memory", _raising)

    job_id = _make_job("ws1")
    req = DuelRequest(task="test", model_a="mA", model_b="mB")
    asyncio.run(_run_duel(job_id, req, "ws1"))

    job = _JOBS[job_id]
    assert "[Fehler]" in job["result_a"]
    assert "[Fehler]" in job["result_b"]
    assert job["progress"] == "done"
