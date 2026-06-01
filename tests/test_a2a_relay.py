import asyncio

import pytest
from mayring_core.memory.store import init_memory_db
from mayring_pi_agent import pi_jobs

from src.api.a2a_relay import PiJobsTaskStore, RelayAgentExecutor, register_a2a_relay


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "jobs.db"
    init_memory_db(p).close()
    return p


def test_taskstore_maps_completed_job(db):
    job = pi_jobs.insert_cloud_job("recherchiere X", workspace_id="ws1",
                                   capability_required="research", db_path=db)
    # claim → running, then complete (complete_job only flips a 'running' row)
    pi_jobs.claim_cloud_next("wkr", capabilities=["research"], workspace_id="ws1", db_path=db)
    pi_jobs.complete_job(job.job_id, {"text": "ERGEBNIS 42"}, db_path=db)
    store = PiJobsTaskStore(db_path=db)
    task = asyncio.run(store.get(job.job_id, None))
    assert task is not None
    assert task.id == job.job_id
    from a2a.types import TaskState
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert "ERGEBNIS 42" in str(task)
    # A2A clients (Langdock) read the deliverable from artifacts, not status.message.
    # Without this the client sees "completed" but no research text.
    assert task.artifacts, "completed task must carry the result as an artifact"
    assert "ERGEBNIS 42" in task.artifacts[0].parts[0].text


def test_taskstore_unknown_returns_none(db):
    store = PiJobsTaskStore(db_path=db)
    assert asyncio.run(store.get("nope", None)) is None


def test_taskstore_scopes_to_workspace(db):
    # A job in workspace ws1 must NOT be readable by a store pinned to ws2
    # (else a guessed task_id leaks cross-tenant results).
    job = pi_jobs.insert_cloud_job("geheim", workspace_id="ws1",
                                   capability_required="research", db_path=db)
    store_other = PiJobsTaskStore(db_path=db, workspace_id="ws2")
    assert asyncio.run(store_other.get(job.job_id, None)) is None
    store_own = PiJobsTaskStore(db_path=db, workspace_id="ws1")
    assert asyncio.run(store_own.get(job.job_id, None)) is not None


class _Q:
    def __init__(self):
        self.events = []

    async def enqueue_event(self, e):
        self.events.append(e)


class _Ctx:
    task_id = "a2a-task-xyz"
    context_id = "ctx"
    current_task = None

    def get_user_input(self, d="\n"):
        return "recherchiere Quantencomputing"


def test_relay_executor_returns_result_message(db):
    """execute() must enqueue the cloud job, block until the out-of-process worker
    completes, and return the result as a single agent MESSAGE (not a Task).

    WHY(2026-06-01): Langdock uses message/send (card streaming=false) and renders
    the agent's message result. A Task result only showed "completed" with no text.
    """
    task_id = _Ctx.task_id

    async def scenario():
        ex = RelayAgentExecutor(workspace_id="ws1", model="qwen3.5:9b",
                                capability="research", db_path=db,
                                poll_interval=0.02)
        q = _Q()

        async def completer():
            for _ in range(200):
                await asyncio.sleep(0.02)
                if any(j.job_id == task_id for j in pi_jobs.list_recent(db_path=db)):
                    pi_jobs.claim_cloud_next("wkr", capabilities=["research"],
                                             workspace_id="ws1", db_path=db)
                    pi_jobs.complete_job(task_id, {"text": "SEND RESULT 77"}, db_path=db)
                    return

        await asyncio.wait_for(asyncio.gather(ex.execute(_Ctx(), q), completer()), timeout=10)
        return q

    q = asyncio.run(scenario())
    # job was created with job_id == task_id
    assert any(j.job_id == task_id for j in pi_jobs.list_recent(db_path=db))
    # exactly one agent Message carrying the result text (no Task/status events)
    assert len(q.events) == 1, f"expected one message event, got {len(q.events)}: {q.events}"
    from a2a.types import Message
    msg = q.events[0]
    assert isinstance(msg, Message), f"executor must return a Message, got {type(msg)}"
    assert "SEND RESULT 77" in str(msg), f"result text missing: {str(msg)[:300]}"
    job = next(j for j in pi_jobs.list_recent(db_path=db) if j.job_id == task_id)
    assert job.scope == "cloud" and job.capability_required == "research"
    assert "Quantencomputing" in job.task_text


def test_agent_card_served_with_research_skill(db):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    register_a2a_relay(app, base_url="http://testserver", model="qwen3.5:9b", db_path=db)
    r = TestClient(app).get("/.well-known/agent-card.json")
    assert r.status_code == 200
    body = r.json()
    assert any(s["id"] == "deep-research" for s in body["skills"])
    # WHY(2026-06-01): Langdock's reference agent uses streaming:false → its client
    # calls message/send (blocking). streaming:true made it open an SSE stream it
    # rejected with "blocked by security measures". Guard the flag.
    assert body["capabilities"]["streaming"] is False, \
        "card must declare streaming:false so Langdock uses message/send"
    # Card MUST declare HTTP Bearer so A2A clients (Langdock) attach the token to
    # RPC calls — else the /a2a JWT gate 401s while card discovery looks fine.
    schemes = body.get("securitySchemes") or body.get("security_schemes") or {}
    assert "bearer" in schemes, "agent-card must declare a bearer security scheme"
    assert body.get("security") or body.get("securityRequirements"), "card must require auth"


def test_rpc_dispatches_v0_3_methods(db):
    """Standard A2A clients (Langdock) send v0.3 method names. a2a-sdk 1.1.0 defaults
    to v1 names → every classic method 404s as -32601 unless v0.3 compat is enabled.
    Regression guard for the silent 'Method not found' that broke Langdock connect.
    Uses tasks/get (non-blocking) — message/send now blocks on the async worker."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    register_a2a_relay(app, base_url="http://testserver", model="qwen3.5:9b", db_path=db)
    r = TestClient(app).post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": 1, "method": "tasks/get",
              "params": {"id": "no-such-task"}},
    )
    body = r.json()
    # The method must be FOUND (dispatched). -32601 = "Method not found" = compat off.
    assert body.get("error", {}).get("code") != -32601, f"v0.3 tasks/get not routed: {body}"
