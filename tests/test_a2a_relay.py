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


def test_relay_executor_streams_result_artifact(db):
    """execute() must enqueue the cloud job AND bridge the out-of-process worker's
    completion into the event stream as a result artifact (streaming=true) — else a
    streaming client (Langdock) only ever sees the initial 'working' event."""
    task_id = _Ctx.task_id

    async def scenario():
        ex = RelayAgentExecutor(workspace_id="ws1", model="qwen3.5:9b",
                                capability="research", db_path=db,
                                poll_interval=0.02, keepalive_s=0.05)
        q = _Q()

        async def completer():
            for _ in range(200):
                await asyncio.sleep(0.02)
                if any(j.job_id == task_id for j in pi_jobs.list_recent(db_path=db)):
                    pi_jobs.claim_cloud_next("wkr", capabilities=["research"],
                                             workspace_id="ws1", db_path=db)
                    pi_jobs.complete_job(task_id, {"text": "STREAM RESULT 77"}, db_path=db)
                    return

        await asyncio.wait_for(asyncio.gather(ex.execute(_Ctx(), q), completer()), timeout=10)
        return q

    q = asyncio.run(scenario())
    # job was created with job_id == task_id
    assert any(j.job_id == task_id for j in pi_jobs.list_recent(db_path=db))
    # the result reached the stream as an artifact (not just a status note)
    blob = " ".join(str(e) for e in q.events)
    assert "STREAM RESULT 77" in blob, f"result not streamed as artifact: {blob[:400]}"
    # WHY(2026-05-31): the terminal completed status-update MUST also carry the
    # result text (Langdock renders only the final status message, not the
    # artifact event). Find the completed event and assert its message has the text.
    completed = [e for e in q.events if "COMPLETED" in str(e) or "completed" in str(e)]
    assert completed, "no completed event emitted"
    assert any("STREAM RESULT 77" in str(e) for e in completed), \
        "completed event must carry the result in its status message, not just the artifact"
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
