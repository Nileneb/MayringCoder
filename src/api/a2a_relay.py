"""A2A research-relay: public Agent2Agent gateway over the cloud pi_jobs queue.

Langdock (or any A2A client) hits this gateway on mcp.linn.games. `message/send`
enqueues a cloud-scoped pi_job (capability_required="research") and returns an
async A2A task (task_id == job_id). A laptop research-worker pulls the job, runs
locally, and reports back via /pi_task_complete_cloud. `tasks/get` maps the
pi_jobs status → A2A task state via PiJobsTaskStore.

See docs/superpowers/specs/2026-05-30-a2a-research-worker-relay-design.md.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from a2a.helpers import new_task, new_text_artifact, new_text_part
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import TaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Message,
    Task,
    TaskState,
)
from a2a.utils import TransportProtocol
from mayring_pi_agent import pi_jobs

# JSON-RPC mounts on /a2a (NOT /) so it doesn't collide with the MCP default
# nginx upstream at `location /`.
_RPC_URL = "/a2a"

_STATE = {
    "queued": TaskState.TASK_STATE_SUBMITTED,
    "running": TaskState.TASK_STATE_WORKING,
    "completed": TaskState.TASK_STATE_COMPLETED,
    "failed": TaskState.TASK_STATE_FAILED,
}
_AGENT_ROLE = Message.DESCRIPTOR.fields_by_name["role"].enum_type.values_by_name["ROLE_AGENT"].number


def _agent_message(job_id: str, text: str) -> Message:
    return Message(message_id=job_id, context_id=job_id, task_id=job_id,
                   role=_AGENT_ROLE, parts=[new_text_part(text)])


def _result_text(result_json: str) -> str:
    """Worker stores the result as {"text": ...} JSON; tolerate a raw string."""
    try:
        return json.loads(result_json).get("text", result_json)
    except (ValueError, AttributeError):
        return result_json


class PiJobsTaskStore(TaskStore):
    """Read-through TaskStore: pi_jobs is the authoritative state, so save/delete
    are no-ops and get() reconstructs the A2A Task from the job row."""

    def __init__(self, db_path: Path | None = None, workspace_id: str | None = None):
        self._db_path = db_path
        self._ws = workspace_id

    async def get(self, task_id: str, context=None) -> Task | None:
        # WHY(security): scope by workspace so a guessed task_id cannot leak
        # another tenant's job result.
        job = pi_jobs.get_job(task_id, workspace_id=self._ws, db_path=self._db_path)
        if job is None:
            return None
        state = _STATE.get(job.status, TaskState.TASK_STATE_WORKING)
        task = new_task(job.job_id, job.job_id, state)
        if job.status == "completed" and job.result_json:
            text = _result_text(job.result_json)
            task.status.message.CopyFrom(_agent_message(job.job_id, text))
            # WHY(2026-05-31): A2A clients (Langdock) read the deliverable from
            # `task.artifacts`, NOT `status.message` (that's a status note). Without
            # an artifact the client sees "completed" but no research text. Carry the
            # result as a text artifact so the payload is actually retrievable.
            task.artifacts.append(new_text_artifact(
                name="research_result", text=text,
                description="Research result from the MayringCoder worker.",
                artifact_id=job.job_id,
            ))
        elif job.status == "failed" and job.error:
            task.status.message.CopyFrom(_agent_message(job.job_id, job.error))
        return task

    async def save(self, task: Task, context=None) -> None:
        return None

    async def delete(self, task_id: str, context=None) -> None:
        return None

    async def list(self, context=None):
        return []


class RelayAgentExecutor(AgentExecutor):
    """Enqueues a cloud research job (instead of running inline) and returns an
    async A2A task whose id == the pi_jobs job_id."""

    def __init__(self, workspace_id: str, model: str, capability: str = "research",
                 timeout_s: float = 600.0, db_path: Path | None = None,
                 poll_interval: float = 3.0, keepalive_s: float = 20.0):
        self._ws = workspace_id
        self._model = model
        self._cap = capability
        self._timeout_s = timeout_s
        self._db_path = db_path
        self._poll_interval = poll_interval
        self._keepalive_s = keepalive_s

    async def execute(self, context, event_queue) -> None:
        text = context.get_user_input()
        # The handler assigns the A2A task_id; pin the cloud job_id to it so the
        # worker's completion and a client's tasks/get(task_id) resolve to the
        # same job.
        task_id = context.task_id
        pi_jobs.insert_cloud_job(
            text, workspace_id=self._ws, model=self._model,
            capability_required=self._cap, timeout_s=self._timeout_s,
            job_id=task_id, db_path=self._db_path,
        )
        # WHY(2026-06-01 langdock): Langdock uses message/send (card streaming=false)
        # and renders the agent's MESSAGE result. When we returned a Task it only
        # showed "completed the task successfully" with NO text (the artifact +
        # status message were ignored) and long research even tripped a generic
        # "blocked by security measures". The result is produced out-of-process by
        # the laptop worker, so block here until it finishes, then return the result
        # as a single agent text Message — which message/send surfaces as
        # `result.kind=message` for the client to render. tasks/get still works:
        # PiJobsTaskStore reconstructs the Task from the pi_jobs row.
        result_text = await self._await_result(task_id)
        await event_queue.enqueue_event(_agent_message(task_id, result_text))

    async def _await_result(self, task_id: str) -> str:
        """Block until the out-of-process worker finishes; return its result text
        (or a human-readable error/timeout note — never silent)."""
        start = time.monotonic()
        while True:
            await asyncio.sleep(self._poll_interval)
            job = pi_jobs.get_job(task_id, workspace_id=self._ws, db_path=self._db_path)
            if job is not None and job.status == "completed":
                return _result_text(job.result_json or "")
            if job is not None and job.status == "failed":
                return job.error or "Recherche fehlgeschlagen."
            if time.monotonic() - start > self._timeout_s:
                return "Zeitüberschreitung: Der Worker hat die Recherche nicht rechtzeitig abgeschlossen."

    async def cancel(self, context, event_queue) -> None:
        return None


def _research_card(base_url: str, model: str) -> AgentCard:
    url = base_url.rstrip("/") + _RPC_URL
    card = AgentCard(
        name="MayringCoder Research Worker",
        description=(
            f"Deep-research agent ({model}) — web search (SearXNG) + cloud memory, "
            "laptop-powered, async. Long-running tasks welcome."
        ),
        version="0.1.0",
        # WHY(2026-06-01 langdock-block): streaming=False. Langdock's OWN reference
        # A2A agent (github.com/Langdock/langdock-adk-a2a-agent agent_card.json)
        # declares streaming:false → its client uses message/send (one blocking
        # response), NOT message/stream (SSE). With streaming:true Langdock opened
        # an SSE stream it then rejected with a generic "blocked by security
        # measures" — even though our stream was spec-valid. execute() already
        # blocks via _bridge until the laptop worker completes, so message/send
        # returns the finished Task (artifact + status message) in one response,
        # matching Langdock's synchronous-agent expectation.
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        supported_interfaces=[AgentInterface(url=url, protocol_binding=TransportProtocol.JSONRPC)],
        skills=[AgentSkill(
            id="deep-research", name="Deep Research",
            description="Mehrstufige Web- + Memory-Recherche, async (lange Aufträge).",
            tags=["research", "web", "memory"],
        )],
    )
    # WHY(2026-05-30): the /a2a JSON-RPC endpoint is JWT-gated (nginx auth_request),
    # but the agent-card is public. A2A clients (Langdock) only attach their stored
    # credential to RPC calls if the card DECLARES a security scheme — otherwise they
    # fetch the public card ("connection ok") yet send message/send WITHOUT the Bearer
    # → 401. Declare HTTP Bearer (JWT) so the client passes the token through.
    card.security_schemes["bearer"].http_auth_security_scheme.scheme = "bearer"
    card.security_schemes["bearer"].http_auth_security_scheme.bearer_format = "JWT"
    req = card.security_requirements.add()
    req.schemes["bearer"].list.extend([])  # bearer required, no scopes
    return card


def register_a2a_relay(app, *, base_url: str, model: str, workspace_id: str = "default",
                       db_path: Path | None = None) -> AgentCard:
    card = _research_card(base_url, model)
    executor = RelayAgentExecutor(workspace_id=workspace_id, model=model, db_path=db_path)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=PiJobsTaskStore(db_path=db_path, workspace_id=workspace_id),
        agent_card=card,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        # WHY(2026-05-30): a2a-sdk 1.1.0 defaults to v1 method names; standard A2A
        # clients (Langdock) send the v0.3 spec methods (`message/send`, `tasks/get`).
        # Without compat the dispatcher returns -32601 "Method not found" for ALL of
        # them → connection test fails. Enabling v0.3 compat serves both on /a2a.
        jsonrpc_routes=create_jsonrpc_routes(handler, _RPC_URL, enable_v0_3_compat=True),
    )
    return card
