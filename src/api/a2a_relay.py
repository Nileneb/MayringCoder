"""A2A research-relay: public Agent2Agent gateway over the cloud pi_jobs queue.

Langdock (or any A2A client) hits this gateway on mcp.linn.games. `message/send`
enqueues a cloud-scoped pi_job (capability_required="research") and returns an
async A2A task (task_id == job_id). A laptop research-worker pulls the job, runs
locally, and reports back via /pi_task_complete_cloud. `tasks/get` maps the
pi_jobs status → A2A task state via PiJobsTaskStore.

See docs/superpowers/specs/2026-05-30-a2a-research-worker-relay-design.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from a2a.helpers import new_task, new_text_part
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import TaskStore, TaskUpdater
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


class PiJobsTaskStore(TaskStore):
    """Read-through TaskStore: pi_jobs is the authoritative state, so save/delete
    are no-ops and get() reconstructs the A2A Task from the job row."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path

    async def get(self, task_id: str, context=None) -> Task | None:
        job = pi_jobs.get_job(task_id, db_path=self._db_path)
        if job is None:
            return None
        state = _STATE.get(job.status, TaskState.TASK_STATE_WORKING)
        task = new_task(job.job_id, job.job_id, state)
        if job.status == "completed" and job.result_json:
            try:
                text = json.loads(job.result_json).get("text", job.result_json)
            except (ValueError, AttributeError):
                text = job.result_json
            task.status.message.CopyFrom(_agent_message(job.job_id, text))
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
                 timeout_s: float = 600.0, db_path: Path | None = None):
        self._ws = workspace_id
        self._model = model
        self._cap = capability
        self._timeout_s = timeout_s
        self._db_path = db_path

    async def execute(self, context, event_queue) -> None:
        text = context.get_user_input()
        # The handler assigns the A2A task_id; the executor must use it (it cannot
        # override). Pin the cloud job_id to it so the worker's completion and the
        # client's tasks/get(task_id) both resolve to the same job.
        task_id = context.task_id
        context_id = context.context_id
        pi_jobs.insert_cloud_job(
            text, workspace_id=self._ws, model=self._model,
            capability_required=self._cap, timeout_s=self._timeout_s,
            job_id=task_id, db_path=self._db_path,
        )
        await event_queue.enqueue_event(
            new_task(task_id, context_id, TaskState.TASK_STATE_SUBMITTED)
        )
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()

    async def cancel(self, context, event_queue) -> None:
        return None


def _research_card(base_url: str, model: str) -> AgentCard:
    url = base_url.rstrip("/") + _RPC_URL
    return AgentCard(
        name="MayringCoder Research Worker",
        description=(
            f"Deep-research agent ({model}) — web search (SearXNG) + cloud memory, "
            "laptop-powered, async. Long-running tasks welcome."
        ),
        version="0.1.0",
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


def register_a2a_relay(app, *, base_url: str, model: str, workspace_id: str = "default",
                       db_path: Path | None = None) -> AgentCard:
    card = _research_card(base_url, model)
    executor = RelayAgentExecutor(workspace_id=workspace_id, model=model, db_path=db_path)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=PiJobsTaskStore(db_path=db_path),
        agent_card=card,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, _RPC_URL),
    )
    return card
