"""In-memory job queue for background pipeline jobs."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent.parent
_JOBS: dict[str, dict] = {}


def make_job(workspace_id: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "started",
        "output": "",
        "workspace_id": workspace_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    return _JOBS.get(job_id)


def python_exe() -> str:
    p = str(_ROOT / ".venv" / "bin" / "python")
    return p if Path(p).exists() else "python"


async def run_checker_job(job_id: str, checker_args: list[str], workspace_id: str) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            python_exe(), "-m", "src.pipeline", *checker_args,
            "--workspace-id", workspace_id,
            cwd=str(_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        _JOBS[job_id]["status"] = "done" if proc.returncode == 0 else "error"
        _JOBS[job_id]["output"] = stdout.decode(errors="replace")
    except Exception as exc:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["output"] = str(exc)
