"""In-memory job queue for background pipeline jobs.

Live-Progress: ``run_checker_job`` liest den Subprocess-Output Zeile für
Zeile (statt erst am Ende via ``proc.communicate()``). tqdm-Progress-
Zeilen werden geparst und unter ``_JOBS[id]["progress"]`` abgelegt — der
Client pollt via GET /jobs/{id} und sieht Fortschritt statt stoischer
"started"-Schweigminuten.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent.parent

# Persistent shadow of _JOBS so the dashboard's job history doesn't reset
# every time the API container restarts. Atomic write (temp + rename) on
# every status change; load at module import.
_JOBS_STATE_FILE = Path(
    os.environ.get("MAYRING_JOBS_STATE", str(_ROOT / "cache" / "jobs_state.json"))
)
_JOBS_LOCK = threading.Lock()


def _load_jobs() -> dict[str, dict]:
    try:
        if _JOBS_STATE_FILE.exists():
            return json.loads(_JOBS_STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        # Corrupted state shouldn't take the API down — start fresh, the
        # broken file gets overwritten on the next save.
        pass
    return {}


def _save_jobs() -> None:
    """Atomic write of _JOBS to disk. Best-effort; never raises into callers."""
    try:
        _JOBS_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _JOBS_STATE_FILE.with_suffix(_JOBS_STATE_FILE.suffix + ".tmp")
        with _JOBS_LOCK:
            tmp.write_text(json.dumps(_JOBS, default=str), encoding="utf-8")
            tmp.replace(_JOBS_STATE_FILE)
    except OSError:
        pass


_JOBS: dict[str, dict] = _load_jobs()


# tqdm default format example:
#   "Chunks embedden:  45%|████▌     | 9/20 [00:05<00:06,  1.74chunk/s]"
# Wir matchen: label, percent, current, total (optional: rate).
_TQDM_RE = re.compile(
    r"(?P<label>\S[^:\n\r]*?):\s*"
    r"(?P<pct>\d+)%\|[^|]*\|\s*"
    r"(?P<current>\d+)\s*/\s*(?P<total>\d+)"
    r"(?:\s*\[(?P<time>[^\]]+)\])?"
)

# "[populate-memory] 207 Dateien gefunden" → total signal fürs progress-Label
_FILECOUNT_RE = re.compile(r"\[populate-memory\]\s+(\d+)\s+Dateien gefunden")

# "[POPULATE-PROGRESS] 42/207" → per-file progress (#275). The populate loop's
# tqdm writes \r-only frames readline() can't see; this newline-terminated marker
# is the runner-parseable progress signal.
_POPULATE_PROGRESS_RE = re.compile(
    r"\[POPULATE-PROGRESS\]\s+(?P<current>\d+)\s*/\s*(?P<total>\d+)"
)

# "[STAGE] fetch_repo done files=274" → stages dict
_STAGE_RE = re.compile(r"\[STAGE\]\s+(?P<name>\S+)\s+(?P<detail>.*)")


def make_job(workspace_id: str, repo: str | None = None, source: str = "",
             head_sha: str | None = None) -> str:
    job_id = str(uuid.uuid4())[:8]
    record: dict = {
        "job_id": job_id,
        "status": "started",
        "output": "",
        "progress": None,
        "workspace_id": workspace_id,
        # WHY(#253): provenance tag so the job-history UI can default-filter
        # smoke-triggered jobs (source="smoke") out of the noise. Empty = real.
        "source": source,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if head_sha is not None:
        # WHY(lost-update guard): the commit sha that triggered this populate. The
        # ingest clones HEAD-at-run; if HEAD moved past this sha during the (long)
        # run, debounce swallowed those pushes → re-ingest once. See jobs.py.
        record["head_sha"] = head_sha
    if repo is not None:
        # WHY(repo-watching): include repo BEFORE _save_jobs so any worker that
        # reads the shared file sees it for cross-worker debounce.
        record["repo"] = repo
    _JOBS[job_id] = record
    _save_jobs()
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    job = _JOBS.get(job_id)
    if job is not None:
        return job
    # Cross-worker (uvicorn --workers): a job created/updated by ANOTHER worker
    # lives only in that process's _JOBS. _save_jobs writes the whole registry
    # atomically (tmp+rename) on every change to the shared-volume file, so
    # re-read it to find jobs this process didn't make. WHY(smoke-fix 2026-05-24):
    # without this /jobs/{id} 404'd across workers → pipeline_stage_observability
    # red after the move to --workers 4.
    return _load_jobs().get(job_id)


def python_exe() -> str:
    p = str(_ROOT / ".venv" / "bin" / "python")
    return p if Path(p).exists() else "python"


def _parse_progress_line(line: str) -> dict | None:
    """Return {label, pct, current, total, eta} if the line looks like a
    tqdm progress update, else None.
    """
    m = _TQDM_RE.search(line)
    if not m:
        return None
    return {
        "label":   m.group("label").strip(),
        "pct":     int(m.group("pct")),
        "current": int(m.group("current")),
        "total":   int(m.group("total")),
        "eta":     (m.group("time") or "").strip(),
    }


def _parse_populate_progress(line: str) -> dict | None:
    """Return a progress dict for a ``[POPULATE-PROGRESS] cur/total`` marker
    (#275), else None. The populate loop emits this newline-terminated marker
    because its tqdm bar uses \\r-only frames the line reader never sees.
    """
    pm = _POPULATE_PROGRESS_RE.search(line)
    if not pm:
        return None
    cur, tot = int(pm.group("current")), int(pm.group("total"))
    return {
        "label":   "populate-memory",
        "pct":     int(cur * 100 / tot) if tot else 0,
        "current": cur,
        "total":   tot,
        "eta":     "",
    }


async def run_checker_job(job_id: str, checker_args: list[str], workspace_id: str) -> None:
    try:
        # WHY(#1): route the ingest subprocess' generate-load through the central
        # PiQueue (/pi/run) instead of direct Ollama. The subprocess runs in the same
        # container as the API, so localhost:8090 reaches it; MCP_SERVICE_TOKEN auths.
        sub_env = {
            **os.environ,
            "MAYRING_GENERATE_VIA_QUEUE": "1",
            "MAYRING_API_URL": os.getenv("MAYRING_API_URL", "http://localhost:8090"),
        }
        proc = await asyncio.create_subprocess_exec(
            python_exe(), "-m", "src.pipeline", *checker_args,
            "--workspace-id", workspace_id,
            cwd=str(_ROOT),
            env=sub_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        chunks: list[str] = []
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode(errors="replace")
            chunks.append(line)
            # tqdm often writes \r-updates; split and take the last segment.
            last_segment = line.split("\r")[-1]
            progress = _parse_progress_line(last_segment)
            if progress is not None:
                _JOBS[job_id]["progress"] = progress
                continue
            pop_progress = _parse_populate_progress(last_segment)
            if pop_progress is not None:
                _JOBS[job_id]["progress"] = pop_progress
                continue
            sm = _STAGE_RE.search(last_segment)
            if sm:
                if "stages" not in _JOBS[job_id]:
                    _JOBS[job_id]["stages"] = {}
                _JOBS[job_id]["stages"][sm.group("name")] = {
                    "detail": sm.group("detail").strip(),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                continue
            m = _FILECOUNT_RE.search(last_segment)
            if m:
                _JOBS[job_id]["progress"] = {
                    "label":   "populate-memory",
                    "pct":     0,
                    "current": 0,
                    "total":   int(m.group(1)),
                    "eta":     "",
                }

        await proc.wait()
        _JOBS[job_id]["status"] = "done" if proc.returncode == 0 else "error"
        _JOBS[job_id]["output"] = "".join(chunks)
        # mark progress as complete so pollers see 100% even if the last
        # tqdm frame was buffered out
        if proc.returncode == 0 and _JOBS[job_id].get("progress"):
            _JOBS[job_id]["progress"] = {
                **_JOBS[job_id]["progress"],
                "pct": 100,
                "current": _JOBS[job_id]["progress"].get("total", 0),
                "eta": "done",
            }
        _JOBS[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save_jobs()
    except Exception as exc:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["output"] = str(exc)
        _JOBS[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save_jobs()
