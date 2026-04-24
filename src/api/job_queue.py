"""In-memory job queue for background pipeline jobs.

Live-Progress: ``run_checker_job`` liest den Subprocess-Output Zeile für
Zeile (statt erst am Ende via ``proc.communicate()``). tqdm-Progress-
Zeilen werden geparst und unter ``_JOBS[id]["progress"]`` abgelegt — der
Client pollt via GET /jobs/{id} und sieht Fortschritt statt stoischer
"started"-Schweigminuten.
"""
from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent.parent
_JOBS: dict[str, dict] = {}


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

# "[STAGE] fetch_repo done files=274" → stages dict
_STAGE_RE = re.compile(r"\[STAGE\]\s+(?P<name>\S+)\s+(?P<detail>.*)")


def make_job(workspace_id: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "started",
        "output": "",
        "progress": None,
        "workspace_id": workspace_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    return _JOBS.get(job_id)


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


async def run_checker_job(job_id: str, checker_args: list[str], workspace_id: str) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            python_exe(), "-m", "src.pipeline", *checker_args,
            "--workspace-id", workspace_id,
            cwd=str(_ROOT),
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
    except Exception as exc:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["output"] = str(exc)
