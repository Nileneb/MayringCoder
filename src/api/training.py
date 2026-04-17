"""Training pipeline API endpoints.

Provides:
  GET  /api/training/status              — sample counts, last finetune timestamp
  POST /api/training/batches/export      — export current training log as JSONL batch
  POST /api/training/langdock/webhook    — receive annotated batches from Langdock
  POST /api/training/merge               — merge all annotated batches into train.jsonl
  POST /api/training/finetune            — start fine-tuning as background process
  GET  /api/training/finetune/status     — current finetune job status
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError("FastAPI required: pip install fastapi")

_ROOT = Path(__file__).parent.parent.parent
_CACHE_DIR = _ROOT / "cache"
_FINETUNING_DIR = _CACHE_DIR / "finetuning"
_LANGDOCK_BATCHES_DIR = _CACHE_DIR / "langdock_batches"
_TRAIN_JSONL = _FINETUNING_DIR / "train.jsonl"
_VAL_JSONL = _FINETUNING_DIR / "val.jsonl"
_HAIKU_ANNOTATIONS = _CACHE_DIR / "haiku_annotations.jsonl"

_LANGDOCK_WEBHOOK_SECRET = os.getenv("LANGDOCK_WEBHOOK_SECRET", "")
_FINETUNE_OUTPUT_DIR = os.getenv("FINETUNE_OUTPUT_DIR", str(_ROOT / "models" / "finetuned"))

# In-memory finetune job state (single process — no DB needed)
_finetune_job: dict[str, Any] = {
    "job_id": None,
    "status": "idle",          # idle | running | done | error
    "started_at": None,
    "finished_at": None,
    "error": None,
    "pid": None,
}

router = APIRouter(prefix="/api/training", tags=["training"])


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _last_modified_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _prompt_hash(messages: list[dict]) -> str:
    """Stable hash for deduplication: sha256 of system+user messages."""
    key = json.dumps([m for m in messages if m.get("role") in ("system", "user")], sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@router.get("/status")
async def training_status() -> dict:
    """Return training data counts and finetune state."""
    train_count = _count_jsonl_lines(_TRAIN_JSONL)
    val_count = _count_jsonl_lines(_VAL_JSONL)
    haiku_count = _count_jsonl_lines(_HAIKU_ANNOTATIONS)

    batch_files = list(_LANGDOCK_BATCHES_DIR.glob("*.jsonl")) if _LANGDOCK_BATCHES_DIR.exists() else []
    langdock_count = sum(_count_jsonl_lines(f) for f in batch_files)

    return {
        "sample_count": train_count,
        "val_count": val_count,
        "annotated": haiku_count + langdock_count,
        "haiku_annotations": haiku_count,
        "langdock_batches": langdock_count,
        "langdock_batch_files": len(batch_files),
        "last_train_modified": _last_modified_iso(_TRAIN_JSONL),
        "finetune_job": _finetune_job.copy(),
    }


@router.post("/batches/export")
async def export_batch(request: Request) -> dict:
    """Export recent training log entries as a JSONL batch for Langdock annotation.

    Reads all *_training_log.jsonl files in cache/, writes a new batch file to
    cache/langdock_batches/batch_export_{timestamp}.jsonl.
    Returns batch_id and sample_count.
    """
    batch_id = f"export_{int(time.time())}"
    _LANGDOCK_BATCHES_DIR.mkdir(parents=True, exist_ok=True)

    log_files = list(_CACHE_DIR.glob("*_training_log.jsonl"))

    existing_hashes: set[str] = set()
    if _TRAIN_JSONL.exists():
        with _TRAIN_JSONL.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    msgs = entry.get("messages", [])
                    existing_hashes.add(_prompt_hash(msgs))
                except json.JSONDecodeError:
                    pass

    new_samples: list[dict] = []
    for log_file in log_files:
        with log_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    msgs = entry.get("messages", [])
                    ph = _prompt_hash(msgs)
                    if ph not in existing_hashes:
                        new_samples.append(entry)
                        existing_hashes.add(ph)
                except json.JSONDecodeError:
                    pass

    if not new_samples:
        return {"batch_id": batch_id, "sample_count": 0, "message": "Keine neuen Samples gefunden."}

    batch_path = _LANGDOCK_BATCHES_DIR / f"{batch_id}.jsonl"
    with batch_path.open("w", encoding="utf-8") as f:
        for sample in new_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return {
        "batch_id": batch_id,
        "sample_count": len(new_samples),
        "batch_file": str(batch_path.relative_to(_ROOT)),
    }


@router.post("/langdock/webhook")
async def langdock_webhook(request: Request) -> dict:
    """Receive annotated batch from Langdock.

    Expected Authorization: Bearer {LANGDOCK_WEBHOOK_SECRET}
    Body: JSON array of annotated samples:
      [{
        "messages": [...],           # OpenAI messages format
        "label": "good|bad|skip",    # annotation label
        "quality_score": 0.85,       # 0.0–1.0
        "_meta": {...}               # optional metadata
      }]

    Writes to cache/langdock_batches/webhook_{timestamp}.jsonl.
    """
    if _LANGDOCK_WEBHOOK_SECRET:
        auth = request.headers.get("authorization", "")
        token = auth.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token.encode(), _LANGDOCK_WEBHOOK_SECRET.encode()):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook secret")

    body = await request.body()
    try:
        samples = json.loads(body)
        if not isinstance(samples, list):
            raise ValueError("Expected JSON array")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid JSON: {e}")

    if not samples:
        return {"received": 0, "batch_id": None}

    _LANGDOCK_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    batch_id = f"webhook_{int(time.time())}"
    batch_path = _LANGDOCK_BATCHES_DIR / f"{batch_id}.jsonl"

    written = 0
    with batch_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            if isinstance(sample, dict):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1

    return {"received": written, "batch_id": batch_id, "batch_file": str(batch_path.relative_to(_ROOT))}


@router.post("/merge")
async def merge_annotations() -> dict:
    """Merge all annotated batches into cache/finetuning/train.jsonl.

    Sources merged (deduplicated by prompt hash):
    - cache/langdock_batches/*.jsonl
    - cache/haiku_annotations.jsonl

    Only samples with label != "skip" and quality_score >= 0.5 are included.
    Existing train.jsonl entries are preserved.
    """
    _FINETUNING_DIR.mkdir(parents=True, exist_ok=True)

    existing_hashes: dict[str, dict] = {}
    if _TRAIN_JSONL.exists():
        with _TRAIN_JSONL.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ph = _prompt_hash(entry.get("messages", []))
                    existing_hashes[ph] = entry
                except json.JSONDecodeError:
                    pass

    initial_count = len(existing_hashes)
    added = 0
    skipped = 0

    sources: list[Path] = []
    if _HAIKU_ANNOTATIONS.exists():
        sources.append(_HAIKU_ANNOTATIONS)
    if _LANGDOCK_BATCHES_DIR.exists():
        sources.extend(sorted(_LANGDOCK_BATCHES_DIR.glob("*.jsonl")))

    for source_file in sources:
        with source_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                label = entry.get("label", "good")
                quality = float(entry.get("quality_score", 1.0))
                if label == "skip" or quality < 0.5:
                    skipped += 1
                    continue

                msgs = entry.get("messages", [])
                if not msgs:
                    skipped += 1
                    continue

                ph = _prompt_hash(msgs)
                if ph not in existing_hashes:
                    existing_hashes[ph] = entry
                    added += 1

    with _TRAIN_JSONL.open("w", encoding="utf-8") as f:
        for entry in existing_hashes.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {
        "total": len(existing_hashes),
        "existing": initial_count,
        "added": added,
        "skipped": skipped,
    }


@router.post("/finetune")
async def trigger_finetune(background_tasks: BackgroundTasks) -> dict:
    """Start fine-tuning as a background subprocess.

    Runs tools/finetune_qwen.py with current train.jsonl.
    Returns job_id immediately. Poll /finetune/status for updates.
    """
    global _finetune_job

    if _finetune_job["status"] == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Fine-tuning already running (job: {_finetune_job['job_id']})"
        )

    train_count = _count_jsonl_lines(_TRAIN_JSONL)
    if train_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Keine Trainingsdaten vorhanden. Zuerst Annotationen mergen."
        )

    job_id = str(uuid.uuid4())[:8]
    _finetune_job.update({
        "job_id": job_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "error": None,
        "pid": None,
    })

    background_tasks.add_task(_run_finetune, job_id)
    return {"job_id": job_id, "status": "started", "train_samples": train_count}


async def _run_finetune(job_id: str) -> None:
    """Background task: run tools/finetune_qwen.py."""
    global _finetune_job

    finetune_script = _ROOT / "tools" / "finetune_qwen.py"
    if not finetune_script.exists():
        _finetune_job.update({
            "status": "error",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": f"finetune_qwen.py nicht gefunden: {finetune_script}",
        })
        return

    cmd = [
        sys.executable, str(finetune_script),
        "--train", str(_TRAIN_JSONL),
        "--val", str(_VAL_JSONL),
        "--output-dir", _FINETUNE_OUTPUT_DIR,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _finetune_job["pid"] = proc.pid

        proc.wait()

        if proc.returncode == 0:
            _finetune_job.update({
                "status": "done",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
            })
        else:
            _finetune_job.update({
                "status": "error",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": f"Prozess beendet mit Code {proc.returncode}",
            })
    except Exception as exc:
        _finetune_job.update({
            "status": "error",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        })


@router.get("/finetune/status")
async def finetune_status() -> dict:
    """Return current finetune job status."""
    return _finetune_job.copy()
