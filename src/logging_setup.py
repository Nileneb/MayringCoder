"""Strukturiertes JSON-Logging für MayringCoder.

Wird beim FastAPI-/Pi-Server-Start aufgerufen. Konfiguriert root-Logger
mit JSON-Formatter und schreibt nach
``$MAYRING_LOG_FILE`` (default ``cache/logs/mayring.json``).

Das ist die EINE Logfile, die der log-ingest-Cron in den
``bene:logs``-Workspace einspielt. Vorteile gegenüber `docker logs`:
  - strukturiert (level/timestamp/message vorgekapselt → kein Regex-
    Block-Parser nötig)
  - kein PII-Leak: was nicht ins logger-Statement geht, kann auch
    nicht ingested werden
  - line-orientiert: Tail-via-offset statt --since-Polling
"""
from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        # Optional extra-fields die Caller mitschicken:
        # logger.info("...", extra={"workspace_id": "bene", "job_id": "x"})
        for key, val in record.__dict__.items():
            if key in ("args", "asctime", "created", "exc_info", "exc_text",
                      "filename", "funcName", "levelname", "levelno",
                      "lineno", "module", "msecs", "message", "msg", "name",
                      "pathname", "process", "processName", "relativeCreated",
                      "stack_info", "thread", "threadName", "taskName"):
                continue
            try:
                json.dumps(val)
                payload[key] = val
            except (TypeError, ValueError):
                payload[key] = str(val)
        return json.dumps(payload, ensure_ascii=False)


def configure_json_logging() -> None:
    """Idempotent — wird beim Server-Start aufgerufen."""
    if os.environ.get("MAYRING_LOG_FILE_CONFIGURED") == "1":
        return
    log_file = os.environ.get("MAYRING_LOG_FILE", "cache/logs/mayring.json")
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        str(path),
        maxBytes=int(os.environ.get("MAYRING_LOG_MAX_BYTES", str(50 * 1024 * 1024))),
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # NICHT die existierende stdout-handler entfernen — die landet im
    # docker logs für ad-hoc kubectl/docker-Inspektion.
    root.addHandler(handler)

    os.environ["MAYRING_LOG_FILE_CONFIGURED"] = "1"
