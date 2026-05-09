"""Tests für tools/log_ingest_cron.py — PII-Scrubbing + Block-Extraction."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Tool wird via spec geladen, damit es auch ohne package install läuft.
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "log_ingest_cron", ROOT / "tools" / "log_ingest_cron.py"
)
log_ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log_ingest)


# ─── PII-Scrubber ──────────────────────────────────────────────────


def test_scrub_bearer_token():
    out = log_ingest.scrub_pii("Authorization: Bearer abc.def-123_xyz=")
    assert "Bearer <REDACTED>" in out
    assert "abc.def-123_xyz" not in out


def test_scrub_jwt():
    jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjMifQ.signaturepart_xyz"
    out = log_ingest.scrub_pii(f"token: {jwt}")
    assert "<JWT_REDACTED>" in out
    assert "eyJ" not in out


def test_scrub_email():
    out = log_ingest.scrub_pii("user bene@linn.games said")
    assert "<EMAIL>" in out
    assert "bene@linn.games" not in out


def test_scrub_ipv4():
    out = log_ingest.scrub_pii("connection from 192.168.178.11 to ...")
    assert "<IP>" in out
    assert "192.168.178.11" not in out


def test_scrub_keeps_unrelated_numbers():
    """timestamps wie 2026-05-09 sollen NICHT als IP getroffen werden."""
    out = log_ingest.scrub_pii("[2026-05-09 06:42:00] startup")
    assert "<IP>" not in out
    assert "2026-05-09" in out


def test_scrub_password_query():
    out = log_ingest.scrub_pii("?password=hunter2&user=foo")
    assert "password=<REDACTED>" in out
    assert "hunter2" not in out


def test_scrub_long_hex_key():
    """Standalone-hex (z.B. service-token-leak im body) wird gescrubbt."""
    raw = "service hash 8aecd5782447a8276f0989152f91267be27fcfe4028459a016126f5c6efe7537 ok"
    out = log_ingest.scrub_pii(raw)
    assert "<HEX_REDACTED>" in out
    assert "8aecd57" not in out


# ─── Block-Extraction ──────────────────────────────────────────────


def test_extract_keeps_only_severity_blocks():
    raw = """\
2026-05-09 06:00:00 INFO: server started
2026-05-09 06:01:00 DEBUG: request id=42
2026-05-09 06:02:00 ERROR: ValueError: bad input
  File "x.py", line 42, in foo
    raise ValueError("bad input")
2026-05-09 06:03:00 INFO: heartbeat ok
"""
    blocks = log_ingest.extract_severity_blocks(raw)
    assert len(blocks) == 1
    assert "ValueError: bad input" in blocks[0]
    assert "heartbeat ok" not in blocks[0]


def test_extract_traceback_block_intact():
    raw = """\
2026-05-09 06:00:00 INFO: ok
Traceback (most recent call last):
  File "x.py", line 1, in <module>
    raise RuntimeError('boom')
RuntimeError: boom
2026-05-09 06:00:01 INFO: next
"""
    blocks = log_ingest.extract_severity_blocks(raw)
    assert len(blocks) == 1
    block = blocks[0]
    assert "Traceback" in block
    assert 'File "x.py"' in block
    assert "RuntimeError: boom" in block


def test_extract_returns_empty_when_only_info():
    raw = """\
2026-05-09 06:00:00 INFO: started
2026-05-09 06:00:01 INFO: heartbeat
"""
    assert log_ingest.extract_severity_blocks(raw) == []


def test_extract_warning_kept():
    raw = "2026-05-09 06:00:00 WARNING: cache cold\n"
    blocks = log_ingest.extract_severity_blocks(raw)
    assert len(blocks) == 1


# ─── Source-ID-Stabilität (idempotent dedup) ───────────────────────


def test_post_chunk_dry_run_logs_stable_source_id(capsys):
    """In dry-run: identischer block → identische source_id → dedup
    durch /memory/put-content_hash auf Server-Seite."""
    block = "2026-05-09 ERROR: same\n"
    log_ingest.post_chunk(
        "http://x", "tok", "bene:logs", "myc", block,
        "2026-05-09T00:00:00+00:00", dry_run=True,
    )
    out = capsys.readouterr().out
    log_ingest.post_chunk(
        "http://x", "tok", "bene:logs", "myc", block,
        "2026-05-09T00:00:00+00:00", dry_run=True,
    )
    out2 = capsys.readouterr().out
    # gleiche source_id in beiden runs. Output-Format:
    # `# DRY-RUN POST <source_id>: <block-prefix>...`
    import re as _re
    m1 = _re.search(r"POST (log:[^\s:]+:[a-f0-9]+):", out)
    m2 = _re.search(r"POST (log:[^\s:]+:[a-f0-9]+):", out2)
    assert m1 and m2
    assert m1.group(1) == m2.group(1)
    assert m1.group(1).startswith("log:myc:")
