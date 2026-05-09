"""Tests für tools/log_ingest_cron.py — JSON-Tail + Filter + Scrub."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "log_ingest_cron", ROOT / "tools" / "log_ingest_cron.py"
)
log_ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log_ingest)


# ─── parse_json_line ───────────────────────────────────────────────


def test_parse_valid_json():
    r = log_ingest.parse_json_line('{"level":"ERROR","msg":"x"}')
    assert r == {"level": "ERROR", "msg": "x"}


def test_parse_invalid_json_returns_none():
    assert log_ingest.parse_json_line("plain text") is None


def test_parse_empty_returns_none():
    assert log_ingest.parse_json_line("") is None


# ─── line_to_chunk ─────────────────────────────────────────────────


def test_line_to_chunk_basic():
    record = {"ts": "2026-05-09T08:00:00Z", "level": "ERROR",
              "logger": "src.api.foo", "msg": "boom"}
    out = log_ingest.line_to_chunk(record)
    assert "2026-05-09" in out
    assert "ERROR" in out
    assert "[src.api.foo]" in out
    assert "boom" in out


def test_line_to_chunk_includes_traceback():
    record = {"level": "ERROR", "msg": "x",
              "exc": "Traceback (most recent call last):\n  …"}
    out = log_ingest.line_to_chunk(record)
    assert "Traceback" in out


def test_line_to_chunk_scrubs_token_in_msg():
    record = {"level": "ERROR",
              "msg": "got Bearer abc.def-123_xyz= for caller"}
    out = log_ingest.line_to_chunk(record)
    assert "Bearer <REDACTED>" in out
    assert "abc.def-123_xyz" not in out


# ─── tail_new_lines ────────────────────────────────────────────────


def test_tail_new_lines_reads_only_new(tmp_path):
    p = tmp_path / "log.json"
    p.write_text('{"level":"INFO","msg":"a"}\n{"level":"WARNING","msg":"b"}\n')
    # initial: offset=0 → liest beide
    lines, new_offset = log_ingest.tail_new_lines(p, 0)
    assert len(lines) == 2
    assert new_offset == p.stat().st_size

    # weiter schreiben, nur neues lesen
    with p.open("a") as f:
        f.write('{"level":"ERROR","msg":"c"}\n')
    lines2, _ = log_ingest.tail_new_lines(p, new_offset)
    assert len(lines2) == 1
    assert "c" in lines2[0]


def test_tail_new_lines_handles_rotation(tmp_path):
    p = tmp_path / "log.json"
    p.write_text("a-very-long-original-line\n" * 100)
    initial_offset = p.stat().st_size

    # Rotation: Datei wird kleiner
    p.write_text('{"level":"WARNING","msg":"after-rotate"}\n')
    lines, new_offset = log_ingest.tail_new_lines(p, initial_offset)
    assert len(lines) == 1
    assert "after-rotate" in lines[0]
    assert new_offset == p.stat().st_size


def test_tail_new_lines_keeps_partial_for_next_run(tmp_path):
    """Wenn die letzte Zeile NICHT mit \\n endet (Logger schreibt
    grad), behalten wir sie für den nächsten Run im offset."""
    p = tmp_path / "log.json"
    p.write_text('{"level":"INFO","msg":"complete"}\n{"level":"ERR')
    lines, new_offset = log_ingest.tail_new_lines(p, 0)
    assert len(lines) == 1
    assert "complete" in lines[0]
    # offset zeigt NICHT auf das Datei-Ende, sondern auf den Anfang
    # der unvollständigen Zeile → nächster Run fängt dort an.
    assert new_offset < p.stat().st_size


# ─── PII-Scrubber (Defensive) ──────────────────────────────────────


def test_scrub_bearer():
    out = log_ingest.scrub_pii("auth: Bearer abc.def_xyz")
    assert "Bearer <REDACTED>" in out


def test_scrub_jwt():
    jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ4In0.signaturepart_xyz"
    out = log_ingest.scrub_pii(f"tok={jwt}")
    assert "<JWT_REDACTED>" in out


def test_scrub_long_hex_standalone():
    out = log_ingest.scrub_pii(
        "service hash 8aecd5782447a8276f0989152f91267be27fcfe4028459a016126f5c6efe7537 ok"
    )
    assert "<HEX_REDACTED>" in out


# ─── Level-Filter via main() integration ───────────────────────────


def test_main_skips_info_level(tmp_path, monkeypatch, capsys):
    log_file = tmp_path / "log.json"
    log_file.write_text(
        '{"ts":"2026-01-01","level":"INFO","msg":"heartbeat"}\n'
        '{"ts":"2026-01-01","level":"ERROR","msg":"crashed","exc":"Trace"}\n'
        '{"ts":"2026-01-01","level":"DEBUG","msg":"x"}\n'
    )
    state_file = tmp_path / "state.json"
    monkeypatch.setenv("LOG_INGEST_STATE_FILE", str(state_file))
    monkeypatch.setenv("LOG_INGEST_FILES", str(log_file))
    monkeypatch.setenv("MAYRING_API_URL", "http://x")
    monkeypatch.setenv("MCP_SERVICE_TOKEN", "tok")

    monkeypatch.setattr("sys.argv", ["log_ingest_cron.py", "--dry-run"])
    log_ingest.main()

    err = capsys.readouterr().err
    assert "3 lines total" in err
    assert "1 accepted" in err
