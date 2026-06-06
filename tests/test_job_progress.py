"""Live progress for long-running pipeline jobs.

Background: run_checker_job used proc.communicate() — the client polled
/jobs/{id} for 45 minutes and only saw status='started' because stdout
was buffered until process exit. The new implementation reads stdout
line-by-line, parses tqdm progress lines and fills _JOBS[id]['progress'].
"""
from __future__ import annotations

import pytest

from src.api.job_queue import (
    _JOBS,
    _parse_populate_progress,
    _parse_progress_line,
    make_job,
)


class TestProgressLineParser:
    def test_chunk_progress(self):
        line = "Chunks embedden:  45%|████▌     | 9/20 [00:05<00:06,  1.74chunk/s]"
        p = _parse_progress_line(line)
        assert p["label"] == "Chunks embedden"
        assert p["pct"] == 45
        assert p["current"] == 9
        assert p["total"] == 20

    def test_file_progress(self):
        line = "populate-memory: 100%|##########| 228/228 [12:34<00:00,  3.3s/file]"
        p = _parse_progress_line(line)
        assert p["pct"] == 100
        assert p["current"] == p["total"] == 228
        assert p["label"] == "populate-memory"

    def test_zero_percent_line(self):
        line = "populate-memory:   0%|          | 0/228 [00:00<?, ?file/s]"
        p = _parse_progress_line(line)
        assert p["pct"] == 0
        assert p["current"] == 0
        assert p["total"] == 228

    def test_non_progress_line(self):
        assert _parse_progress_line("2026-04-22 INFO ingest_start") is None
        assert _parse_progress_line("just text") is None
        assert _parse_progress_line("") is None

    def test_carriage_return_split_picks_latest_frame(self):
        # Real tqdm writes \r-overwrites; run_checker_job splits by \r and
        # takes the last segment — the parser sees only that piece.
        compound = (
            "Chunks embedden:  10%|#         | 2/20 [00:01<00:09,  2chunk/s]"
            "\r"
            "Chunks embedden:  50%|#####     | 10/20 [00:05<00:05,  2chunk/s]"
        )
        last = compound.split("\r")[-1]
        p = _parse_progress_line(last)
        assert p["pct"] == 50


class TestPopulateProgressParser:
    """#275: the populate loop's tqdm uses \\r-only frames the newline-based
    line reader never sees → progress.current stuck at 0/N. The loop now emits
    an explicit ``[POPULATE-PROGRESS] cur/total`` marker; this parser fills it."""

    def test_mid_run_progress(self):
        p = _parse_populate_progress("[POPULATE-PROGRESS] 42/207")
        assert p is not None
        assert p["label"] == "populate-memory"
        assert p["current"] == 42
        assert p["total"] == 207
        assert p["pct"] == 20  # int(42*100/207)

    def test_final_tick_is_100(self):
        p = _parse_populate_progress("[POPULATE-PROGRESS] 207/207")
        assert p["current"] == p["total"] == 207
        assert p["pct"] == 100

    def test_zero_total_no_zero_division(self):
        p = _parse_populate_progress("[POPULATE-PROGRESS] 0/0")
        assert p["pct"] == 0
        assert p["current"] == 0

    def test_embedded_in_log_line(self):
        # readline() may hand us the marker with surrounding text; .search finds it.
        p = _parse_populate_progress("2026-06-06 INFO [POPULATE-PROGRESS] 5/10 done")
        assert p["current"] == 5
        assert p["pct"] == 50

    def test_non_marker_line(self):
        assert _parse_populate_progress("[STAGE] ingest_loop start total=10") is None
        assert _parse_populate_progress("populate-memory: 50%|#| 5/10") is None
        assert _parse_populate_progress("") is None


class TestMakeJobInitialsProgressField:
    def test_fresh_job_has_progress_key_as_none(self):
        job_id = make_job("ws-x")
        assert "progress" in _JOBS[job_id]
        assert _JOBS[job_id]["progress"] is None
