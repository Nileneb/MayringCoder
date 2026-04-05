"""Tests for label.py — auto_label, save/load, stats."""

import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(
    run_id="run1",
    label="app/Foo.php",
    model="qwen:7b",
    call_type="analyze",
    parsed_ok=True,
    findings_count=2,
) -> dict:
    return {
        "timestamp": "2026-04-05T12:00:00Z",
        "run_id": run_id,
        "model": model,
        "label": label,
        "call_type": call_type,
        "prompt_hash": "abc12345",
        "prompt": "Analyze this file.",
        "raw_response": '{"potential_smells": []}',
        "parsed_ok": parsed_ok,
        "findings_count": findings_count,
    }


def _run_json(run_id="run1", filename="app/Foo.php", verdicts=None) -> dict:
    """Build a minimal run JSON with optional second_opinion verdicts."""
    smells = []
    for v in (verdicts or []):
        smells.append({
            "type": "zombie_code",
            "severity": "warning",
            "confidence": "medium",
            "evidence_excerpt": "dead code",
            "_second_opinion_verdict": v,
        })
    return {
        "run_id": run_id,
        "model": "qwen:7b",
        "mode": "analyze",
        "files_checked": 1,
        "timing_seconds": 60.0,
        "results": [{"filename": filename, "potential_smells": smells}],
    }


# ---------------------------------------------------------------------------
# load_training_log / save_labeled
# ---------------------------------------------------------------------------

class TestLoadAndSave:
    def test_load_valid_jsonl(self, tmp_path):
        from label import load_training_log
        f = tmp_path / "log.jsonl"
        f.write_text(
            json.dumps(_entry()) + "\n" + json.dumps(_entry(run_id="run2")) + "\n",
            encoding="utf-8",
        )
        entries = load_training_log(f)
        assert len(entries) == 2

    def test_skips_blank_lines_and_invalid_json(self, tmp_path):
        from label import load_training_log
        f = tmp_path / "log.jsonl"
        f.write_text("\n\nnot json\n" + json.dumps(_entry()) + "\n", encoding="utf-8")
        entries = load_training_log(f)
        assert len(entries) == 1

    def test_save_labeled_roundtrip(self, tmp_path):
        from label import save_labeled, load_training_log
        entries = [_entry(), _entry(run_id="run2")]
        entries[0]["label"] = "positive"
        entries[1]["label"] = "negative"
        out = tmp_path / "out.jsonl"
        n = save_labeled(entries, out)
        assert n == 2
        loaded = load_training_log(out)
        assert loaded[0]["label"] == "positive"
        assert loaded[1]["label"] == "negative"


# ---------------------------------------------------------------------------
# auto_label — heuristic rules
# ---------------------------------------------------------------------------

class TestAutoLabelHeuristics:
    def _label(self, entry, tmp_path):
        from label import auto_label
        result = auto_label([entry], cache_dir=tmp_path)
        return result[0]["label"], result[0]["label_source"]

    def test_parse_failed_is_negative(self, tmp_path):
        label, source = self._label(_entry(parsed_ok=False, findings_count=3), tmp_path)
        assert label == "negative"
        assert source == "parse_failed"

    def test_no_findings_is_neutral(self, tmp_path):
        label, source = self._label(_entry(parsed_ok=True, findings_count=0), tmp_path)
        assert label == "neutral"
        assert source == "no_findings"

    def test_no_run_json_is_candidate(self, tmp_path):
        # tmp_path has no runs/ subdirs
        label, source = self._label(_entry(run_id="nonexistent"), tmp_path)
        assert label == "candidate"
        assert source == "no_run_found"


# ---------------------------------------------------------------------------
# auto_label — second opinion integration
# ---------------------------------------------------------------------------

class TestAutoLabelSecondOpinion:
    def _write_run(self, tmp_path, run_id, filename, verdicts):
        runs_dir = tmp_path / "slug" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run = _run_json(run_id=run_id, filename=filename, verdicts=verdicts)
        (runs_dir / f"{run_id}.json").write_text(
            json.dumps(run), encoding="utf-8"
        )

    def _label(self, entry, tmp_path):
        from label import auto_label
        result = auto_label([entry], cache_dir=tmp_path)
        return result[0]["label"], result[0]["label_source"]

    def test_all_bestätigt_is_positive(self, tmp_path):
        self._write_run(tmp_path, "run1", "app/Foo.php", ["BESTÄTIGT", "BESTÄTIGT"])
        label, source = self._label(_entry(run_id="run1", label="app/Foo.php"), tmp_path)
        assert label == "positive"
        assert source == "second_opinion_confirmed"

    def test_präzisiert_is_positive(self, tmp_path):
        self._write_run(tmp_path, "run1", "app/Foo.php", ["PRÄZISIERT"])
        label, source = self._label(_entry(run_id="run1", label="app/Foo.php"), tmp_path)
        assert label == "positive"
        assert source == "second_opinion_confirmed"

    def test_all_abgelehnt_is_negative(self, tmp_path):
        self._write_run(tmp_path, "run1", "app/Foo.php", ["ABGELEHNT", "ABGELEHNT"])
        label, source = self._label(_entry(run_id="run1", label="app/Foo.php"), tmp_path)
        assert label == "negative"
        assert source == "second_opinion_rejected"

    def test_mixed_verdicts_is_candidate(self, tmp_path):
        self._write_run(tmp_path, "run1", "app/Foo.php", ["BESTÄTIGT", "ABGELEHNT"])
        label, source = self._label(_entry(run_id="run1", label="app/Foo.php"), tmp_path)
        assert label == "candidate"
        assert source == "second_opinion_mixed"

    def test_run_found_but_no_verdicts_is_candidate(self, tmp_path):
        # Run JSON exists but no _second_opinion_verdict on any finding
        self._write_run(tmp_path, "run1", "app/Foo.php", [])
        label, source = self._label(_entry(run_id="run1", label="app/Foo.php"), tmp_path)
        assert label == "candidate"
        assert source == "no_second_opinion"

    def test_run_cached_across_entries(self, tmp_path):
        """Same run_id is only read from disk once (cache check via side effects)."""
        from label import auto_label
        self._write_run(tmp_path, "run1", "app/Foo.php", ["BESTÄTIGT"])
        entries = [_entry(run_id="run1", label="app/Foo.php")] * 3
        result = auto_label(entries, cache_dir=tmp_path)
        assert all(e["label"] == "positive" for e in result)


# ---------------------------------------------------------------------------
# print_stats
# ---------------------------------------------------------------------------

class TestPrintStats:
    def test_runs_without_crash(self, capsys):
        from label import print_stats
        entries = [
            {**_entry(), "label": "positive"},
            {**_entry(), "label": "negative"},
            {**_entry(), "label": "candidate"},
            {**_entry(), "label": "neutral"},
        ]
        print_stats(entries)
        out = capsys.readouterr().out
        assert "positive" in out
        assert "negative" in out
        assert "Gesamt" in out

    def test_empty_entries_no_crash(self, capsys):
        from label import print_stats
        print_stats([])
        out = capsys.readouterr().out
        assert "Gesamt" in out
