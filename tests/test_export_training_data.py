"""Tests for export_training_data.py — filter, format, export."""

import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(
    label="positive",
    call_type="analyze",
    prompt="Analyze this file.",
    raw_response='{"potential_smells": []}',
    model="qwen:7b",
) -> dict:
    return {
        "timestamp": "2026-04-05T12:00:00Z",
        "run_id": "run1",
        "model": model,
        "label": label,
        "call_type": call_type,
        "prompt_hash": "abc12345",
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_ok": True,
        "findings_count": 2,
    }


def _write_labeled(tmp_path, entries) -> Path:
    f = tmp_path / "labeled.jsonl"
    f.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )
    return f


# ---------------------------------------------------------------------------
# load_labeled
# ---------------------------------------------------------------------------

class TestLoadLabeled:
    def test_loads_all_entries(self, tmp_path):
        from export_training_data import load_labeled
        f = _write_labeled(tmp_path, [_entry(), _entry(label="negative")])
        entries = load_labeled(f)
        assert len(entries) == 2

    def test_skips_blank_and_invalid(self, tmp_path):
        from export_training_data import load_labeled
        f = tmp_path / "log.jsonl"
        f.write_text("\n\nnot json\n" + json.dumps(_entry()) + "\n", encoding="utf-8")
        entries = load_labeled(f)
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# filter_entries
# ---------------------------------------------------------------------------

class TestFilterEntries:
    def _entries(self):
        return [
            _entry(label="positive", call_type="analyze"),
            _entry(label="negative", call_type="analyze"),
            _entry(label="positive", call_type="overview"),
            _entry(label="candidate", call_type="analyze"),
        ]

    def test_filter_by_label_positive(self):
        from export_training_data import filter_entries
        result = filter_entries(self._entries(), label="positive", call_type="all")
        assert len(result) == 2
        assert all(e["label"] == "positive" for e in result)

    def test_filter_by_label_negative(self):
        from export_training_data import filter_entries
        result = filter_entries(self._entries(), label="negative")
        assert len(result) == 1

    def test_filter_label_all(self):
        from export_training_data import filter_entries
        result = filter_entries(self._entries(), label="all", call_type="all")
        assert len(result) == 4

    def test_filter_by_call_type_analyze(self):
        from export_training_data import filter_entries
        result = filter_entries(self._entries(), label="all", call_type="analyze")
        assert len(result) == 3
        assert all(e["call_type"] == "analyze" for e in result)

    def test_filter_by_call_type_overview(self):
        from export_training_data import filter_entries
        result = filter_entries(self._entries(), label="all", call_type="overview")
        assert len(result) == 1

    def test_combined_filter(self):
        from export_training_data import filter_entries
        result = filter_entries(self._entries(), label="positive", call_type="analyze")
        assert len(result) == 1

    def test_export_only_positive_by_default(self):
        from export_training_data import filter_entries
        result = filter_entries(self._entries())  # defaults: label="positive", call_type="analyze"
        assert len(result) == 1
        assert result[0]["label"] == "positive"
        assert result[0]["call_type"] == "analyze"


# ---------------------------------------------------------------------------
# format_entry
# ---------------------------------------------------------------------------

class TestFormatEntry:
    def test_raw_format(self):
        from export_training_data import format_entry
        e = _entry(prompt="my prompt", raw_response="my response")
        out = format_entry(e, "raw")
        assert out == {"prompt": "my prompt", "response": "my response"}

    def test_alpaca_format_has_required_keys(self):
        from export_training_data import format_entry
        e = _entry(prompt="Do something.\nDatei: app/Foo.php\ncontent here", raw_response="ok")
        out = format_entry(e, "alpaca")
        assert "instruction" in out
        assert "input" in out
        assert "output" in out
        assert out["output"] == "ok"

    def test_alpaca_splits_on_datei(self):
        from export_training_data import format_entry
        e = _entry(prompt="Instruction here.\nDatei: app/Foo.php", raw_response="ok")
        out = format_entry(e, "alpaca")
        assert out["instruction"] == "Instruction here."
        assert "Datei:" in out["input"]

    def test_alpaca_no_datei_line(self):
        from export_training_data import format_entry
        e = _entry(prompt="Just an instruction.", raw_response="ok")
        out = format_entry(e, "alpaca")
        assert out["instruction"] == "Just an instruction."
        assert out["input"] == ""

    def test_sharegpt_format(self):
        from export_training_data import format_entry
        e = _entry(prompt="human prompt", raw_response="gpt response")
        out = format_entry(e, "sharegpt")
        assert "conversations" in out
        convs = out["conversations"]
        assert len(convs) == 2
        assert convs[0] == {"from": "human", "value": "human prompt"}
        assert convs[1] == {"from": "gpt", "value": "gpt response"}

    def test_unknown_format_raises(self):
        from export_training_data import format_entry
        with pytest.raises(ValueError, match="Unbekanntes Format"):
            format_entry(_entry(), "xml")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_raw_creates_valid_jsonl(self, tmp_path):
        from export_training_data import export
        entries = [_entry(label="positive"), _entry(label="positive")]
        out = tmp_path / "out.jsonl"
        n = export(entries, out, fmt="raw")
        assert n == 2
        lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2
        assert all("prompt" in l and "response" in l for l in lines)

    def test_export_alpaca_format(self, tmp_path):
        from export_training_data import export
        entries = [_entry()]
        out = tmp_path / "out.jsonl"
        n = export(entries, out, fmt="alpaca")
        assert n == 1
        line = json.loads(out.read_text(encoding="utf-8").strip())
        assert "instruction" in line
        assert "output" in line

    def test_export_sharegpt_format(self, tmp_path):
        from export_training_data import export
        entries = [_entry()]
        out = tmp_path / "out.jsonl"
        n = export(entries, out, fmt="sharegpt")
        assert n == 1
        line = json.loads(out.read_text(encoding="utf-8").strip())
        assert "conversations" in line

    def test_output_is_valid_jsonl(self, tmp_path):
        from export_training_data import export
        entries = [_entry(prompt=f"prompt {i}", raw_response=f"resp {i}") for i in range(5)]
        out = tmp_path / "out.jsonl"
        n = export(entries, out, fmt="raw")
        assert n == 5
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5
        for line in lines:
            obj = json.loads(line)
            assert isinstance(obj, dict)

    def test_creates_parent_dirs(self, tmp_path):
        from export_training_data import export
        out = tmp_path / "deep" / "subdir" / "out.jsonl"
        export([_entry()], out, fmt="raw")
        assert out.exists()

    def test_returns_count(self, tmp_path):
        from export_training_data import export
        out = tmp_path / "out.jsonl"
        n = export([_entry(), _entry(), _entry()], out, fmt="raw")
        assert n == 3
