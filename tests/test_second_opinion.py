"""Unit tests for second_opinion_validate() in src.extractor (Issue #15).

All tests are fully offline — _ollama_generate is monkeypatched so no
real Ollama instance is needed.
"""

import json
import pytest
from unittest.mock import patch

from src.extractor import second_opinion_validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finding(fn: str = "a.py", **overrides) -> dict:
    base = {
        "_filename": fn,
        "type": "zombie_code",
        "severity": "warning",
        "confidence": "medium",
        "line_hint": "~10",
        "evidence_excerpt": "dead function here",
        "fix_suggestion": "Remove it.",
    }
    base.update(overrides)
    return base


def _file(fn: str = "a.py", content: str = "def foo(): pass") -> dict:
    return {"filename": fn, "content": content}


def _raw_json(verdict: str, reasoning: str = "ok", adjusted_severity=None, additional_note=None) -> str:
    return json.dumps({
        "verdict": verdict,
        "reasoning": reasoning,
        "adjusted_severity": adjusted_severity,
        "additional_note": additional_note,
    })


# ---------------------------------------------------------------------------
# Basic verdict routing
# ---------------------------------------------------------------------------

class TestSecondOpinionValidate:

    def test_bestätigt_keeps_finding(self):
        raw = _raw_json("BESTÄTIGT")
        with patch("src.analyzer._ollama_generate", return_value=raw):
            validated, stats = second_opinion_validate(
                [_finding()], [_file()], "http://localhost:11434", "deepseek-coder:6.7b"
            )
        assert len(validated) == 1
        assert stats["confirmed"] == 1
        assert stats["rejected"] == 0
        assert stats["refined"] == 0
        assert validated[0]["_second_opinion_verdict"] == "BESTÄTIGT"

    def test_abgelehnt_drops_finding(self):
        raw = _raw_json("ABGELEHNT", reasoning="This is a framework convention.")
        with patch("src.analyzer._ollama_generate", return_value=raw):
            validated, stats = second_opinion_validate(
                [_finding()], [_file()], "http://localhost:11434", "deepseek-coder:6.7b"
            )
        assert len(validated) == 0
        assert stats["rejected"] == 1
        assert stats["confirmed"] == 0

    def test_präzisiert_keeps_finding_and_adjusts_severity(self):
        raw = _raw_json("PRÄZISIERT", reasoning="True but only info.", adjusted_severity="info")
        with patch("src.analyzer._ollama_generate", return_value=raw):
            validated, stats = second_opinion_validate(
                [_finding(severity="warning")], [_file()],
                "http://localhost:11434", "deepseek-coder:6.7b"
            )
        assert len(validated) == 1
        assert stats["refined"] == 1
        assert validated[0]["severity"] == "info"
        assert validated[0]["_second_opinion_verdict"] == "PRÄZISIERT"

    def test_präzisiert_without_adjusted_severity_leaves_severity_unchanged(self):
        raw = _raw_json("PRÄZISIERT", reasoning="Close enough.", adjusted_severity=None)
        with patch("src.analyzer._ollama_generate", return_value=raw):
            validated, stats = second_opinion_validate(
                [_finding(severity="warning")], [_file()],
                "http://localhost:11434", "model"
            )
        assert validated[0]["severity"] == "warning"
        assert stats["refined"] == 1

    def test_invalid_adjusted_severity_leaves_severity_unchanged(self):
        raw = _raw_json("PRÄZISIERT", adjusted_severity="extreme")
        with patch("src.analyzer._ollama_generate", return_value=raw):
            validated, stats = second_opinion_validate(
                [_finding(severity="warning")], [_file()],
                "http://localhost:11434", "model"
            )
        assert validated[0]["severity"] == "warning"

    def test_additional_note_stored_on_finding(self):
        raw = _raw_json("BESTÄTIGT", additional_note="Check also bar.py")
        with patch("src.analyzer._ollama_generate", return_value=raw):
            validated, _ = second_opinion_validate(
                [_finding()], [_file()], "http://localhost:11434", "model"
            )
        assert validated[0].get("_second_opinion_note") == "Check also bar.py"

    def test_llm_error_keeps_finding_and_increments_errors(self):
        with patch("src.analyzer._ollama_generate", side_effect=Exception("timeout")):
            validated, stats = second_opinion_validate(
                [_finding()], [_file()], "http://localhost:11434", "model"
            )
        assert len(validated) == 1
        assert stats["errors"] == 1
        assert validated[0]["_second_opinion_verdict"] == "ERROR"

    def test_malformed_json_defaults_to_bestätigt(self):
        with patch("src.analyzer._ollama_generate", return_value="not json at all"):
            validated, stats = second_opinion_validate(
                [_finding()], [_file()], "http://localhost:11434", "model"
            )
        assert len(validated) == 1
        assert stats["confirmed"] == 1

    def test_multiple_findings_mixed_verdicts(self):
        responses = [
            _raw_json("BESTÄTIGT"),
            _raw_json("ABGELEHNT"),
            _raw_json("PRÄZISIERT", adjusted_severity="critical"),
        ]
        findings = [_finding("a.py"), _finding("b.py"), _finding("c.py")]
        files = [_file("a.py"), _file("b.py"), _file("c.py")]
        with patch("src.analyzer._ollama_generate", side_effect=responses):
            validated, stats = second_opinion_validate(
                findings, files, "http://localhost:11434", "model"
            )
        assert len(validated) == 2
        assert stats == {"confirmed": 1, "rejected": 1, "refined": 1, "errors": 0}

    def test_empty_findings_returns_empty(self):
        with patch("src.analyzer._ollama_generate") as mock:
            validated, stats = second_opinion_validate(
                [], [], "http://localhost:11434", "model"
            )
        mock.assert_not_called()
        assert validated == []
        assert stats == {"confirmed": 0, "rejected": 0, "refined": 0, "errors": 0}

    def test_json_in_fenced_code_block_is_parsed(self):
        raw = f"```json\n{_raw_json('ABGELEHNT')}\n```"
        with patch("src.analyzer._ollama_generate", return_value=raw):
            validated, stats = second_opinion_validate(
                [_finding()], [_file()], "http://localhost:11434", "model"
            )
        assert stats["rejected"] == 1

    def test_file_not_in_map_still_processes_finding(self):
        """Finding references a file not in the files list → code_snippet is empty."""
        raw = _raw_json("BESTÄTIGT")
        with patch("src.analyzer._ollama_generate", return_value=raw):
            validated, stats = second_opinion_validate(
                [_finding("missing.py")], [], "http://localhost:11434", "model"
            )
        assert len(validated) == 1


# ---------------------------------------------------------------------------
# aggregator integration
# ---------------------------------------------------------------------------

class TestAggregatorSecondOpinionStats:
    def test_second_opinion_stats_in_aggregation(self):
        from src.aggregator import aggregate_findings
        results = [{"filename": "a.py", "potential_smells": [], "_parse_error": False}]
        so_stats = {"confirmed": 3, "rejected": 1, "refined": 1, "errors": 0}
        agg = aggregate_findings(results, second_opinion_stats=so_stats)
        assert agg["second_opinion_stats"] == so_stats

    def test_second_opinion_stats_defaults_to_empty_dict(self):
        from src.aggregator import aggregate_findings
        agg = aggregate_findings([])
        assert agg["second_opinion_stats"] == {}


# ---------------------------------------------------------------------------
# report integration
# ---------------------------------------------------------------------------

class TestReportSecondOpinionStats:
    def test_second_opinion_appears_in_report(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        from src.report import generate_report

        so_stats = {"confirmed": 2, "rejected": 1, "refined": 1, "errors": 0}
        agg = {
            "total_findings": 0,
            "by_severity": {"critical": 0, "warning": 0, "info": 0},
            "top_findings": [],
            "needs_explikation": [],
            "next_steps": [],
            "parse_errors": [],
            "adversarial_stats": {},
            "second_opinion_stats": so_stats,
        }
        diff = {
            "changed": [], "added": [], "removed": [], "unchanged": [],
            "unanalyzed": [], "selected": [], "skipped": [], "snapshot_id": 1,
        }
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="qwen2.5-coder",
            results=[],
            aggregation=agg,
            diff=diff,
            timing=1.0,
        )
        content = (tmp_path / path.split("/")[-1]).read_text(encoding="utf-8")
        assert "Second Opinion" in content
        assert "2 BESTÄTIGT" in content
        assert "1 ABGELEHNT" in content
        assert "1 PRÄZISIERT" in content

    def test_second_opinion_absent_when_stats_empty(self, tmp_path):
        from src import report as mod
        mod.REPORTS_DIR = tmp_path
        from src.report import generate_report

        agg = {
            "total_findings": 0,
            "by_severity": {"critical": 0, "warning": 0, "info": 0},
            "top_findings": [],
            "needs_explikation": [],
            "next_steps": [],
            "parse_errors": [],
            "adversarial_stats": {},
            "second_opinion_stats": {},
        }
        diff = {
            "changed": [], "added": [], "removed": [], "unchanged": [],
            "unanalyzed": [], "selected": [], "skipped": [], "snapshot_id": 1,
        }
        path = generate_report(
            repo_url="https://github.com/test/repo",
            model="m",
            results=[],
            aggregation=agg,
            diff=diff,
            timing=1.0,
        )
        content = (tmp_path / path.split("/")[-1]).read_text(encoding="utf-8")
        assert "Second Opinion" not in content


# ---------------------------------------------------------------------------
# Targeted question generation (Issue #15)
# ---------------------------------------------------------------------------

class TestBuildSecondOpinionQuestion:

    def test_redundanz_returns_targeted_question(self):
        from src.extractor import _build_second_opinion_question
        q = _build_second_opinion_question({"type": "redundanz", "evidence_excerpt": "getUserData()"})
        assert "getUserData()" in q
        assert "aehnlicher" in q.lower() or "gleicher" in q.lower()

    def test_sicherheit_mentions_sanitization(self):
        from src.extractor import _build_second_opinion_question
        q = _build_second_opinion_question({"type": "sicherheit", "evidence_excerpt": "$_POST['name']"})
        assert "unsanitisiert" in q.lower() or "validiert" in q.lower()

    def test_zombie_code_mentions_reference(self):
        from src.extractor import _build_second_opinion_question
        q = _build_second_opinion_question({"type": "zombie_code", "evidence_excerpt": "oldMethod()"})
        assert "aufgerufen" in q.lower() or "referenziert" in q.lower()

    def test_unknown_type_uses_fallback(self):
        from src.extractor import _build_second_opinion_question
        q = _build_second_opinion_question({"type": "neuartig", "evidence_excerpt": "some code"})
        assert "some code" in q
        assert len(q) > 20

    def test_empty_evidence_uses_fix_suggestion(self):
        from src.extractor import _build_second_opinion_question
        q = _build_second_opinion_question({
            "type": "redundanz",
            "evidence_excerpt": "",
            "fix_suggestion": "Zusammenfuehren"
        })
        assert "Zusammenfuehren" in q or "siehe Code" in q

    def test_completely_empty_finding_still_returns_question(self):
        from src.extractor import _build_second_opinion_question
        q = _build_second_opinion_question({})
        assert isinstance(q, str)
        assert len(q) > 10

    def test_all_known_types_produce_questions(self):
        from src.extractor import _build_second_opinion_question, _QUESTION_TEMPLATES
        for ftype in _QUESTION_TEMPLATES:
            q = _build_second_opinion_question({"type": ftype, "evidence_excerpt": "test"})
            assert len(q) > 20, f"Typ '{ftype}' liefert zu kurze Frage"

    def test_question_injected_into_prompt(self):
        """Verify targeted question appears in the prompt sent to second-opinion model."""
        captured_prompts = []

        def mock_generate(prompt, url, model, label):
            captured_prompts.append(prompt)
            return _raw_json("BESTÄTIGT")

        findings = [_finding(type="zombie_code", evidence_excerpt="deadFunc()")]
        with patch("src.analyzer._ollama_generate", side_effect=mock_generate):
            second_opinion_validate(findings, [_file()], "http://localhost:11434", "model")

        assert len(captured_prompts) == 1
        assert "deadFunc()" in captured_prompts[0]
        assert "aufgerufen" in captured_prompts[0].lower() or "referenziert" in captured_prompts[0].lower()
