"""Tests for the P4 search-string review Pi-Task (#261) — pure logic."""
from src.workflows.searchstring_review import build_prompt, parse_response


def test_build_prompt_fills_placeholders():
    p = build_prompt(
        searchstring="diabetes AND therapy",
        database="PubMed",
        forschungsfrage="Wirkt Therapie X bei Diabetes Typ 2?",
    )
    assert "diabetes AND therapy" in p
    assert "PubMed" in p
    assert "Wirkt Therapie X bei Diabetes Typ 2?" in p
    assert "{{" not in p  # all placeholders filled


def test_build_prompt_marks_empty_inputs():
    p = build_prompt("foo", database="", forschungsfrage="")
    assert "(unbekannt)" in p
    assert "(keine angegeben)" in p


def test_parse_response_extracts_json():
    raw = '{"revised": "x AND y", "reasoning": "Synonyme ergänzt"}'
    out = parse_response(raw)
    assert out["revised"] == "x AND y"
    assert out["reasoning"] == "Synonyme ergänzt"
    assert out["parsed"] is True


def test_parse_response_handles_code_fence_and_prose():
    raw = 'Hier mein Vorschlag:\n```json\n{"revised": "a OR b", "reasoning": "r"}\n```'
    out = parse_response(raw)
    assert out["revised"] == "a OR b"
    assert out["parsed"] is True


def test_parse_response_fallback_to_raw_when_no_json():
    out = parse_response("nur freitext ohne json")
    assert out["revised"] == "nur freitext ohne json"
    assert out["reasoning"] == ""
    assert out["parsed"] is False


def test_parse_response_empty():
    out = parse_response("")
    assert out["revised"] == ""
    assert out["parsed"] is False
