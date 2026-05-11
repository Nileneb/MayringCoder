"""Tests for tools/cleanup_hallucinated_categories — _process_labels + promote-known."""

from __future__ import annotations

from tools.cleanup_hallucinated_categories import (
    _process_labels, is_valid_neu_label, strip_neu_labels,
)


def test_strip_strict_removes_all_neu():
    out, removed = strip_neu_labels("api, [neu]observability, data_access, [neu]gibberish", strict=True)
    assert out == "api,data_access"
    assert removed == 2


def test_strip_non_strict_keeps_valid_neu_drops_invalid():
    # [neu]ai is valid (2 chars, alphanumeric); [neu]zzzzz fails the vowel/repeat heuristic
    out, removed = strip_neu_labels("[neu]ai, [neu]zzzzzzz, api", strict=False)
    assert "[neu]ai" in out
    assert "[neu]zzzzzzz" not in out
    assert "api" in out
    assert removed == 1


def test_promote_known_rewrites_neu_to_plain():
    out, removed = _process_labels(
        "api, [neu]observability, data_access",
        strict=False, promote_known={"observability", "monitoring"},
    )
    assert out == "api,observability,data_access"
    assert removed == 0  # promotion is a rewrite, not a removal


def test_promote_known_plus_strict_drops_unknown_neu():
    out, removed = _process_labels(
        "[neu]observability, [neu]customthing, api",
        strict=True, promote_known={"observability"},
    )
    assert out == "observability,api"  # [neu]customthing dropped (strict, not in codebook)
    assert removed == 1


def test_promote_known_dedups_against_existing_plain():
    # If chunk already has 'observability' AND '[neu]observability' → only one kept
    out, _ = _process_labels(
        "observability, [neu]observability, api",
        strict=False, promote_known={"observability"},
    )
    assert out == "observability,api"


def test_promote_known_only_no_strict_keeps_valid_neu():
    out, removed = _process_labels(
        "[neu]observability, [neu]customvalid, api",
        strict=False, promote_known={"observability"},
    )
    assert out == "observability,[neu]customvalid,api"
    assert removed == 0


def test_no_promote_no_strict_is_passthrough_for_valid():
    out, removed = _process_labels("api, [neu]ai, data_access", strict=False, promote_known=None)
    assert out == "api,[neu]ai,data_access"
    assert removed == 0


def test_is_valid_neu_label():
    assert is_valid_neu_label("observability")
    assert is_valid_neu_label("ai")
    assert is_valid_neu_label("machine_learning")
    assert not is_valid_neu_label("x")  # too short
    assert not is_valid_neu_label("")  # empty
    assert not is_valid_neu_label("a!b@c")  # bad chars
    assert not is_valid_neu_label("aaaaaaaaa")  # repeat heuristic


def test_load_universal_categories_includes_promoted():
    from tools.cleanup_hallucinated_categories import _load_universal_categories
    cats = _load_universal_categories()
    # promoted in PR — these should now be anchors
    assert "observability" in cats
    assert "healthcare" in cats
    assert "research" in cats
    # original ones still there
    assert "api" in cats
    assert "data_access" in cats
