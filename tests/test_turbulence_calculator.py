"""Unit tests for src.turbulence_calculator (Issue #12 — split turbulence_analyzer)."""

import pytest
from pathlib import Path

from src.turbulence_calculator import (
    Chunk,
    FileAnalysis,
    Redundancy,
    CATEGORIES,
    SIMILARITY_THRESHOLD,
    THRESHOLD_DEEP,
    THRESHOLD_SKIP,
    MIN_CHUNKS_FOR_TRIAGE,
    calculate_turbulence,
    categorize_chunk_heuristic,
    chunkify,
    find_redundancies,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    file: str = "f.py",
    start: int = 1,
    end: int = 10,
    code: str = "",
    category: str = "Logik",
    functional_name: str = "block_1",
) -> Chunk:
    return Chunk(
        file=file,
        start_line=start,
        end_line=end,
        code=code,
        category=category,
        functional_name=functional_name,
    )


# ---------------------------------------------------------------------------
# chunkify
# ---------------------------------------------------------------------------

class TestChunkify:
    def test_nonexistent_file_returns_empty(self, tmp_path):
        result = chunkify(str(tmp_path / "does_not_exist.py"))
        assert result == []

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        assert chunkify(str(f)) == []

    def test_small_file_produces_one_chunk(self, tmp_path):
        f = tmp_path / "small.py"
        f.write_text("\n".join(f"line{i}" for i in range(5)))
        chunks = chunkify(str(f), chunk_size=15)
        assert len(chunks) == 1

    def test_large_file_produces_multiple_chunks(self, tmp_path):
        f = tmp_path / "big.py"
        f.write_text("\n".join(f"line{i}" for i in range(100)))
        chunks = chunkify(str(f), chunk_size=10)
        assert len(chunks) > 1

    def test_chunks_cover_all_lines(self, tmp_path):
        n_lines = 47
        f = tmp_path / "file.py"
        f.write_text("\n".join(f"x{i}" for i in range(n_lines)))
        chunks = chunkify(str(f), chunk_size=10)
        # Every line should appear in exactly one chunk
        covered = set()
        for c in chunks:
            covered.update(range(c.start_line, c.end_line + 1))
        assert len(covered) == n_lines

    def test_chunk_has_correct_file_attribute(self, tmp_path):
        f = tmp_path / "myfile.py"
        f.write_text("a\nb\nc\n")
        chunks = chunkify(str(f))
        assert all(c.file == str(f) for c in chunks)

    def test_chunk_code_is_not_empty_for_nonempty_file(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def foo():\n    return 1\n")
        chunks = chunkify(str(f))
        assert all(c.code.strip() for c in chunks)


# ---------------------------------------------------------------------------
# categorize_chunk_heuristic
# ---------------------------------------------------------------------------

class TestCategorizeChunkHeuristic:
    def test_detects_daten(self):
        chunk = _make_chunk(code="User::create(['name' => $name])")
        categorize_chunk_heuristic(chunk)
        assert chunk.category == "Daten"

    def test_detects_ki(self):
        chunk = _make_chunk(code="$response = ollama()->chat($messages);")
        categorize_chunk_heuristic(chunk)
        assert chunk.category == "KI"

    def test_detects_ui(self):
        chunk = _make_chunk(code="<div wire:model='name' class='form-control'>")
        categorize_chunk_heuristic(chunk)
        assert chunk.category == "UI"

    def test_detects_sicherheit(self):
        chunk = _make_chunk(code="$this->authorize('view', $model);")
        categorize_chunk_heuristic(chunk)
        assert chunk.category == "Sicherheit"

    def test_defaults_to_logik_for_plain_code(self):
        chunk = _make_chunk(code="$result = $a + $b * 2;")
        categorize_chunk_heuristic(chunk)
        assert chunk.category == "Logik"

    def test_sets_functional_name_from_function_keyword(self):
        chunk = _make_chunk(code="function calculateTotal($items) {\n    return 0;\n}")
        categorize_chunk_heuristic(chunk)
        assert chunk.functional_name == "calculateTotal"

    def test_sets_block_name_when_no_function(self):
        chunk = _make_chunk(code="$x = 1;", start=42)
        categorize_chunk_heuristic(chunk)
        assert chunk.functional_name == "block_42"

    def test_category_is_always_valid(self):
        for code in ["", "random text", "<?php echo 'hi';"]:
            chunk = _make_chunk(code=code)
            categorize_chunk_heuristic(chunk)
            assert chunk.category in CATEGORIES


# ---------------------------------------------------------------------------
# calculate_turbulence
# ---------------------------------------------------------------------------

class TestCalculateTurbulence:
    def test_single_chunk_returns_zero(self):
        chunks = [_make_chunk(category="Logik")]
        score, zones = calculate_turbulence(chunks)
        assert score == 0.0
        assert zones == []

    def test_uniform_categories_produce_low_score(self):
        chunks = [_make_chunk(category="Logik") for _ in range(10)]
        score, zones = calculate_turbulence(chunks)
        assert score < THRESHOLD_SKIP

    def test_highly_mixed_categories_produce_high_score(self):
        cats = ["Daten", "UI", "KI", "Config", "Sicherheit"] * 4
        chunks = [_make_chunk(category=c) for c in cats]
        score, zones = calculate_turbulence(chunks)
        assert score > THRESHOLD_SKIP

    def test_score_is_between_zero_and_one(self):
        cats = ["Daten", "UI", "Logik", "KI"] * 5
        chunks = [_make_chunk(category=c) for c in cats]
        score, _ = calculate_turbulence(chunks)
        assert 0.0 <= score <= 1.0

    def test_hot_zones_only_when_score_exceeds_threshold(self):
        # All same category → no hot zones
        chunks = [_make_chunk(category="Logik") for _ in range(20)]
        _, zones = calculate_turbulence(chunks)
        assert zones == []

    def test_hot_zone_fields_present(self):
        cats = ["Daten", "UI", "KI", "Config", "Sicherheit"] * 6
        chunks = [_make_chunk(category=c, start=i * 5 + 1, end=(i + 1) * 5)
                  for i, c in enumerate(cats)]
        _, zones = calculate_turbulence(chunks)
        for z in zones:
            assert "start_line" in z
            assert "end_line" in z
            assert "peak_score" in z


# ---------------------------------------------------------------------------
# find_redundancies
# ---------------------------------------------------------------------------

class TestFindRedundancies:
    def test_empty_chunks_returns_empty(self):
        assert find_redundancies([]) == []

    def test_no_redundancy_for_same_file(self):
        chunks = [
            _make_chunk(file="a.py", functional_name="save_user"),
            _make_chunk(file="a.py", functional_name="save_user"),
        ]
        assert find_redundancies(chunks) == []

    def test_detects_similar_names_across_files(self):
        chunks = [
            _make_chunk(file="a.py", functional_name="save_user"),
            _make_chunk(file="b.py", functional_name="save_user"),
        ]
        result = find_redundancies(chunks)
        assert len(result) == 1
        assert result[0].similarity == 1.0

    def test_does_not_flag_dissimilar_names(self):
        chunks = [
            _make_chunk(file="a.py", functional_name="fetch_invoice"),
            _make_chunk(file="b.py", functional_name="send_email"),
        ]
        result = find_redundancies(chunks)
        assert result == []

    def test_skips_block_names(self):
        chunks = [
            _make_chunk(file="a.py", functional_name="block_1"),
            _make_chunk(file="b.py", functional_name="block_1"),
        ]
        assert find_redundancies(chunks) == []

    def test_skips_error_names(self):
        chunks = [
            _make_chunk(file="a.py", functional_name="fehler_Timeout"),
            _make_chunk(file="b.py", functional_name="fehler_Timeout"),
        ]
        assert find_redundancies(chunks) == []

    def test_skips_unbekannt(self):
        chunks = [
            _make_chunk(file="a.py", functional_name="unbekannt"),
            _make_chunk(file="b.py", functional_name="unbekannt"),
        ]
        assert find_redundancies(chunks) == []

    def test_skips_nicht_erkannt(self):
        """LLM returned 'nicht_erkannt' for both chunks — must not create false redundancy."""
        chunks = [
            _make_chunk(file="a.py", functional_name="nicht_erkannt"),
            _make_chunk(file="b.py", functional_name="nicht_erkannt"),
        ]
        assert find_redundancies(chunks) == []

    def test_sorted_by_descending_similarity(self):
        chunks = [
            _make_chunk(file="a.py", functional_name="save_user"),
            _make_chunk(file="b.py", functional_name="save_user"),   # sim=1.0
            _make_chunk(file="c.py", functional_name="save_users"),  # sim<1.0
            _make_chunk(file="d.py", functional_name="save_users"),
        ]
        result = find_redundancies(chunks)
        sims = [r.similarity for r in result]
        assert sims == sorted(sims, reverse=True)

    def test_no_duplicate_pairs(self):
        chunks = [
            _make_chunk(file="a.py", functional_name="load_data"),
            _make_chunk(file="b.py", functional_name="load_data"),
        ]
        result = find_redundancies(chunks)
        pairs = [(r.file_a, r.file_b) for r in result]
        assert len(pairs) == len(set(pairs))


# ---------------------------------------------------------------------------
# turbulence_report integration: build_report output shape
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_returns_expected_keys(self):
        from src.turbulence_report import build_report
        analyses = [
            FileAnalysis(path="a.py", total_lines=10, tier="skip",
                         turbulence_score=0.1),
        ]
        report = build_report(analyses, [])
        assert "summary" in report
        assert "critical_files" in report
        assert "redundancies" in report

    def test_summary_counts_tiers(self):
        from src.turbulence_report import build_report
        analyses = [
            FileAnalysis(path="a.py", total_lines=5, tier="skip", turbulence_score=0.1),
            FileAnalysis(path="b.py", total_lines=5, tier="light", turbulence_score=0.3),
            FileAnalysis(path="c.py", total_lines=5, tier="deep", turbulence_score=0.7),
        ]
        report = build_report(analyses, [])
        assert report["summary"]["stable"] == 1
        assert report["summary"]["medium"] == 1
        assert report["summary"]["critical"] == 1

    def test_build_markdown_returns_string(self):
        from src.turbulence_report import build_report, build_markdown
        analyses = [
            FileAnalysis(path="x.py", total_lines=20, tier="light", turbulence_score=0.25),
        ]
        report = build_report(analyses, [])
        md = build_markdown(report, "https://github.com/test/repo", "mistral", 5.0)
        assert isinstance(md, str)
        assert "Turbulenz-Analyse" in md
        assert "Zusammenfassung" in md

    def test_stable_tier_counted_in_summary(self):
        """Files with < MIN_CHUNKS_FOR_TRIAGE should be counted in stable summary."""
        from src.turbulence_report import build_report
        analyses = [
            FileAnalysis(path="tiny.py", total_lines=5, tier="stable", turbulence_score=0.0),
            FileAnalysis(path="normal.py", total_lines=50, tier="deep", turbulence_score=0.7),
        ]
        report = build_report(analyses, [])
        assert report["summary"]["stable"] == 1
        assert report["summary"]["critical"] == 1

    def test_stable_and_skip_both_counted_as_stable(self):
        from src.turbulence_report import build_report
        analyses = [
            FileAnalysis(path="a.py", total_lines=5, tier="stable", turbulence_score=0.0),
            FileAnalysis(path="b.py", total_lines=20, tier="skip", turbulence_score=0.1),
        ]
        report = build_report(analyses, [])
        assert report["summary"]["stable"] == 2


# ---------------------------------------------------------------------------
# MIN_CHUNKS_FOR_TRIAGE constant
# ---------------------------------------------------------------------------

class TestMinChunksForTriage:
    def test_constant_is_at_least_3(self):
        assert MIN_CHUNKS_FOR_TRIAGE >= 3

    def test_analyze_repo_marks_tiny_file_as_stable(self, tmp_path):
        """A file that produces < MIN_CHUNKS_FOR_TRIAGE chunks must get tier='stable'."""
        from src.turbulence_analyzer import analyze_repo

        # Write a tiny file (< 30 lines → at most 2 chunks with default chunk_size=15)
        tiny = tmp_path / "tiny.py"
        tiny.write_text("\n".join(f"line{i}" for i in range(20)))

        report = analyze_repo(str(tmp_path), use_llm=False)
        # Should not appear in critical_files
        critical_paths = [f["path"] for f in report.get("critical_files", [])]
        assert "tiny.py" not in critical_paths

    def test_model_param_passed_to_report(self, tmp_path):
        """analyze_repo with model param runs without error and returns report dict."""
        from src.turbulence_analyzer import analyze_repo

        f = tmp_path / "sample.py"
        f.write_text("\n".join(f"x = {i}" for i in range(50)))

        report = analyze_repo(str(tmp_path), use_llm=False, model="some-model")
        assert "summary" in report
