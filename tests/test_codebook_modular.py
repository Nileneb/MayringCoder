"""Tests for modular codebook loading and profile auto-detection."""

import pytest
from src.categorizer import load_codebook_modular, detect_profile


class TestLoadCodebookModular:
    def test_loads_generic_profile(self):
        excludes, cats = load_codebook_modular("generic")
        assert len(excludes) > 0
        assert len(cats) > 0
        cat_names = [c["name"] for c in cats]
        assert "api" in cat_names
        assert "domain" in cat_names

    def test_loads_laravel_profile(self):
        excludes, cats = load_codebook_modular("laravel")
        cat_names = [c["name"] for c in cats]
        assert "laravel_livewire" in cat_names
        assert "laravel_filament" in cat_names
        # Laravel excludes should include blade
        assert any("blade" in p for p in excludes)

    def test_loads_sozialforschung_profile(self):
        excludes, cats = load_codebook_modular("sozialforschung")
        cat_names = [c["name"] for c in cats]
        assert "argumentation" in cat_names
        assert "methodik" in cat_names

    def test_categories_have_risk_level(self):
        _, cats = load_codebook_modular("generic")
        for cat in cats:
            assert "risk_level" in cat, f"Missing risk_level in {cat['name']}"

    def test_nonexistent_profile_falls_back(self):
        excludes, cats = load_codebook_modular("nonexistent_xyz")
        # Should fall back to monolithic codebook.yaml
        assert len(excludes) > 0 or len(cats) > 0

    def test_python_profile(self):
        excludes, cats = load_codebook_modular("python")
        cat_names = [c["name"] for c in cats]
        assert "api" in cat_names
        # Python profile should NOT have laravel categories
        assert "laravel_livewire" not in cat_names


class TestDetectProfile:
    def test_detects_laravel(self):
        files = [{"filename": "artisan"}, {"filename": "app/Http/Controller.php"}]
        assert detect_profile(files) == "laravel"

    def test_detects_python(self):
        files = [{"filename": "setup.py"}, {"filename": "src/main.py"}]
        assert detect_profile(files) == "python"

    def test_detects_python_pyproject(self):
        files = [{"filename": "pyproject.toml"}, {"filename": "app.py"}]
        assert detect_profile(files) == "python"

    def test_generic_fallback(self):
        files = [{"filename": "index.html"}, {"filename": "style.css"}]
        assert detect_profile(files) == "generic"

    def test_empty_files(self):
        assert detect_profile([]) == "generic"

    def test_blade_triggers_laravel(self):
        files = [{"filename": "resources/views/app.blade.php"}]
        assert detect_profile(files) == "laravel"
