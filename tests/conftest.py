"""Shared pytest fixtures for all test modules."""

import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def _default_test_identity(monkeypatch):
    """Stelle eine Default-User-ID bereit, damit Mock-Tests
    (args=MagicMock) keinen IdentityRequiredError werfen.

    Test-spezifische Fixtures, die ihre eigene Identity setzen wollen,
    können `monkeypatch.delenv("MAYRING_USER_ID")` aufrufen.
    """
    monkeypatch.setenv("MAYRING_USER_ID", "1")
    # Tests sollen NICHT in das echte ~/.config/mayring schreiben.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(Path("/tmp/.mayring-test-config")))


@pytest.fixture
def sample_codebook(tmp_path: Path) -> Path:
    """A minimal codebook YAML file for testing."""
    yaml = tmp_path / "codebook.yaml"
    yaml.write_text(
        "categories:\n"
        "  - name: source\n"
        "    description: Source files\n"
        "    patterns:\n"
        "      - 'src/**/*.py'\n"
        "      - 'lib/*.py'\n"
        "  - name: config\n"
        "    description: Config files\n"
        "    patterns:\n"
        "      - '*.yaml'\n"
        "      - '*.yml'\n"
        "      - '.env*'\n"
        "exclude_patterns:\n"
        "  - '*.log'\n"
        "  - '**/.git/**'\n"
    )
    return yaml


@pytest.fixture
def mayringignore_file(tmp_path: Path) -> Path:
    """A .mayringignore file for testing."""
    f = tmp_path / ".mayringignore"
    f.write_text("# This is a comment\n\n*.tmp\n  # another comment\n*.swp\n")
    return f
