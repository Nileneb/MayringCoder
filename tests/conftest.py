"""Shared pytest fixtures for all test modules."""

import pytest
from pathlib import Path

# Register host-side providers (embed/generate/vision) into mayring_core so the
# memory layer uses the real src.analysis/mayring_pi_agent implementations and the
# canonical patch("src.analysis...") test seams keep working (#267).
from src.provider_setup import setup_providers

setup_providers()


@pytest.fixture(autouse=True)
def _clear_dashboard_cache():
    """The /stats dashboard endpoints share a process-global TTL cache
    (api-saturation 2026-05-24). Tests reuse the same workspace_id against
    fresh per-test DBs, so a leftover entry would bleed a prior test's result
    into the next. Clear it before each test for isolation (prod keys by real
    workspace_id, so this is purely a test concern)."""
    from src.api.routes.dashboard import _DASH_CACHE
    _DASH_CACHE.clear()
    yield


@pytest.fixture(autouse=True)
def _default_test_identity(monkeypatch, tmp_path_factory):
    """Stelle eine Default-Test-Identity (User+Email) bereit, damit
    Mock-Tests (args=MagicMock) keinen IdentityRequiredError werfen.

    Test-spezifische Fixtures, die ihre eigene Identity setzen wollen,
    können MAYRING_USER_ID/Email override oder die identity.json löschen.
    """
    # Pre-launch: Email ist PFLICHT — Test-Default 'test@example.com'.
    cfg_dir = tmp_path_factory.mktemp("mayring-test-config")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    monkeypatch.setenv("MAYRING_USER_ID", "1")
    # identity.json mit Test-Email schreiben, damit local_identity().email
    # für CLI-Pfade gefüllt ist.
    import json
    mr_dir = cfg_dir / "mayring"
    mr_dir.mkdir(parents=True, exist_ok=True)
    (mr_dir / "identity.json").write_text(json.dumps({
        "user_id": 1, "email": "test@example.com", "token": None,
    }))


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
