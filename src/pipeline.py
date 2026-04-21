"""Backward-compat Re-Export Shim.

Alle Workflow-Funktionen wohnen jetzt in `src/workflows/*.py`. Dieser Shim
bleibt, damit bestehende Imports (`from src.pipeline import run_analysis`)
weiter funktionieren — insbesondere `src/cli.py` und einige Tests.

Nach außen keine Funktionsänderung. Für neue Aufrufer: direkt aus
`src.workflows.<name>` importieren.
"""
from src.workflows._common import (
    is_test_file,
    load_prompt,
    load_turbulence_cache,
)
from src.workflows.analysis import _run_overview, run_analysis
from src.workflows.image_ingest import run_ingest_images
from src.workflows.issue_ingest import run_ingest_issues
from src.workflows.memory_ingest import run_populate_memory
from src.workflows.paper_ingest import run_ingest_paper
from src.workflows.pi_task import run_pi_task
from src.workflows.turbulence import run_turbulence

__all__ = [
    "is_test_file",
    "load_prompt",
    "load_turbulence_cache",
    "run_analysis",
    "run_ingest_images",
    "run_ingest_issues",
    "run_ingest_paper",
    "run_pi_task",
    "run_populate_memory",
    "run_turbulence",
    "_run_overview",
]


if __name__ == "__main__":
    from src.cli import main
    main()
