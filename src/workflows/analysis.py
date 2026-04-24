"""Haupt-Analyse-Workflow — fetch → split → filter → categorize → diff → analyze → report.

Enthält sowohl den vollen Analyse-Modus als auch den Overview-Modus (Helper
`_run_overview`, der nur innerhalb von run_analysis aufgerufen wird).
"""
from src.workflows.analysis_main import *
from src.workflows.analysis_overview import *

# _run_overview starts with _ so * won't export it — explicit re-export for pipeline.py
from src.workflows.analysis_overview import _run_overview
