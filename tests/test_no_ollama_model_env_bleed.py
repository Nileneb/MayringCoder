"""Issue #88 regression guard — OLLAMA_MODEL env-bleed.

Acceptance from #88: ``ModelRouter`` is the only path for resolving the
text/vision/embedding model. Direct ``os.getenv("OLLAMA_MODEL")`` reads
in production code bypass /stats/admin/model-routes runtime overrides
(Issue #140) so admins can no longer change the model without a restart.

This test grep's the source tree once and asserts there are zero direct
``os.getenv("OLLAMA_MODEL")`` reads in src/. Pytests, ad-hoc scripts and
docs are exempt — the rule is "src/ uses ModelRouter, period."

The original Issue #88 fix removed the read from src/model_router.py.
A later round of new admin endpoints reintroduced it in 4 files
(pi_worker.py, stats_admin.py, mcp_second_opinion.py,
wiki_second_opinion.py) — caught by post-deploy audit, fixed in the
same commit that added this test. Without the test, the next feature
that needs a default model name will silently re-introduce it again.
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).parent.parent / "src"
_PATTERN = re.compile(r'os\.getenv\(\s*["\']OLLAMA_MODEL["\']')


def test_no_direct_ollama_model_env_reads_in_src() -> None:
    offenders: list[str] = []
    for path in _ROOT.rglob("*.py"):
        # ModelRouter itself is allowed to handle the env layer if needed;
        # the rule applies to *callers*. Today ModelRouter doesn't read
        # OLLAMA_MODEL either, but exempting the file keeps the test
        # forward-compatible if that ever changes.
        if path.name == "model_router.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            if _PATTERN.search(line):
                offenders.append(f"{path.relative_to(_ROOT.parent)}:{line_no}: {line.strip()}")
    assert not offenders, (
        "Issue #88 regression — direct OLLAMA_MODEL env read in src/. "
        "Use ModelRouter.resolve(task) instead.\n"
        + "\n".join(offenders)
    )
