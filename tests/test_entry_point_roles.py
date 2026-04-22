"""Guard: zwei Entry-Points (`checker.py` HTTP-Client, `python -m src.cli`
Direkt-Executor) müssen distinkte Rollen behalten. Issue #66 Phase 5 hat
diese Struktur als absichtlich doppelten Entry-Point eingeordnet; der
Regression-Test stellt sicher, dass ihre Module-Docstrings die Rollen
benennen, damit niemand sie versehentlich zusammenführt.
"""
from __future__ import annotations

import ast
import pathlib


def _docstring_of_file(path: pathlib.Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return ast.get_docstring(tree) or ""


class TestRoleMarkers:
    def test_checker_is_http_client(self):
        repo = pathlib.Path(__file__).resolve().parent.parent
        doc = _docstring_of_file(repo / "checker.py")
        assert "HTTP" in doc and "Client" in doc, \
            "checker.py-Docstring muss die Rolle 'HTTP-Client' benennen"
        assert "checker.py" in doc

    def test_src_cli_is_direct_executor(self):
        repo = pathlib.Path(__file__).resolve().parent.parent
        doc = _docstring_of_file(repo / "src" / "cli.py")
        assert "Direkt-Executor" in doc or "subprocess" in doc, \
            "src/cli.py-Docstring muss die Rolle 'Direkt-Executor' benennen"
        assert "checker.py" in doc, \
            "src/cli.py-Docstring soll auf checker.py als Gegenstück verweisen"


class TestEntryPointsAreStillCallable:
    def test_checker_loads_as_module(self):
        import checker  # noqa: F401

    def test_src_cli_loads_as_module(self):
        import src.cli  # noqa: F401
