"""Unit tests for src.analysis.splitter.smart_split()."""

import pytest
from src.analysis.splitter import smart_split


# ---------------------------------------------------------------------------
# Python block extraction
# ---------------------------------------------------------------------------

class TestSmartSplitPython:
    def test_extracts_functions_as_blocks(self):
        code = (
            "def greet(name):\n"
            "    return f'Hello, {name}'\n"
            "\n"
            "def farewell(name):\n"
            "    return f'Goodbye, {name}'\n"
        )
        result = smart_split(code, "example.py", max_chars=10000)
        names = [b["name"] for b in result["blocks"]]
        assert "greet" in names
        assert "farewell" in names

    def test_security_functions_get_higher_priority(self):
        code = (
            "def format_name(s):\n"
            "    return s.strip()\n"
            "\n"
            "def authenticate_user(username, password):\n"
            "    pass\n"
            "\n"
            "def delete_account(user_id):\n"
            "    pass\n"
        )
        result = smart_split(code, "auth.py", max_chars=10000)
        blocks_by_name = {b["name"]: b for b in result["blocks"]}
        assert blocks_by_name["authenticate_user"]["priority"] > blocks_by_name["format_name"]["priority"]
        assert blocks_by_name["delete_account"]["priority"] > blocks_by_name["format_name"]["priority"]

    def test_respects_max_chars_limit(self):
        # 50 small functions, max_chars=200
        funcs = "\n".join(
            f"def func_{i}():\n    return {i}\n"
            for i in range(50)
        )
        result = smart_split(funcs, "many.py", max_chars=200)
        total_chars = sum(len(b["text"]) for b in result["selected"])
        assert total_chars <= 200

    def test_skipped_summary_lists_omitted_functions(self):
        # Many functions so some will be skipped
        funcs = "\n".join(
            f"def func_{i}():\n    return {i}\n"
            for i in range(20)
        )
        result = smart_split(funcs, "many.py", max_chars=100)
        # Some functions must have been skipped
        assert len(result["selected"]) < len(result["blocks"])
        # skipped_summary should mention at least one function name
        assert result["skipped_summary"] != ""
        # At least one skipped name should appear in summary
        skipped_names = [b["name"] for b in result["blocks"] if b not in result["selected"]]
        assert any(name in result["skipped_summary"] for name in skipped_names)

    def test_classes_are_extracted(self):
        code = (
            "class UserManager:\n"
            "    def create(self):\n"
            "        pass\n"
            "    def delete(self):\n"
            "        pass\n"
        )
        result = smart_split(code, "manager.py", max_chars=10000)
        names = [b["name"] for b in result["blocks"]]
        assert "UserManager" in names

    def test_syntax_error_falls_back_to_truncation(self):
        bad_python = "def broken(\n    this is not valid Python @@@\n"
        result = smart_split(bad_python, "broken.py", max_chars=10000)
        assert len(result["blocks"]) == 1
        assert result["blocks"][0]["name"] == "__fallback__"
        assert len(result["selected"]) == 1


# ---------------------------------------------------------------------------
# JS block extraction
# ---------------------------------------------------------------------------

class TestSmartSplitJS:
    def test_extracts_js_functions(self):
        code = (
            "function greet(name) {\n"
            "    return 'Hello ' + name;\n"
            "}\n"
            "\n"
            "function adminDelete(id) {\n"
            "    db.delete(id);\n"
            "}\n"
        )
        result = smart_split(code, "app.js", max_chars=10000)
        names = [b["name"] for b in result["blocks"]]
        assert "greet" in names
        assert "adminDelete" in names

    def test_admin_function_higher_priority(self):
        code = (
            "function greet(name) {\n"
            "    return 'Hello ' + name;\n"
            "}\n"
            "\n"
            "function adminDelete(id) {\n"
            "    db.delete(id);\n"
            "}\n"
        )
        result = smart_split(code, "app.js", max_chars=10000)
        blocks_by_name = {b["name"]: b for b in result["blocks"]}
        assert blocks_by_name["adminDelete"]["priority"] > blocks_by_name["greet"]["priority"]


# ---------------------------------------------------------------------------
# Fallback cases
# ---------------------------------------------------------------------------

class TestSmartSplitFallback:
    def test_unknown_ext_returns_truncated(self):
        content = "col1,col2,col3\n1,2,3\n4,5,6\n"
        result = smart_split(content, "data.csv", max_chars=10000)
        assert len(result["blocks"]) == 1
        assert result["blocks"][0]["name"] == "__fallback__"
        assert result["selected"] == result["blocks"]

    def test_empty_content(self):
        result = smart_split("", "empty.py", max_chars=10000)
        assert result["blocks"] == []
        assert result["selected"] == []
        assert result["skipped_summary"] == ""
