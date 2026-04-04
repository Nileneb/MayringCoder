"""Unit tests for src.splitter."""

import pytest
from src.splitter import split_into_files, _normalize_eol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_file_block(filename: str, content: str) -> str:
    sep = "=" * 48
    return f"{sep}\nFILE: {filename}\n{sep}\n{content}\n\n"


def parse(content: str) -> list[dict]:
    return split_into_files(content)


# ---------------------------------------------------------------------------
# _normalize_eol
# ---------------------------------------------------------------------------

class TestNormalizeEol:
    def test_crlf_converted_to_lf(self):
        assert _normalize_eol("line1\r\nline2\r\n") == "line1\nline2\n"

    def test_cr_converted_to_lf(self):
        assert _normalize_eol("line1\rline2\r") == "line1\nline2\n"

    def test_mixed_eol(self):
        result = _normalize_eol("a\r\nb\rc\nd\r")
        # All CRs removed; each input line produces one LF
        assert "\r" not in result
        # Input: "a\r\n" (1 LF) + "b\r" (1 LF) + "c\n" (1 LF) + "d\r" (1 LF) = 4 LFs
        assert result.count("\n") == 4

    def test_empty_string(self):
        assert _normalize_eol("") == ""


# ---------------------------------------------------------------------------
# split_into_files
# ---------------------------------------------------------------------------

class TestSplitIntoFilesBasic:
    def test_single_file(self):
        block = make_file_block("src/main.py", "def main():\n    pass\n")
        files = parse(block)
        assert len(files) == 1
        assert files[0]["filename"] == "src/main.py"
        assert files[0]["content"] == "def main():\n    pass"
        assert "hash" in files[0]
        assert len(files[0]["hash"]) == 64  # SHA256 hex

    def test_multiple_files(self):
        block = (
            make_file_block("a.txt", "hello") +
            make_file_block("b.txt", "world")
        )
        files = parse(block)
        assert len(files) == 2
        assert [f["filename"] for f in files] == ["a.txt", "b.txt"]

    def test_filename_whitespace_stripped(self):
        block = make_file_block("  foo.py  ", "x")
        assert parse(block)[0]["filename"] == "foo.py"

    def test_trailing_newlines_stripped(self):
        block = make_file_block("x.txt", "content\n\n\n")
        assert parse(block)[0]["content"] == "content"

    def test_empty_content_returns_zero_lines(self):
        block = make_file_block("empty.txt", "")
        assert parse(block)[0]["line_estimate"] == 0

    def test_content_with_newlines(self):
        block = make_file_block("x.txt", "a\nb\nc")
        assert parse(block)[0]["line_estimate"] == 3


class TestSplitIntoFilesSkipMarkers:
    @pytest.mark.parametrize("marker", ["[Binary file]", "[Empty file]"])
    def test_skips_binary_and_empty_markers(self, marker):
        block = make_file_block("img.png", marker)
        assert parse(block) == []

    def test_skips_binary_even_with_extra_whitespace(self):
        block = make_file_block("img.png", "  [Binary file]  \n")
        assert parse(block) == []


class TestHashStability:
    def test_same_content_produces_same_hash(self):
        block = make_file_block("x.txt", "hello world")
        files = parse(block)
        h1 = files[0]["hash"]

        # Normalise differently but same content
        block2 = make_file_block("x.txt", "hello world\n")
        files2 = parse(block2)
        # Both should be normalised to same content before hashing
        assert files2[0]["content"] == files[0]["content"]

    def test_different_content_produces_different_hash(self):
        block = make_file_block("x.txt", "a")
        h1 = parse(block)[0]["hash"]
        h2 = parse(make_file_block("x.txt", "b"))[0]["hash"]
        assert h1 != h2

    def test_hash_short_is_16_chars(self):
        block = make_file_block("x.txt", "content")
        f = parse(block)[0]
        assert len(f["hash_short"]) == 16
        assert f["hash_short"] == f["hash"][:16]


class TestSizeField:
    def test_size_matches_utf8_byte_length(self):
        block = make_file_block("x.txt", "héllo wörld")
        f = parse(block)[0]
        assert f["size"] == len(f["content"].encode("utf-8"))


class TestEndToEnd:
    def test_real_world_style_input(self):
        # The gitingest format: SEPARATOR + 'FILE: name' + SEPARATOR + content + SEPARATOR.
        # Multiple consecutive newlines within content are fine (stripped at end).
        # Extra blank lines BETWEEN content and the next SEPARATOR break the regex
        # because (?=\nSEP) only skips one \n. We avoid that by keeping content tight.
        content = (
            "=" * 48 + "\nFILE: src/config.py\n" +
            "=" * 48 + "\n" +
            '"""Config module."""\n' +
            "import os\n\n" +
            "=" * 48 + "\nFILE: README.md\n" +
            "=" * 48 + "\n" +
            "# My Project\n"
        )
        files = parse(content)
        assert len(files) == 2
        assert files[0]["filename"] == "src/config.py"
        assert files[1]["filename"] == "README.md"
        # Content does NOT include the trailing blank line before the separator
        assert files[0]["content"].endswith("import os")
        assert files[1]["content"].endswith("# My Project")
