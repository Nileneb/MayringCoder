"""Unit tests for src.github.input.parse_github_input."""
from __future__ import annotations

import pytest

from src.github import GitHubInput, GitHubInputError, parse_github_input


class TestSlugForm:
    def test_simple_slug(self) -> None:
        g = parse_github_input("Nileneb/MayringCoder")
        assert g.owner == "Nileneb"
        assert g.repo == "MayringCoder"
        assert g.branch is None
        assert g.slug == "Nileneb/MayringCoder"
        assert g.url == "https://github.com/Nileneb/MayringCoder"
        assert g.clone_url == "https://github.com/Nileneb/MayringCoder.git"

    def test_slug_with_dots_and_dashes(self) -> None:
        g = parse_github_input("a-b/c.d-e_f")
        assert g.owner == "a-b"
        assert g.repo == "c.d-e_f"

    def test_slug_strips_git_suffix(self) -> None:
        g = parse_github_input("owner/repo.git")
        assert g.repo == "repo"


class TestHttpsUrl:
    def test_plain_https(self) -> None:
        g = parse_github_input("https://github.com/Nileneb/MayringCoder")
        assert g.owner == "Nileneb"
        assert g.repo == "MayringCoder"
        assert g.branch is None

    def test_with_git_suffix(self) -> None:
        g = parse_github_input("https://github.com/Nileneb/MayringCoder.git")
        assert g.repo == "MayringCoder"

    def test_with_trailing_slash(self) -> None:
        g = parse_github_input("https://github.com/Nileneb/MayringCoder/")
        assert g.repo == "MayringCoder"

    def test_with_branch_path(self) -> None:
        g = parse_github_input("https://github.com/Nileneb/MayringCoder/tree/develop")
        assert g.branch == "develop"

    def test_with_www_subdomain(self) -> None:
        g = parse_github_input("https://www.github.com/Nileneb/MayringCoder")
        assert g.owner == "Nileneb"

    def test_with_query_string(self) -> None:
        g = parse_github_input("https://github.com/Nileneb/MayringCoder?foo=1")
        assert g.repo == "MayringCoder"


class TestSshUrl:
    def test_ssh(self) -> None:
        g = parse_github_input("git@github.com:Nileneb/MayringCoder.git")
        assert g.slug == "Nileneb/MayringCoder"

    def test_ssh_without_git_suffix(self) -> None:
        g = parse_github_input("git@github.com:Nileneb/MayringCoder")
        assert g.slug == "Nileneb/MayringCoder"


class TestRejects:
    def test_empty_string(self) -> None:
        with pytest.raises(GitHubInputError, match="Bitte GitHub-Repo"):
            parse_github_input("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(GitHubInputError):
            parse_github_input("   ")

    def test_none(self) -> None:
        with pytest.raises(GitHubInputError):
            parse_github_input(None)  # type: ignore[arg-type]

    def test_random_text(self) -> None:
        with pytest.raises(GitHubInputError, match="nicht erkennen"):
            parse_github_input("hello world")

    def test_not_github_host(self) -> None:
        with pytest.raises(GitHubInputError):
            parse_github_input("https://gitlab.com/foo/bar")

    def test_single_segment(self) -> None:
        with pytest.raises(GitHubInputError):
            parse_github_input("Nileneb")


class TestInputNormalization:
    def test_strip_surrounding_whitespace(self) -> None:
        g = parse_github_input("  owner/repo  \n")
        assert g.slug == "owner/repo"
