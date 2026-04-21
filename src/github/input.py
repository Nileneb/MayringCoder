"""Parse and normalise GitHub repository references.

One source of truth for: URL variants, owner/repo slugs, optional branch,
and the raw/API URLs other MayringCoder modules need. Callers never have
to touch regex themselves — they pass whatever the user typed and get
back a ``GitHubInput`` or an error message.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


class GitHubInputError(ValueError):
    """Raised when a user-supplied GitHub reference cannot be parsed."""


_SLUG_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9-]{0,38})/([A-Za-z0-9_.-]{1,100})$")

_URL_PATTERNS = [
    # https://github.com/owner/repo          (.git / trailing slash optional)
    re.compile(
        r"^https?://(?:www\.)?github\.com/"
        r"(?P<owner>[A-Za-z0-9][A-Za-z0-9-]{0,38})/"
        r"(?P<repo>[A-Za-z0-9_.-]{1,100})"
        r"(?:\.git)?/?"
        r"(?:/tree/(?P<branch>[^/?#]+))?"
        r"/?(?:[?#].*)?$"
    ),
    # git@github.com:owner/repo.git
    re.compile(
        r"^git@github\.com:"
        r"(?P<owner>[A-Za-z0-9][A-Za-z0-9-]{0,38})/"
        r"(?P<repo>[A-Za-z0-9_.-]{1,100})"
        r"(?:\.git)?$"
    ),
]


@dataclass(frozen=True)
class GitHubInput:
    """Normalised GitHub reference.

    ``slug``  — canonical ``owner/repo`` form used by gh-CLI and as display label
    ``url``   — canonical HTTPS clone URL (no .git suffix, no branch)
    ``branch``— optional branch / tag (only set when the input embedded one)
    """

    owner: str
    repo: str
    branch: str | None = None

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"

    @property
    def clone_url(self) -> str:
        return self.url + ".git"


def parse_github_input(raw: str) -> GitHubInput:
    """Parse a GitHub reference. Accepts:

        owner/repo                       (slug)
        https://github.com/owner/repo    (URL, optional .git, /tree/branch)
        git@github.com:owner/repo.git    (SSH)

    Raises ``GitHubInputError`` with a human-readable German message if the
    input is empty or doesn't match any supported form.
    """
    text = (raw or "").strip()
    if not text:
        raise GitHubInputError("Bitte GitHub-Repo angeben (owner/repo oder https://github.com/owner/repo).")

    # Strip trailing .git once for slug-form inputs
    slug_match = _SLUG_PATTERN.match(text)
    if slug_match:
        owner, repo = slug_match.group(1), slug_match.group(2)
        if repo.endswith(".git"):
            repo = repo[:-4]
        return GitHubInput(owner=owner, repo=repo)

    for pattern in _URL_PATTERNS:
        m = pattern.match(text)
        if m:
            repo = m.group("repo")
            if repo.endswith(".git"):
                repo = repo[:-4]
            branch = m.groupdict().get("branch") or None
            return GitHubInput(owner=m.group("owner"), repo=repo, branch=branch)

    raise GitHubInputError(
        f"Konnte GitHub-Repo nicht erkennen: {text!r}. "
        f"Erlaubt: 'owner/repo' oder 'https://github.com/owner/repo'."
    )
