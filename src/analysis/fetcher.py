"""Fetch repository content via gitingest."""

from gitingest import ingest


def fetch_repo(repo_url: str, token: str | None = None) -> tuple[str, str, str]:
    kwargs: dict = {"source": repo_url}
    if token:
        kwargs["token"] = token
    summary, tree, content = ingest(**kwargs)
    return summary, tree, content
