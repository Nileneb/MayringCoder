"""Fetch repository content via gitingest."""

from gitingest import ingest


class RepoNotFoundError(RuntimeError):
    """Repo existiert nicht / nicht zugreifbar.

    Caller (populate-Endpoint, workflows) sollen daraus
    retrieval_status='source_not_found' machen statt den 30-Zeilen-
    Stack-Trace ins Job-Output zu leaken.
    """


def fetch_repo(repo_url: str, token: str | None = None) -> tuple[str, str, str]:
    kwargs: dict = {"source": repo_url}
    if token:
        kwargs["token"] = token
    try:
        summary, tree, content = ingest(**kwargs)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if (
            "could not read username" in msg
            or "terminal prompts disabled" in msg
            or "repository not found" in msg
            or "not found" in msg
            or "fatal:" in msg
        ):
            raise RepoNotFoundError(
                f"Repository {repo_url} nicht erreichbar — existiert "
                f"nicht oder ist privat ohne Token. Original: {str(exc)[:200]}"
            ) from exc
        raise
    return summary, tree, content
