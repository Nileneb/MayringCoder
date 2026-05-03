"""Recap indexer — issue-centric view that joins IGIO chunks, plans, and git.

A recap binds together what was raised (Issue), what was planned/changed
(Intervention), and what came out (Outcome) for a single GitHub issue. The
index is a pure data-gatherer; rendering lives in `recap_renderer.py`.

Inputs (already-on-disk data, no LLM calls):
    - chunks rows where igio_axis matches and source_id/text references the issue
    - plan markdowns under `~/.claude/plans/` that mention the issue
    - git log commits since the plan's mtime

Output: a `Recap` dataclass — caller renders it.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.memory.db_adapter import DBAdapter

PLANS_DIR_DEFAULT = Path.home() / ".claude" / "plans"


@dataclass
class RecapChunk:
    chunk_id: str
    source_id: str
    text_preview: str
    category_labels: list[str]
    igio_axis: str
    igio_confidence: float


@dataclass
class RecapPlan:
    path: Path
    title: str
    mtime_iso: str


@dataclass
class RecapCommit:
    sha: str
    subject: str
    iso_date: str


@dataclass
class Recap:
    issue_id: str
    workspace_id: str
    issue_chunks: list[RecapChunk] = field(default_factory=list)
    goal_chunks: list[RecapChunk] = field(default_factory=list)
    intervention_chunks: list[RecapChunk] = field(default_factory=list)
    outcome_chunks: list[RecapChunk] = field(default_factory=list)
    plans: list[RecapPlan] = field(default_factory=list)
    commits: list[RecapCommit] = field(default_factory=list)


_PREVIEW_LEN = 200


def _to_chunk(row) -> RecapChunk:
    return RecapChunk(
        chunk_id=row["chunk_id"],
        source_id=row["source_id"],
        text_preview=(row["text"] or "")[:_PREVIEW_LEN],
        category_labels=[c for c in (row["category_labels"] or "").split(",") if c],
        igio_axis=row["igio_axis"] or "",
        igio_confidence=float(row["igio_confidence"] or 0.0),
    )


def _chunks_for_axis(
    conn: DBAdapter,
    axis: str,
    *,
    issue_id: str,
    workspace_id: str | None,
    limit: int = 25,
) -> list[RecapChunk]:
    """Return chunks classified to `axis` that touch the issue.

    Issue-touch heuristic: source_id contains "issue-{id}" or "issues/{id}",
    or the chunk text mentions "#{id}" / "issue {id}".
    """
    issue_key = str(issue_id).strip().lstrip("#")
    if not issue_key:
        return []

    # Issue-touch heuristic packed as four parameterised LIKE clauses — every
    # value is bound, no concatenation. Two static SQL strings cover the
    # optional workspace filter without joining WHERE fragments.
    issue_patterns = (
        f"%issue-{issue_key}%",
        f"%issues/{issue_key}%",
        f"%#{issue_key}%",
        f"%issue {issue_key}%",
    )
    if workspace_id:
        rows = conn.execute(
            "SELECT chunk_id, source_id, text, category_labels, igio_axis, "
            "igio_confidence FROM chunks "
            "WHERE igio_axis = ? AND is_active = 1 AND workspace_id = ? "
            "AND (source_id LIKE ? OR source_id LIKE ? "
            "     OR text LIKE ? OR text LIKE ?) "
            "ORDER BY igio_confidence DESC, created_at DESC LIMIT ?",
            (axis, workspace_id, *issue_patterns, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT chunk_id, source_id, text, category_labels, igio_axis, "
            "igio_confidence FROM chunks "
            "WHERE igio_axis = ? AND is_active = 1 "
            "AND (source_id LIKE ? OR source_id LIKE ? "
            "     OR text LIKE ? OR text LIKE ?) "
            "ORDER BY igio_confidence DESC, created_at DESC LIMIT ?",
            (axis, *issue_patterns, limit),
        ).fetchall()
    return [_to_chunk(r) for r in rows]


_ISSUE_RE_TEMPLATE = r"(?:#|issue[\s_-]+)({})\b"


def _plans_mentioning_issue(
    plans_dir: Path,
    issue_id: str,
) -> list[RecapPlan]:
    """Find plan markdowns whose body mentions the issue."""
    if not plans_dir.exists():
        return []
    issue_key = str(issue_id).strip().lstrip("#")
    if not issue_key:
        return []
    pattern = re.compile(_ISSUE_RE_TEMPLATE.format(re.escape(issue_key)), re.I)
    out: list[RecapPlan] = []
    for path in sorted(plans_dir.glob("*.md")):
        try:
            body = path.read_text(errors="ignore")
        except OSError:
            continue
        if not pattern.search(body):
            continue
        title = next(
            (
                line.lstrip("# ").strip()
                for line in body.splitlines()
                if line.startswith("# ")
            ),
            path.stem,
        )
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        out.append(RecapPlan(path=path, title=title, mtime_iso=mtime))
    return out


def _commits_since(
    repo_root: Path,
    since_iso: str | None,
    *,
    grep: str | None = None,
    limit: int = 30,
) -> list[RecapCommit]:
    """Run `git log` against repo_root and return parsed commits."""
    if not (repo_root / ".git").exists():
        return []
    args = [
        "git", "log",
        f"--max-count={limit}",
        "--pretty=format:%H%x09%aI%x09%s",
    ]
    if since_iso:
        args.append(f"--since={since_iso}")
    if grep:
        args.extend(["--grep", grep, "-i"])
    try:
        proc = subprocess.run(  # nosec B603
            args,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    out: list[RecapCommit] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        sha, iso, subject = parts
        out.append(RecapCommit(sha=sha[:12], iso_date=iso, subject=subject.strip()))
    return out


def build_recap(
    issue_id: str,
    *,
    conn: DBAdapter,
    workspace_id: str | None = None,
    repo_root: Path | None = None,
    plans_dir: Path | None = None,
    chunk_limit: int = 25,
    commit_limit: int = 30,
) -> Recap:
    """Assemble a Recap for a single issue id from already-classified data."""
    issue_id = str(issue_id).strip().lstrip("#")
    plans_dir = plans_dir or PLANS_DIR_DEFAULT
    repo_root = repo_root or _detect_repo_root()

    recap = Recap(issue_id=issue_id, workspace_id=workspace_id or "")

    recap.issue_chunks = _chunks_for_axis(
        conn, "issue", issue_id=issue_id, workspace_id=workspace_id, limit=chunk_limit,
    )
    recap.goal_chunks = _chunks_for_axis(
        conn, "goal", issue_id=issue_id, workspace_id=workspace_id, limit=chunk_limit,
    )
    recap.intervention_chunks = _chunks_for_axis(
        conn, "intervention", issue_id=issue_id, workspace_id=workspace_id,
        limit=chunk_limit,
    )
    recap.outcome_chunks = _chunks_for_axis(
        conn, "outcome", issue_id=issue_id, workspace_id=workspace_id, limit=chunk_limit,
    )

    recap.plans = _plans_mentioning_issue(plans_dir, issue_id)

    earliest_plan = min((p.mtime_iso for p in recap.plans), default=None)
    recap.commits = _commits_since(
        repo_root,
        since_iso=earliest_plan,
        grep=f"#{issue_id}",
        limit=commit_limit,
    )
    return recap


def _detect_repo_root() -> Path:
    """Best-effort: walk up from $CWD until we hit a `.git` dir."""
    p = Path(os.getcwd()).resolve()
    for parent in (p, *p.parents):
        if (parent / ".git").exists():
            return parent
    return p


def discover_issue_ids(conn: DBAdapter, workspace_id: str | None = None) -> list[str]:
    """Return issue ids that appear in github_issue source_ids.

    Lets callers ask "which issues do I have data for" without scanning text.
    """
    # Two fixed SQL strings — workspace_id is the only optional filter and
    # rides on a placeholder. No string concatenation of caller input into
    # the query body; satisfies the opengrep raw-query audit.
    if workspace_id:
        rows = conn.execute(
            "SELECT DISTINCT source_id FROM sources "
            "WHERE source_type = 'github_issue' AND workspace_id = ?",
            (workspace_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT DISTINCT source_id FROM sources "
            "WHERE source_type = 'github_issue'",
            (),
        ).fetchall()
    out: set[str] = set()
    pattern = re.compile(r"issue[s]?[/_-](\d+)", re.I)
    for r in rows:
        sid = r["source_id"] if hasattr(r, "keys") else r[0]
        m = pattern.search(sid or "")
        if m:
            out.add(m.group(1))
    return sorted(out, key=lambda x: int(x) if x.isdigit() else 0)


__all__: Iterable[str] = (
    "Recap", "RecapChunk", "RecapPlan", "RecapCommit",
    "build_recap", "discover_issue_ids",
)
