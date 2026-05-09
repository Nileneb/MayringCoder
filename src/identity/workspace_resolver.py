"""Kanonische Workspace-Auflösung.

Diese Schicht ist die EINE Quelle der Wahrheit für workspace_id.
Davor:
  - JWT-Pfad: f"user-{sub}"
  - CLI-Pfad: args.workspace_id or "default"
  → gleicher User → 2 verschiedene Buckets je nach Eingangsweg.

Jetzt:
  resolve_workspace(input, default_user_id) → kanonischer Workspace-ID,
  validiert gegen die `workspaces`-Tabelle, ggf. via aliases-Lookup.

Auflösungs-Reihenfolge:
  1. input ist None → fallback auf default_user_id → user-{id}
  2. input matcht 'user-N' Pattern → ensure(id, kind=user, owner=N)
  3. input ist canonical workspace (existiert) → return as-is
  4. input ist alias → return aliased canonical
  5. else: raise UnknownWorkspaceError
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from src.memory.db_adapter import DBAdapter

USER_WORKSPACE_RE = re.compile(r"^user-(\d+)(?::([\w\-]+))?$")


class UnknownWorkspaceError(ValueError):
    """Raised when an alias/workspace cannot be resolved against the DB."""


class IdentityRequiredError(ValueError):
    """Raised when no workspace_id and no default_user_id was provided."""


def resolve_workspace(
    conn: DBAdapter,
    workspace_input: str | None,
    *,
    default_user_id: int | None = None,
    auto_create_user_workspace: bool = True,
) -> str:
    """Resolve a free-form workspace identifier to its canonical form.

    Args:
        conn: open memory.db adapter
        workspace_input: what the caller passed (CLI arg, JWT claim, etc.)
        default_user_id: app.linn.games User.id used if input is None
        auto_create_user_workspace: if True (default), 'user-N' inputs
            that don't yet exist in the workspaces table are upserted as
            kind=user. This keeps JWT-flows zero-friction: a fresh user's
            first request mints their workspace on the fly.

    Returns:
        canonical workspace_id string

    Raises:
        IdentityRequiredError: input is None and no default_user_id
        UnknownWorkspaceError: input is a string we can't map
    """
    # Tolerant gegen non-string Inputs (z.B. MagicMock in Tests). Production-
    # CLI/JWT-Pfade liefern immer str | None. Alles andere wird wie missing
    # behandelt — der hard-error kommt dann sauber über default_user_id.
    if not isinstance(workspace_input, str):
        workspace_input = None
    if workspace_input is None or workspace_input.strip() == "":
        if default_user_id is None:
            raise IdentityRequiredError(
                "workspace_id is None and no default_user_id provided. "
                "Set MAYRING_USER_ID env or run `mayring login`, or pass "
                "--workspace explicitly."
            )
        canonical = f"user-{default_user_id}"
        if auto_create_user_workspace:
            ensure_user_workspace(conn, default_user_id)
        return canonical

    candidate = workspace_input.strip()

    # 'user-N' Pattern → kanonisch, ensure existence.
    m = USER_WORKSPACE_RE.match(candidate)
    if m:
        user_id = int(m.group(1))
        sub_slug = m.group(2)
        if auto_create_user_workspace:
            ensure_user_workspace(conn, user_id)
            if sub_slug:
                ensure_project_workspace(conn, user_id, sub_slug)
            return candidate
        # Read-only-Modus: pattern matcht zwar, aber Workspace muss
        # bereits existieren, sonst hat der Caller eine ungültige ID.
        existing = conn.execute(
            "SELECT id FROM workspaces WHERE id = ?", (candidate,)
        ).fetchone()
        if existing:
            return candidate
        raise UnknownWorkspaceError(
            f"workspace_id={candidate!r} matches user-N pattern but does "
            f"not exist (auto_create disabled)."
        )

    # Lookup direct workspace
    row = conn.execute(
        "SELECT id FROM workspaces WHERE id = ?", (candidate,)
    ).fetchone()
    if row:
        return row[0]

    # Lookup alias
    row = conn.execute(
        "SELECT workspace_id FROM workspace_aliases WHERE alias = ?",
        (candidate,),
    ).fetchone()
    if row:
        return row[0]

    raise UnknownWorkspaceError(
        f"workspace_id={candidate!r} is neither a known workspace nor a "
        f"registered alias. Use `mayring workspace add` or pass user-N."
    )


def ensure_user_workspace(conn: DBAdapter, user_id: int) -> str:
    """Upsert kind=user workspace for app.linn.games User.id."""
    workspace_id = f"user-{user_id}"
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO workspaces (id, kind, owner_user_id, display_name,
                                   created_at, updated_at)
           VALUES (?, 'user', ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at""",
        (workspace_id, user_id, f"User {user_id}", now, now),
    )
    conn.commit()
    return workspace_id


def ensure_project_workspace(
    conn: DBAdapter, user_id: int, slug: str
) -> str:
    """Upsert kind=project sub-workspace under user-{id}."""
    parent = f"user-{user_id}"
    workspace_id = f"{parent}:{slug}"
    now = datetime.now(timezone.utc).isoformat()
    ensure_user_workspace(conn, user_id)
    conn.execute(
        """INSERT INTO workspaces (id, kind, parent_id, owner_user_id,
                                   display_name, created_at, updated_at)
           VALUES (?, 'project', ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at""",
        (workspace_id, parent, user_id, slug, now, now),
    )
    conn.commit()
    return workspace_id


def add_alias(conn: DBAdapter, alias: str, workspace_id: str) -> None:
    """Register a non-canonical name as alias for a canonical workspace.

    Useful for legacy-import: `add_alias(conn, "default", "user-2")`.
    """
    row = conn.execute(
        "SELECT id FROM workspaces WHERE id = ?", (workspace_id,)
    ).fetchone()
    if row is None:
        raise UnknownWorkspaceError(
            f"cannot alias unknown workspace {workspace_id!r}"
        )
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO workspace_aliases (alias, workspace_id, created_at)
           VALUES (?, ?, ?)
           ON CONFLICT(alias) DO UPDATE SET workspace_id = excluded.workspace_id""",
        (alias, workspace_id, now),
    )
    conn.commit()


def list_workspaces_for_user(conn: DBAdapter, user_id: int) -> list[dict]:
    """Return canonical + project workspaces owned by a user."""
    rows = conn.execute(
        """SELECT id, kind, parent_id, display_name FROM workspaces
           WHERE owner_user_id = ? ORDER BY kind, id""",
        (user_id,),
    ).fetchall()
    return [
        {"id": r[0], "kind": r[1], "parent_id": r[2], "display_name": r[3]}
        for r in rows
    ]
