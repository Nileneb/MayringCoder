"""Kanonische Workspace-Auflösung.

Diese Schicht ist die EINE Quelle der Wahrheit für workspace_id.

Workspace-IDs sind seit 2026-05-09 KEINE 'user-N'-Krücken mehr,
sondern human-readable Slugs aus dem Email-Localpart (z.B. 'bene'
für bene@linn.games). Vorteile:
  - Logs lesbar: 'workspace=bene' statt 'workspace=user-2'.
  - Multi-Tenant-vorbereitet: 'team-acme', 'bene', 'system' im selben
    Bucket-Namespace ohne Prefix-Konflikte.
  - Deterministisch: gleiche Email → gleicher Slug, auch wenn 3 Apps
    parallel JWT-Tokens minten.

Auflösungs-Reihenfolge:
  1. input ist None + default_user_id+default_email → ensure_user_workspace
  2. 'system' → kanonischer system-bucket
  3. input existiert in workspaces → return as-is
  4. input ist alias → return aliased canonical
  5. legacy 'user-N' Pattern (für alte JWT-Tokens vor dem Refactor)
     → ensure(slug=user-N, owner=N) als Fallback bis alle Clients
     emails senden
  6. else: raise UnknownWorkspaceError
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


def email_to_slug(email: str) -> str:
    """Email → human-readable slug für Workspace-IDs.

    'bene@linn.games' → 'bene'
    'foo.bar@example.com' → 'foo-bar'
    'admin+tag@firma.de' → 'admin-tag'

    Slugs sind ASCII-only, lowercase, ohne special chars außer '-'.
    Idempotent: slug(slug(x)) == slug(x).
    """
    local = (email or "").split("@", 1)[0].strip().lower()
    if not local:
        return ""
    out = []
    prev_dash = False
    for ch in local:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-")


def resolve_workspace(
    conn: DBAdapter,
    workspace_input: str | None,
    *,
    default_user_id: int | None = None,
    default_email: str | None = None,
    default_display_name: str | None = None,
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
        # Email → Slug. Wenn keine email da ist, fallback auf user-N
        # (legacy, sollte nur für mock-tests oder alte JWT-Tokens
        # passieren).
        canonical = email_to_slug(default_email or "") or f"user-{default_user_id}"
        if auto_create_user_workspace:
            ensure_user_workspace(
                conn, default_user_id,
                slug=canonical, email=default_email,
                display_name=default_display_name,
            )
        return canonical

    candidate = workspace_input.strip()

    # 'system' ist ein reservierter Server-Side-Bucket: post-deploy-ingest,
    # Service-Token-Aufrufe (api/auth.py:37), conversation_watcher,
    # ambient-Snapshots. Wird beim DB-init via ensure_system_workspace
    # angelegt; Resolver akzeptiert ihn auch wenn die Row noch fehlt.
    if candidate == "system":
        if auto_create_user_workspace:
            ensure_system_workspace(conn)
        return candidate

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


def ensure_system_workspace(conn: DBAdapter) -> str:
    """Upsert kind=system workspace (Service-Token, Cron-Jobs, ambient)."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO workspaces (id, kind, owner_user_id, display_name,
                                   created_at, updated_at)
           VALUES ('system', 'system', NULL, 'System (Service-Token / Cron)',
                   ?, ?)
           ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at""",
        (now, now),
    )
    conn.commit()
    return "system"


def ensure_user_workspace(
    conn: DBAdapter,
    user_id: int,
    *,
    slug: str | None = None,
    email: str | None = None,
    display_name: str | None = None,
) -> str:
    """Upsert kind=user workspace.

    Bevorzugt slug aus email — fallback user-N nur wenn weder slug
    noch email mitgegeben werden (legacy / mock-tests).
    """
    workspace_id = slug or email_to_slug(email or "") or f"user-{user_id}"
    now = datetime.now(timezone.utc).isoformat()
    pretty_name = display_name or email or f"User {user_id}"
    conn.execute(
        """INSERT INTO workspaces (id, kind, owner_user_id, email,
                                   display_name, created_at, updated_at)
           VALUES (?, 'user', ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               email = COALESCE(excluded.email, workspaces.email),
               display_name = COALESCE(excluded.display_name, workspaces.display_name),
               updated_at = excluded.updated_at""",
        (workspace_id, user_id, email, pretty_name, now, now),
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
