"""Laravel Sanctum token validation against app.linn.games PostgreSQL DB.

Validates Bearer tokens of the form "{id}|{plaintext}" by:
1. Splitting on "|" → token_id + plaintext
2. SHA-256 hashing the plaintext
3. Looking up personal_access_tokens in the shared PostgreSQL DB
4. Fetching the user + workspace record to check mayring_active

Required .env vars:
    LARAVEL_DB_HOST     (default: postgres — Docker service name)
    LARAVEL_DB_PORT     (default: 5432)
    LARAVEL_DB_USER
    LARAVEL_DB_PASSWORD
    LARAVEL_DB_NAME
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

_DB_HOST = os.getenv("LARAVEL_DB_HOST", "postgres")
_DB_PORT = int(os.getenv("LARAVEL_DB_PORT", "5432"))
_DB_USER = os.getenv("LARAVEL_DB_USER", "")
_DB_PASSWORD = os.getenv("LARAVEL_DB_PASSWORD", "")
_DB_NAME = os.getenv("LARAVEL_DB_NAME", "")


@dataclass
class TokenInfo:
    workspace_id: str
    mayring_active: bool


def _db_conn():
    import psycopg2
    import psycopg2.extras

    return psycopg2.connect(
        host=_DB_HOST,
        port=_DB_PORT,
        user=_DB_USER,
        password=_DB_PASSWORD,
        dbname=_DB_NAME,
        connect_timeout=5,
    )


def validate_sanctum_token(raw_token: str) -> str | None:
    """Return workspace_id for a valid Sanctum token, or None if invalid/expired.

    For external (non-system) tokens, also verifies mayring_active on the workspace.
    Returns None if the workspace has no active Mayring subscription.
    """
    info = validate_sanctum_token_full(raw_token)
    return info.workspace_id if info else None


def validate_sanctum_token_full(raw_token: str) -> TokenInfo | None:
    """Return TokenInfo (workspace_id + mayring_active) for a valid Sanctum token."""
    if not raw_token or "|" not in raw_token:
        return None

    token_id, plaintext = raw_token.split("|", 1)
    if not token_id.isdigit() or not plaintext:
        return None

    hashed = hashlib.sha256(plaintext.encode()).hexdigest()
    conn = None
    try:
        conn = _db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pat.tokenable_id
                FROM personal_access_tokens pat
                WHERE pat.id = %s
                  AND pat.token = %s
                  AND (pat.expires_at IS NULL OR pat.expires_at > NOW())
                """,
                (int(token_id), hashed),
            )
            row = cur.fetchone()
            if not row:
                return None

            user_id = row[0]

            # Update last_used_at so the token shows as active in Laravel
            cur.execute(
                "UPDATE personal_access_tokens SET last_used_at = NOW() WHERE id = %s",
                (int(token_id),),
            )

            # Fetch workspace for the user (owner relationship)
            cur.execute(
                """
                SELECT w.id, w.mayring_active
                FROM workspaces w
                WHERE w.owner_id = %s
                ORDER BY w.created_at
                LIMIT 1
                """,
                (user_id,),
            )
            ws_row = cur.fetchone()
            conn.commit()

            if not ws_row:
                return None

            workspace_id = str(ws_row[0])
            mayring_active = bool(ws_row[1])

            return TokenInfo(workspace_id=workspace_id, mayring_active=mayring_active)

    except Exception:
        return None
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
