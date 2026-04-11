"""Laravel Sanctum token validation against app.linn.games MySQL DB.

Validates Bearer tokens of the form "{id}|{plaintext}" by:
1. Splitting on "|" → token_id + plaintext
2. SHA-256 hashing the plaintext
3. Looking up personal_access_tokens in the shared MySQL DB
4. Fetching the user record to derive a workspace_id

Required .env vars:
    LARAVEL_DB_HOST     (default: mysql — Docker service name)
    LARAVEL_DB_PORT     (default: 3306)
    LARAVEL_DB_USER
    LARAVEL_DB_PASSWORD
    LARAVEL_DB_NAME

workspace_id = email prefix before "@", lowercased, max 40 chars.
"""

from __future__ import annotations

import hashlib
import os

_DB_HOST = os.getenv("LARAVEL_DB_HOST", "mysql")
_DB_PORT = int(os.getenv("LARAVEL_DB_PORT", "3306"))
_DB_USER = os.getenv("LARAVEL_DB_USER", "")
_DB_PASSWORD = os.getenv("LARAVEL_DB_PASSWORD", "")
_DB_NAME = os.getenv("LARAVEL_DB_NAME", "")


def _db_conn():
    import pymysql
    import pymysql.cursors

    return pymysql.connect(
        host=_DB_HOST,
        port=_DB_PORT,
        user=_DB_USER,
        password=_DB_PASSWORD,
        database=_DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=5,
    )


def validate_sanctum_token(raw_token: str) -> str | None:
    """Return workspace_id for a valid Sanctum token, or None if invalid/expired.

    Token format: "{id}|{plaintext}"
    DB lookup:    personal_access_tokens WHERE id=? AND token=sha256(plaintext)
                  AND (expires_at IS NULL OR expires_at > NOW())
    workspace_id: users.email prefix before "@", lowercase, max 40 chars.
    """
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
                SELECT tokenable_id
                FROM personal_access_tokens
                WHERE id = %s
                  AND token = %s
                  AND (expires_at IS NULL OR expires_at > NOW())
                """,
                (int(token_id), hashed),
            )
            row = cur.fetchone()
            if not row:
                return None

            user_id = row["tokenable_id"]
            cur.execute(
                "SELECT email FROM users WHERE id = %s",
                (user_id,),
            )
            user = cur.fetchone()
            if not user or not user.get("email"):
                return None

            # Update last_used_at so the token shows as active in Laravel
            cur.execute(
                "UPDATE personal_access_tokens SET last_used_at = NOW() WHERE id = %s",
                (int(token_id),),
            )
            conn.commit()

        workspace_id = user["email"].split("@")[0].lower()[:40]
        # Replace any chars not safe for filesystem/identifiers
        workspace_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in workspace_id)
        return workspace_id or None

    except Exception:
        return None
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
