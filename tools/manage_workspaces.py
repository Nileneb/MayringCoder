#!/usr/bin/env python3
"""Workspace + API-Key management CLI for MayringCoder multi-tenancy.

Usage:
    .venv/bin/python tools/manage_workspaces.py create <workspace_id>
    .venv/bin/python tools/manage_workspaces.py list
    .venv/bin/python tools/manage_workspaces.py rotate <workspace_id>
    .venv/bin/python tools/manage_workspaces.py delete <workspace_id> [--purge]

Examples:
    # Create workspace + get API key (shown once)
    .venv/bin/python tools/manage_workspaces.py create nileneb

    # List all workspaces and key counts
    .venv/bin/python tools/manage_workspaces.py list

    # Rotate API key (old key immediately invalid)
    .venv/bin/python tools/manage_workspaces.py rotate nileneb

    # Delete workspace API keys (--purge also removes all memory data)
    .venv/bin/python tools/manage_workspaces.py delete alice --purge
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root or tools/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api_server import create_api_key, revoke_api_keys, _load_keys


def cmd_create(workspace_id: str) -> None:
    if not workspace_id.replace("-", "").replace("_", "").isalnum():
        print(f"ERROR: workspace_id must be alphanumeric (hyphens/underscores allowed): {workspace_id!r}")
        sys.exit(1)
    raw_key = create_api_key(workspace_id)
    print(f"Workspace '{workspace_id}' created.")
    print(f"API Key (shown once — store securely):\n  {raw_key}")


def cmd_list() -> None:
    keys = _load_keys()
    if not keys:
        print("No workspaces configured.")
        return
    # Group by workspace_id
    workspaces: dict[str, list] = {}
    for hashed, meta in keys.items():
        ws = meta["workspace_id"]
        workspaces.setdefault(ws, []).append(meta["created_at"])
    print(f"{'Workspace':<30} {'Keys':>5}  {'Latest Key Created'}")
    print("-" * 65)
    for ws, dates in sorted(workspaces.items()):
        latest = max(dates)
        print(f"{ws:<30} {len(dates):>5}  {latest}")


def cmd_rotate(workspace_id: str) -> None:
    removed = revoke_api_keys(workspace_id)
    if removed == 0:
        print(f"WARNING: No existing keys found for workspace '{workspace_id}'.")
    else:
        print(f"Revoked {removed} existing key(s) for '{workspace_id}'.")
    raw_key = create_api_key(workspace_id)
    print(f"New API Key (shown once — store securely):\n  {raw_key}")


def cmd_delete(workspace_id: str, purge: bool) -> None:
    removed = revoke_api_keys(workspace_id)
    print(f"Revoked {removed} API key(s) for workspace '{workspace_id}'.")

    if purge:
        from src.memory_store import init_memory_db
        conn = init_memory_db()
        # Deactivate all chunks for this workspace
        conn.execute(
            "UPDATE chunks SET is_active = 0 WHERE workspace_id = ?", (workspace_id,)
        )
        conn.execute(
            "DELETE FROM sources WHERE workspace_id = ?", (workspace_id,)
        )
        conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE workspace_id = ?", (workspace_id,)
        ).fetchone()[0]
        print(f"Purged all memory data for workspace '{workspace_id}'. ({count} chunks deactivated)")
    else:
        print(f"Workspace '{workspace_id}' removed. Memory data retained (use --purge to delete).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MayringCoder workspace management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create workspace + generate API key")
    p_create.add_argument("workspace_id")

    sub.add_parser("list", help="List all workspaces")

    p_rotate = sub.add_parser("rotate", help="Rotate API key for workspace")
    p_rotate.add_argument("workspace_id")

    p_delete = sub.add_parser("delete", help="Delete workspace and optionally purge memory")
    p_delete.add_argument("workspace_id")
    p_delete.add_argument("--purge", action="store_true", help="Also delete all memory data")

    args = parser.parse_args()

    if args.command == "create":
        cmd_create(args.workspace_id)
    elif args.command == "list":
        cmd_list()
    elif args.command == "rotate":
        cmd_rotate(args.workspace_id)
    elif args.command == "delete":
        cmd_delete(args.workspace_id, args.purge)


if __name__ == "__main__":
    main()
