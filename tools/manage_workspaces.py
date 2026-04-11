#!/usr/bin/env python3
"""Workspace management CLI for MayringCoder multi-tenancy.

Auth is now handled via Laravel Sanctum tokens (app.linn.games). This tool
manages workspace memory data only — no API keys.

Usage:
    .venv/bin/python tools/manage_workspaces.py list
    .venv/bin/python tools/manage_workspaces.py delete <workspace_id> [--purge]

Examples:
    # List all workspaces with memory data
    .venv/bin/python tools/manage_workspaces.py list

    # Delete all memory data for a workspace
    .venv/bin/python tools/manage_workspaces.py delete alice --purge
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root or tools/
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_list() -> None:
    from src.memory_store import init_memory_db
    conn = init_memory_db()
    rows = conn.execute(
        "SELECT workspace_id, COUNT(*) as chunk_count FROM chunks GROUP BY workspace_id"
    ).fetchall()
    if not rows:
        print("No workspace memory data found.")
        return
    print(f"{'Workspace':<30} {'Chunks':>8}")
    print("-" * 40)
    for row in sorted(rows, key=lambda r: r[0]):
        print(f"{row[0]:<30} {row[1]:>8}")


def cmd_delete(workspace_id: str, purge: bool) -> None:
    if not purge:
        print(f"Nothing to do for workspace '{workspace_id}' without --purge.")
        print("Auth tokens are managed in app.linn.games — revoke them there.")
        return

    from src.memory_store import init_memory_db
    conn = init_memory_db()
    chunk_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE workspace_id = ?", (workspace_id,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE chunks SET is_active = 0 WHERE workspace_id = ?", (workspace_id,)
    )
    conn.execute(
        "DELETE FROM sources WHERE workspace_id = ?", (workspace_id,)
    )
    conn.commit()
    print(f"Purged {chunk_count} chunk(s) for workspace '{workspace_id}'.")
    print("Note: Revoke the Sanctum token in app.linn.games to block future access.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MayringCoder workspace memory management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all workspaces with memory data")

    p_delete = sub.add_parser("delete", help="Delete workspace memory data")
    p_delete.add_argument("workspace_id")
    p_delete.add_argument("--purge", action="store_true", help="Actually delete all memory data")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list()
    elif args.command == "delete":
        cmd_delete(args.workspace_id, args.purge)


if __name__ == "__main__":
    main()
