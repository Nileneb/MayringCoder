#!/usr/bin/env python3
"""Generate JWT tokens for MCP Memory Server authentication."""
import argparse
import re
import sys
import time

try:
    import jwt
except ImportError:
    print("Error: PyJWT not installed. Run: pip install PyJWT", file=sys.stderr)
    sys.exit(1)


def _parse_expiry(s: str) -> int:
    """Parse expiry string like '30d', '12h', '60m', '3600s' into seconds."""
    m = re.match(r"^(\d+)([dhms]?)$", s.strip())
    if not m:
        raise ValueError(f"Invalid expiry: {s!r}. Use formats like '30d', '12h', '60m', '3600s'.")
    value, unit = int(m.group(1)), m.group(2) or "s"
    return value * {"d": 86400, "h": 3600, "m": 60, "s": 1}[unit]


def generate_token(workspace_id: str, scope: str, secret: str, expiry_seconds: int) -> str:
    """Create a signed HS256 JWT token for MCP Memory Server."""
    now = int(time.time())
    payload = {
        "workspace_id": workspace_id,
        "scope": scope,
        "iat": now,
        "exp": now + expiry_seconds,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate JWT tokens for MCP Memory Server authentication.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --workspace myrepo --secret "$MCP_AUTH_SECRET"
  %(prog)s --workspace ws_prod --scope admin --secret "$MCP_AUTH_SECRET" --expiry 7d
  %(prog)s --workspace dev --secret "$MCP_AUTH_SECRET" --expiry 1h
""",
    )
    p.add_argument("--workspace", required=True, help="workspace_id embedded in the token")
    p.add_argument("--scope", default="repo", help="scope label (default: repo)")
    p.add_argument("--secret", required=True, help="HMAC secret (must match MCP_AUTH_SECRET)")
    p.add_argument(
        "--expiry",
        default="30d",
        help="token expiry: N[d|h|m|s], e.g. '30d', '12h', '3600s' (default: 30d)",
    )
    args = p.parse_args()

    try:
        expiry_seconds = _parse_expiry(args.expiry)
    except ValueError as exc:
        p.error(str(exc))

    token = generate_token(args.workspace, args.scope, args.secret, expiry_seconds)
    print(token)


if __name__ == "__main__":
    main()
