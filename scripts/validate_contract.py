#!/usr/bin/env python3
"""Cross-Repo-Vertrags-Validator (V2 Stufe 4).

Liest docs/cross-repo-contracts/v2.yaml und prüft, dass MayringCoder die
darin definierten JWT-Claims tatsächlich akzeptiert. Failt CI bei Drift.

Beide Seiten (MayringCoder Python + app.linn.games PHP) führen
strukturell ähnliche Checks aus — bei Drift schlägt der CI auf der
Seite an wo das Schema nicht mehr matcht.

Usage:
  python scripts/validate_contract.py
  exit 0 = OK, exit 1 = Drift
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "docs" / "cross-repo-contracts" / "v2.yaml"
JWT_AUTH_PATH = ROOT / "src" / "api" / "jwt_auth.py"


def main() -> int:
    if not CONTRACT_PATH.exists():
        print(f"FAIL: contract not found at {CONTRACT_PATH}", file=sys.stderr)
        return 1
    if not JWT_AUTH_PATH.exists():
        print(f"FAIL: src/api/jwt_auth.py not found", file=sys.stderr)
        return 1

    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    required_claims = set(contract["jwt"]["claims"].keys())

    # Parse src/api/jwt_auth.py via AST und extrahiere TokenInfo-felder
    tree = ast.parse(JWT_AUTH_PATH.read_text(encoding="utf-8"))
    token_info_fields: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TokenInfo":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    token_info_fields.add(item.target.id)
                elif isinstance(item, ast.Assign):
                    for tgt in item.targets:
                        if isinstance(tgt, ast.Name):
                            token_info_fields.add(tgt.id)

    if not token_info_fields:
        print("FAIL: TokenInfo class not parsed", file=sys.stderr)
        return 1

    # Map contract names to TokenInfo field names (some renaming allowed)
    rename_map = {
        "sub": "sub",
        "email": None,  # consumed at decode-time for workspace-slug, not stored
        "workspace_id": "workspace_id",
        "memberships": "memberships",
        "scope": "scopes",  # claim is "scope", field is "scopes" (plural in TokenInfo)
        "version": None,  # claim "version" not stored as TokenInfo field
    }

    # Files where the JWT decoder reads claims that aren't TokenInfo fields
    src_text = JWT_AUTH_PATH.read_text(encoding="utf-8")

    missing: list[str] = []
    for claim in required_claims:
        target = rename_map.get(claim, claim)
        if target is None:
            # Claim is consumed at decode-time — verify the source reads it
            needle = f'"{claim}"'
            alt_needle = f"'{claim}'"
            if needle not in src_text and alt_needle not in src_text:
                missing.append(f"{claim} (not read in jwt_auth.py)")
            continue
        if target not in token_info_fields:
            missing.append(f"{claim} (TokenInfo.{target})")

    if missing:
        print(
            f"FAIL: contract claims missing in jwt_auth.py: {missing}",
            file=sys.stderr,
        )
        print(f"  TokenInfo fields: {sorted(token_info_fields)}", file=sys.stderr)
        print(f"  Contract claims: {sorted(required_claims)}", file=sys.stderr)
        return 1

    print(f"OK: contract claims {sorted(required_claims)} all consumed by jwt_auth.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
