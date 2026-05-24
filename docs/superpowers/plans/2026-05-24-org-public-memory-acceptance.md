# V2 Org/Public Memory — Acceptance & Gap-Closing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the already-built V2 org/public-memory model is correct end-to-end on production by adding an admin-only act-as test harness plus the 9 missing cross-tenant acceptance smoke checks.

**Architecture:** The V2 visibility model (private/user/org/public via JWT `memberships[]` → `TokenInfo.org_ids` → `_scope_filter`) is already implemented and unit-tested (19 green in `tests/test_v2_workspaces.py`). The gap is end-to-end production verification, which needs a *second* caller identity. We add a strictly admin/service-token-gated "act-as" header override in the single auth dependency (`get_token_info`) that builds a synthetic **non-privileged** `TokenInfo` from `X-Act-As-Sub` / `X-Act-As-Orgs` / `X-Act-As-Workspace`. The production smoke then simulates User A, User B, org-member and non-member to assert isolation. No Laravel changes, no seeded prod accounts.

**Tech Stack:** Python 3.13, FastAPI (`Header` deps), PyJWT (RS256), pytest, `tools/smoke_test_production.py` (urllib-based prod smoke), SQLite + Chroma memory store.

**Why this scope (grounding 2026-05-24):** Iter 1–4 of `docs/v2-workspaces-spec.md` are already built (Laravel emits `memberships[]` + `workspaces.type` + reissue; MayringCoder `jwt_auth.py`/`_scope_filter`/`org_ids`; L3/L6/L7/L8 fixed; `/api/workspaces*` CRUD + Livewire UI; `POST /sources/{id}/share`). The stale `hook.jwt` is bridged transparently by the alias-aware `resolve_workspace_from_token` (`src/api/auth.py:62-70`), so a token reissue is **optional** (only org-via-hook needs `memberships[]`). **Out of scope (YAGNI):** `/memory/orgs*` proxies (Laravel owns org CRUD), `GET /api/workspaces` list (no consumer), token reissue (alias bridges it).

---

## File Structure

- `src/api/auth.py` — **modify**: add act-as override to `get_token_info`. One new module-level helper `_maybe_act_as`. This is the only production-code change; it is test-harness infra, gated off by default.
- `tests/test_act_as.py` — **create**: unit tests for the act-as gate (privileged→override, non-privileged→ignored, flag-off→ignored, synthetic identity is non-admin).
- `tools/smoke_test_production.py` — **modify**: extend `_http` with `extra_headers`; add an `_act_as` header helper; add 9 check functions; register them in `ALL_CHECKS`.
- `docs/v2-workspaces-spec.md` — already updated (status Approved); no change here.

---

## Task 1: Admin/service-only act-as identity override

**Files:**
- Modify: `src/api/auth.py`
- Test: `tests/test_act_as.py` (create)

**Context:** `get_token_info` (`src/api/auth.py:20`) is the single FastAPI dependency that turns a Bearer token into a `TokenInfo`. The MCP service token yields `TokenInfo(workspace_id="system", scopes=("*",))`; an admin user JWT has `scopes=("mcp:memory","admin")`. `TokenInfo.is_admin` is `"admin" in scopes` — note the service token's `"*"` is NOT `is_admin`, so the gate must accept either. `TokenInfo` (frozen dataclass, `src/api/jwt_auth.py:38`) has `workspace_id, scopes, sub, memberships: tuple[Membership,...], org_id, active_workspace_id`; `Membership` is `NamedTuple(id, type, role, name=None)`; `org_ids` is derived from `memberships` where `type=="organization"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_act_as.py`:

```python
"""Admin/service-only act-as identity override (V2 org-memory acceptance harness)."""
import asyncio
import os
from unittest.mock import patch

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from src.api.auth import get_token_info
from src.api.jwt_auth import TokenInfo


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _creds(tok="svc"):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=tok)


@pytest.fixture
def service_token(monkeypatch):
    # Make the bearer 'svc' validate as the privileged service token.
    monkeypatch.setattr("src.api.auth._SERVICE_TOKEN", "svc")
    monkeypatch.setenv("MAYRING_ALLOW_ACT_AS", "1")
    yield


def test_act_as_builds_synthetic_nonadmin_identity(service_token):
    info = _run(get_token_info(
        creds=_creds("svc"),
        x_act_as_sub="42",
        x_act_as_orgs="org-a,org-b",
        x_act_as_workspace="ws-alice",
    ))
    assert info.sub == "42"
    assert info.workspace_id == "ws-alice"
    assert set(info.org_ids) == {"org-a", "org-b"}
    # CRITICAL: synthetic identity must NOT be privileged, else admin bypasses
    # _scope_filter and every isolation test trivially "passes".
    assert info.is_admin is False
    assert "*" not in info.scopes


def test_act_as_ignored_when_flag_off(service_token, monkeypatch):
    monkeypatch.delenv("MAYRING_ALLOW_ACT_AS", raising=False)
    info = _run(get_token_info(creds=_creds("svc"), x_act_as_sub="42",
                               x_act_as_orgs="org-a", x_act_as_workspace="ws-alice"))
    # Falls back to the real (privileged) service identity.
    assert info.sub != "42"
    assert "*" in info.scopes


def test_act_as_ignored_for_non_privileged_token(monkeypatch):
    monkeypatch.setattr("src.api.auth._SERVICE_TOKEN", "svc")
    monkeypatch.setenv("MAYRING_ALLOW_ACT_AS", "1")
    fake = TokenInfo(workspace_id="ws-bob", scopes=("mcp:memory",), sub="7")
    with patch("src.api.auth.validate_jwt_token", return_value=fake):
        info = _run(get_token_info(creds=_creds("user-jwt"), x_act_as_sub="42",
                                   x_act_as_orgs="org-a", x_act_as_workspace="ws-alice"))
    # Non-privileged caller cannot impersonate — headers ignored.
    assert info.sub == "7"
    assert info.workspace_id == "ws-bob"


def test_no_act_as_headers_is_passthrough(service_token):
    info = _run(get_token_info(creds=_creds("svc")))
    assert "*" in info.scopes  # unchanged service identity
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.:core python3 -m pytest tests/test_act_as.py -q`
Expected: FAIL — `get_token_info() got an unexpected keyword argument 'x_act_as_sub'`.

- [ ] **Step 3: Implement the act-as override in `src/api/auth.py`**

Add the import for `Membership` and the helper, and extend `get_token_info`. Replace the existing `get_token_info` definition (lines 20-47) and the import on line 10:

```python
from src.api.jwt_auth import Membership, TokenInfo, validate_jwt_token
```

```python
def _is_privileged(info: TokenInfo) -> bool:
    """Service token (scope '*') or admin JWT — the only callers allowed to act-as."""
    return info.is_admin or "*" in info.scopes


def _maybe_act_as(
    info: TokenInfo,
    sub: str | None,
    orgs: str | None,
    workspace: str | None,
) -> TokenInfo:
    """Admin/service-only test harness: simulate another caller so the prod
    smoke can prove cross-tenant isolation.

    WHY(v2-org-acceptance): the 9 acceptance smokes need a SECOND identity
    (User B / org-member / non-member); the service token only carries one.
    STRICTLY gated: only a privileged token, only when MAYRING_ALLOW_ACT_AS is
    set. The synthetic identity DROPS privilege (scope=('mcp:memory',)) — else
    an admin/service caller bypasses _scope_filter and every isolation test
    trivially passes. Non-privileged callers' act-as headers are ignored (never
    escalate).
    """
    if not (sub or orgs or workspace):
        return info
    if os.getenv("MAYRING_ALLOW_ACT_AS", "").lower() not in ("1", "true", "yes"):
        return info
    if not _is_privileged(info):
        return info
    org_tuple = tuple(o.strip() for o in (orgs or "").split(",") if o.strip())
    memberships = tuple(
        Membership(id=o, type="organization", role="viewer") for o in org_tuple
    )
    ws = workspace or info.workspace_id
    return TokenInfo(
        workspace_id=ws,
        scopes=("mcp:memory",),
        sub=sub or info.sub,
        memberships=memberships,
        active_workspace_id=ws,
    )


async def get_token_info(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    x_act_as_sub: str | None = Header(default=None, alias="X-Act-As-Sub"),
    x_act_as_orgs: str | None = Header(default=None, alias="X-Act-As-Orgs"),
    x_act_as_workspace: str | None = Header(default=None, alias="X-Act-As-Workspace"),
) -> TokenInfo:
    """Validate Bearer token — accepts RS256 JWT (users) or MCP_SERVICE_TOKEN.

    Service-Token: scope='*', workspace_id='system'. X-Workspace-Id / body
    override per get_workspace(). X-Act-As-* lets a privileged caller simulate
    another identity (admin-only test harness, see _maybe_act_as).
    """
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )
    token = creds.credentials
    if _SERVICE_TOKEN and hmac.compare_digest(
        token.encode() if isinstance(token, str) else token,
        _SERVICE_TOKEN.encode() if isinstance(_SERVICE_TOKEN, str) else _SERVICE_TOKEN,
    ):
        info: TokenInfo = TokenInfo(workspace_id="system", scopes=("*",))
    else:
        validated = validate_jwt_token(token)
        if not validated:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )
        info = validated
    return _maybe_act_as(info, x_act_as_sub, x_act_as_orgs, x_act_as_workspace)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=.:core python3 -m pytest tests/test_act_as.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the auth + workspace regression suites**

Run: `PYTHONPATH=.:core python3 -m pytest tests/test_v2_workspaces.py tests/ -k "auth or token or workspace" -q`
Expected: all pass (no regression; the new Header params are optional).

- [ ] **Step 6: Commit**

```bash
git add src/api/auth.py tests/test_act_as.py
git commit -m "feat(auth): admin/service-only act-as identity override (X-Act-As-*)"
```

---

## Task 2: Smoke harness — `extra_headers` + `_act_as` helper

**Files:**
- Modify: `tools/smoke_test_production.py` (`_http` at line 174; add helper near it)

**Context:** `_http(method, url, token, body=None, timeout=10.0, workspace_id=None)` builds headers and sets `X-Workspace-Id` when `workspace_id` is given (lines 201-206). The acceptance checks need to attach `X-Act-As-*` headers, so `_http` needs an `extra_headers` param. The act-as checks set the workspace via `X-Act-As-Workspace` (not `X-Workspace-Id`) so the synthetic identity owns it.

- [ ] **Step 1: Add `extra_headers` to `_http`**

In `tools/smoke_test_production.py`, change the `_http` signature and header assembly:

```python
def _http(method: str, url: str, token: str, body: dict | None = None,
          timeout: float = 10.0,
          workspace_id: str | None = None,
          extra_headers: dict | None = None) -> tuple[int, dict | None, float]:
```

In the header-building block (around line 201-206), after `headers["X-Workspace-Id"] = workspace_id`, add:

```python
        if extra_headers:
            headers.update(extra_headers)
```

Also thread `extra_headers` through `_http_await_source` (`tools/smoke_test_production.py:1046`, used by Tasks 4 & 9). Change its signature and both internal `_http` calls:

```python
def _http_await_source(method: str, url: str, token: str, *, body=None,
                       timeout: float = 15.0, tries: int = 6, delay: float = 0.5,
                       extra_headers: dict | None = None):
    """Like _http, but retry while the response is 404 (just-PUT source not yet
    visible across workers). Any non-404 (200/403/...) returns immediately."""
    code, body_r, hdrs = _http(method, url, token, body=body, timeout=timeout, extra_headers=extra_headers)
    for _ in range(tries - 1):
        if code != 404:
            break
        time.sleep(delay)
        code, body_r, hdrs = _http(method, url, token, body=body, timeout=timeout, extra_headers=extra_headers)
    return code, body_r, hdrs
```

(No `expect_status` param is needed — the existing `if code != 404: break` already returns a 403 on the first try without retrying-until-200.)

- [ ] **Step 2: Add the `_act_as` helper** (place directly after `_http`)

```python
def _act_as(sub: str, *, orgs: tuple[str, ...] = (), workspace: str | None = None) -> dict:
    """Build X-Act-As-* headers so a privileged smoke token simulates another
    caller. Requires MAYRING_ALLOW_ACT_AS=1 on the server (set in the smoke
    job env). workspace defaults to the caller's home if omitted."""
    h = {"X-Act-As-Sub": sub}
    if orgs:
        h["X-Act-As-Orgs"] = ",".join(orgs)
    if workspace:
        h["X-Act-As-Workspace"] = workspace
    return h
```

- [ ] **Step 3: Verify import-clean**

Run: `PYTHONPATH=.:core python3 -c "import tools.smoke_test_production as s; print(s._act_as('1', orgs=('o',), workspace='w'))"`
Expected: `{'X-Act-As-Sub': '1', 'X-Act-As-Orgs': 'o', 'X-Act-As-Workspace': 'w'}`

- [ ] **Step 4: Commit**

```bash
git add tools/smoke_test_production.py
git commit -m "test(smoke): _http extra_headers + _act_as helper for cross-tenant checks"
```

---

## Tasks 3–11: the 9 acceptance smoke checks

> **Shared pattern:** each check ingests as identity A (via `_act_as`), then reads/asserts as identity B (or the L8/stats variant). All ingests use `categorize: False` for speed and a unique `suffix = int(time.time())`. The server must run with `MAYRING_ALLOW_ACT_AS=1` (Task 12 wires the env). Each task: add the function, register it in `ALL_CHECKS` (list at `tools/smoke_test_production.py:2017`, entries are `("name", fn)`), run it live against `https://mcp.linn.games` with the service token, commit.

**Files (all of Tasks 3–11):**
- Modify: `tools/smoke_test_production.py` (add function + `ALL_CHECKS` entry)

Live-run command template (used in each task's verify step):
```bash
TOK=$(ssh nileneb@u-server "grep '^MCP_SERVICE_TOKEN=' ~/app.linn.games/.env | cut -d= -f2-")
PYTHONPATH=.:core python3 -c "import tools.smoke_test_production as s; print(s.<fn>('https://mcp.linn.games', '$TOK'))"
```
Expected each: `CheckResult(..., passed=True, ...)`.

### Task 3: `check_private_isolation`

- [ ] **Step 1: Add the function**

```python
def check_private_isolation(api: str, token: str) -> CheckResult:
    """User A ingests a PRIVATE source in workspace WA; User B (different
    workspace) must NOT see it. Proves _scope_filter blocks cross-workspace
    private reads (the core leak guarantee)."""
    suffix = int(time.time())
    wa, wb = f"va-{suffix}", f"vb-{suffix}"
    sid = f"smoke:priv-iso:{suffix}"
    code1, body1, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-iso",
              "path": "p", "content": f"PRIV-ISO {suffix}", "categorize": False},
        extra_headers=_act_as("A", workspace=wa), timeout=15.0)
    if code1 != 200:
        return CheckResult("private_isolation", False, f"ingest A failed http={code1}: {body1}")
    code2, body2, _ = _http("POST", f"{api}/memory/search", token,
        body={"query": f"PRIV-ISO {suffix}", "top_k": 5, "include_text": True, "llm_prefilter": False},
        extra_headers=_act_as("B", workspace=wb), timeout=12.0)
    seen = {r["source_id"] for r in (body2 or {}).get("results", [])}
    leaked = sid in seen
    return CheckResult("private_isolation", not leaked,
        f"B_sees_A_private={leaked} (must be False)  results={len(seen)}  marker={suffix}")
```

- [ ] **Step 2: Register** — add `("private_isolation", check_private_isolation),` to `ALL_CHECKS`.
- [ ] **Step 3: Run live** (template above). Expected: `passed=True`, `B_sees_A_private=False`.
- [ ] **Step 4: Commit** — `git commit -am "test(smoke): private_isolation cross-workspace check"`

### Task 4: `check_public_visibility`

- [ ] **Step 1: Add the function**

```python
def check_public_visibility(api: str, token: str) -> CheckResult:
    """A ingests then shares PUBLIC in WA; B in a different workspace MUST see
    it. Proves public is globally readable to any valid caller."""
    suffix = int(time.time())
    wa, wb = f"pa-{suffix}", f"pb-{suffix}"
    sid = f"smoke:pub-vis:{suffix}"
    code1, _, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-pub",
              "path": "p", "content": f"PUB-VIS {suffix}", "categorize": False},
        extra_headers=_act_as("A", workspace=wa), timeout=15.0)
    if code1 != 200:
        return CheckResult("public_visibility", False, f"ingest A failed http={code1}")
    code2, body2, _ = _http_await_source("POST",
        f"{api}/sources/{urllib.parse.quote(sid, safe='')}/share", token, body={},
        extra_headers=_act_as("A", workspace=wa))
    if code2 != 200 or (body2 or {}).get("visibility") != "public":
        return CheckResult("public_visibility", False, f"share failed http={code2}: {body2}")
    code3, body3, _ = _http("POST", f"{api}/memory/search", token,
        body={"query": f"PUB-VIS {suffix}", "top_k": 5, "include_text": True, "llm_prefilter": False},
        extra_headers=_act_as("B", workspace=wb), timeout=12.0)
    seen = {r["source_id"] for r in (body3 or {}).get("results", [])}
    visible = sid in seen
    return CheckResult("public_visibility", visible,
        f"B_sees_A_public={visible} (must be True)  results={len(seen)}  marker={suffix}")
```

- [ ] **Step 2: Register** `("public_visibility", check_public_visibility),`
- [ ] **Step 3: Run live.** Expected `passed=True`, `B_sees_A_public=True`.
- [ ] **Step 4: Commit** — `git commit -am "test(smoke): public_visibility cross-workspace check"`

### Task 5: `check_user_cross_device`

- [ ] **Step 1: Add the function**

```python
def check_user_cross_device(api: str, token: str) -> CheckResult:
    """Same human (same sub) on two devices/workspaces: ingest visibility='user'
    as sub=S in WA → search as sub=S in WB MUST see it; a different sub must
    NOT. Proves 'user' visibility = cross-device-of-same-human."""
    suffix = int(time.time())
    sub_s = f"cd-{suffix}"
    wa, wb = f"cda-{suffix}", f"cdb-{suffix}"
    sid = f"smoke:user-xd:{suffix}"
    code1, _, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-xd",
              "path": "p", "content": f"USER-XD {suffix}", "visibility": "user", "categorize": False},
        extra_headers=_act_as(sub_s, workspace=wa), timeout=15.0)
    if code1 != 200:
        return CheckResult("user_cross_device", False, f"ingest failed http={code1}")
    code2, body2, _ = _http("POST", f"{api}/memory/search", token,
        body={"query": f"USER-XD {suffix}", "top_k": 5, "include_text": True, "llm_prefilter": False},
        extra_headers=_act_as(sub_s, workspace=wb), timeout=12.0)
    same_sub_sees = sid in {r["source_id"] for r in (body2 or {}).get("results", [])}
    code3, body3, _ = _http("POST", f"{api}/memory/search", token,
        body={"query": f"USER-XD {suffix}", "top_k": 5, "include_text": True, "llm_prefilter": False},
        extra_headers=_act_as(f"other-{suffix}", workspace=wb), timeout=12.0)
    other_sub_sees = sid in {r["source_id"] for r in (body3 or {}).get("results", [])}
    ok = same_sub_sees and not other_sub_sees
    return CheckResult("user_cross_device", ok,
        f"same_sub_sees={same_sub_sees}(want True)  other_sub_sees={other_sub_sees}(want False)  marker={suffix}")
```

- [ ] **Step 2: Register** `("user_cross_device", check_user_cross_device),`
- [ ] **Step 3: Run live.** Expected `passed=True`.
- [ ] **Step 4: Commit** — `git commit -am "test(smoke): user_cross_device visibility check"`

### Task 6: `check_org_member_visibility`

- [ ] **Step 1: Add the function**

```python
def check_org_member_visibility(api: str, token: str) -> CheckResult:
    """A (member of org-X) ingests visibility='org' with org_id=X → B (also
    member of org-X, different sub/workspace) MUST see it."""
    suffix = int(time.time())
    org = f"org-{suffix}"
    sid = f"smoke:org-vis:{suffix}"
    code1, body1, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-org",
              "path": "p", "content": f"ORG-VIS {suffix}",
              "visibility": "org", "org_id": org, "categorize": False},
        extra_headers=_act_as("A", orgs=(org,), workspace=f"oa-{suffix}"), timeout=15.0)
    if code1 != 200:
        return CheckResult("org_member_visibility", False, f"ingest failed http={code1}: {body1}")
    code2, body2, _ = _http("POST", f"{api}/memory/search", token,
        body={"query": f"ORG-VIS {suffix}", "top_k": 5, "include_text": True, "llm_prefilter": False},
        extra_headers=_act_as("B", orgs=(org,), workspace=f"ob-{suffix}"), timeout=12.0)
    member_sees = sid in {r["source_id"] for r in (body2 or {}).get("results", [])}
    return CheckResult("org_member_visibility", member_sees,
        f"org_member_sees={member_sees} (must be True)  marker={suffix}")
```

- [ ] **Step 2: Register** `("org_member_visibility", check_org_member_visibility),`
- [ ] **Step 3: Run live.** Expected `passed=True`.
- [ ] **Step 4: Commit** — `git commit -am "test(smoke): org_member_visibility check"`

### Task 7: `check_org_non_member_blocked`

- [ ] **Step 1: Add the function**

```python
def check_org_non_member_blocked(api: str, token: str) -> CheckResult:
    """A (org-X) ingests visibility='org' → C (NOT a member of org-X) must NOT
    see it. The complement of org_member_visibility — proves org isolation."""
    suffix = int(time.time())
    org = f"orgb-{suffix}"
    sid = f"smoke:org-block:{suffix}"
    code1, _, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-orgb",
              "path": "p", "content": f"ORG-BLOCK {suffix}",
              "visibility": "org", "org_id": org, "categorize": False},
        extra_headers=_act_as("A", orgs=(org,), workspace=f"oba-{suffix}"), timeout=15.0)
    if code1 != 200:
        return CheckResult("org_non_member_blocked", False, f"ingest failed http={code1}")
    code2, body2, _ = _http("POST", f"{api}/memory/search", token,
        body={"query": f"ORG-BLOCK {suffix}", "top_k": 5, "include_text": True, "llm_prefilter": False},
        extra_headers=_act_as("C", orgs=(f"other-{suffix}",), workspace=f"obc-{suffix}"), timeout=12.0)
    non_member_sees = sid in {r["source_id"] for r in (body2 or {}).get("results", [])}
    return CheckResult("org_non_member_blocked", not non_member_sees,
        f"non_member_sees={non_member_sees} (must be False)  marker={suffix}")
```

- [ ] **Step 2: Register** `("org_non_member_blocked", check_org_non_member_blocked),`
- [ ] **Step 3: Run live.** Expected `passed=True`.
- [ ] **Step 4: Commit** — `git commit -am "test(smoke): org_non_member_blocked check"`

### Task 8: `check_org_revoke_isolation`

- [ ] **Step 1: Add the function**

```python
def check_org_revoke_isolation(api: str, token: str) -> CheckResult:
    """A ingests visibility='org' as a member of org-X → the SAME sub A, but
    with a token that no longer carries org-X membership (simulating a
    post-revoke reissued JWT), must NOT see it. Proves access dies with the
    membership claim, not just at ingest time."""
    suffix = int(time.time())
    org = f"orgr-{suffix}"
    sid = f"smoke:org-revoke:{suffix}"
    code1, _, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-orgr",
              "path": "p", "content": f"ORG-REVOKE {suffix}",
              "visibility": "org", "org_id": org, "categorize": False},
        extra_headers=_act_as("A", orgs=(org,), workspace=f"ora-{suffix}"), timeout=15.0)
    if code1 != 200:
        return CheckResult("org_revoke_isolation", False, f"ingest failed http={code1}")
    # Same sub A, but org-X membership revoked (no orgs in the act-as identity).
    code2, body2, _ = _http("POST", f"{api}/memory/search", token,
        body={"query": f"ORG-REVOKE {suffix}", "top_k": 5, "include_text": True, "llm_prefilter": False},
        extra_headers=_act_as("A", workspace=f"ora-{suffix}"), timeout=12.0)
    still_sees = sid in {r["source_id"] for r in (body2 or {}).get("results", [])}
    return CheckResult("org_revoke_isolation", not still_sees,
        f"sees_after_revoke={still_sees} (must be False)  marker={suffix}")
```

- [ ] **Step 2: Register** `("org_revoke_isolation", check_org_revoke_isolation),`
- [ ] **Step 3: Run live.** Expected `passed=True`.
- [ ] **Step 4: Commit** — `git commit -am "test(smoke): org_revoke_isolation check"`

### Task 9: `check_patch_visibility_authz`

- [ ] **Step 1: Add the function**

```python
def check_patch_visibility_authz(api: str, token: str) -> CheckResult:
    """A ingests a private source → B (different sub AND workspace) PATCHes its
    visibility → must be 403. Proves L8 owner-check blocks cross-tenant
    vandalism."""
    suffix = int(time.time())
    sid = f"smoke:authz:{suffix}"
    code1, _, _ = _http("POST", f"{api}/memory/put", token,
        body={"source_id": sid, "source_type": "note", "repo": "smoke-authz",
              "path": "p", "content": f"AUTHZ {suffix}", "categorize": False},
        extra_headers=_act_as("A", workspace=f"aza-{suffix}"), timeout=15.0)
    if code1 != 200:
        return CheckResult("patch_visibility_authz", False, f"ingest failed http={code1}")
    code2, body2, _ = _http_await_source("PATCH",
        f"{api}/sources/{urllib.parse.quote(sid, safe='')}/visibility", token,
        body={"visibility": "public"},
        extra_headers=_act_as("B", workspace=f"azb-{suffix}"))
    ok = code2 == 403
    return CheckResult("patch_visibility_authz", ok,
        f"foreign_patch_status={code2} (must be 403)  body={body2}  marker={suffix}")
```

- [ ] **Step 2: Register** `("patch_visibility_authz", check_patch_visibility_authz),`
- [ ] **Step 3: Run live.** Expected `passed=True`, `foreign_patch_status=403`.
- [ ] **Step 4: Commit** — `git commit -am "test(smoke): patch_visibility_authz L8 owner-check"`

### Task 10: `check_multi_org_membership`

- [ ] **Step 1: Add the function**

```python
def check_multi_org_membership(api: str, token: str) -> CheckResult:
    """A is a member of org-X AND org-Y; ingest one org-source in each →
    a single search as A MUST surface BOTH. Proves org_ids is a multi-value
    IN-filter, not a single org."""
    suffix = int(time.time())
    ox, oy = f"mox-{suffix}", f"moy-{suffix}"
    sx, sy = f"smoke:multi-x:{suffix}", f"smoke:multi-y:{suffix}"
    ws = f"moa-{suffix}"
    for sid, org in ((sx, ox), (sy, oy)):
        code, _, _ = _http("POST", f"{api}/memory/put", token,
            body={"source_id": sid, "source_type": "note", "repo": "smoke-multi",
                  "path": "p", "content": f"MULTI-ORG {suffix} {org}",
                  "visibility": "org", "org_id": org, "categorize": False},
            extra_headers=_act_as("A", orgs=(ox, oy), workspace=ws), timeout=15.0)
        if code != 200:
            return CheckResult("multi_org_membership", False, f"ingest {org} failed http={code}")
    code2, body2, _ = _http("POST", f"{api}/memory/search", token,
        body={"query": f"MULTI-ORG {suffix}", "top_k": 10, "include_text": True, "llm_prefilter": False},
        extra_headers=_act_as("A", orgs=(ox, oy), workspace=ws), timeout=12.0)
    seen = {r["source_id"] for r in (body2 or {}).get("results", [])}
    both = sx in seen and sy in seen
    return CheckResult("multi_org_membership", both,
        f"sees_org_x={sx in seen}  sees_org_y={sy in seen} (both must be True)  marker={suffix}")
```

- [ ] **Step 2: Register** `("multi_org_membership", check_multi_org_membership),`
- [ ] **Step 3: Run live.** Expected `passed=True`.
- [ ] **Step 4: Commit** — `git commit -am "test(smoke): multi_org_membership check"`

### Task 11: `check_stats_workspaces_lists_all`

- [ ] **Step 1: Inspect the endpoint first**

Run: `grep -n "stats/workspaces" src/api/routes/dashboard.py` and read the handler to confirm the response shape (a `workspaces` list with per-ws rows) and that it scopes to the caller's identity (active workspace + org memberships) rather than admin-all. Confirm the JSON key names used below (`workspaces`, each row's `id`).

- [ ] **Step 2: Add the function**

```python
def check_stats_workspaces_lists_all(api: str, token: str) -> CheckResult:
    """GET /stats/workspaces for a caller who is a member of org-X and org-Y
    (plus a personal workspace) must list all of them. Proves the dashboard
    enumerates a multi-membership caller's workspaces, not just one."""
    suffix = int(time.time())
    ws = f"swa-{suffix}"
    ox, oy = f"swx-{suffix}", f"swy-{suffix}"
    code, body, _ = _http("GET", f"{api}/stats/workspaces", token,
        extra_headers=_act_as("A", orgs=(ox, oy), workspace=ws), timeout=12.0)
    if code != 200 or not isinstance(body, dict):
        return CheckResult("stats_workspaces_lists_all", False, f"http={code} body={body}")
    ids = {w.get("id") for w in body.get("workspaces", [])}
    # The personal/active ws must be present; org rows appear once they hold data.
    ok = ws in ids
    return CheckResult("stats_workspaces_lists_all", ok,
        f"active_ws_listed={ws in ids}  rows={len(ids)} (active must be listed)  marker={suffix}")
```

> If Step 1 shows `/stats/workspaces` is admin-cross-workspace only (ignores the caller's memberships), STOP and report — that is a real Iter-3 gap to fix before this test is meaningful (the endpoint must scope to `info` for non-admin callers). Adjust the assertion to the confirmed shape.

- [ ] **Step 3: Register** `("stats_workspaces_lists_all", check_stats_workspaces_lists_all),`
- [ ] **Step 4: Run live.** Expected `passed=True`.
- [ ] **Step 5: Commit** — `git commit -am "test(smoke): stats_workspaces_lists_all check"`

---

## Task 12: Enable `MAYRING_ALLOW_ACT_AS` on the server + deploy + full smoke

**Files:**
- Modify: `app.linn.games/docker-compose.mayring.yml` (the `mayring-api` service env — add `MAYRING_ALLOW_ACT_AS: "1"`). This repo is the canonical clone at `/home/nileneb/Desktop/WebDev/app.linn.games`.

**Context:** The act-as override is off by default (`_maybe_act_as` returns early unless the env is set). The prod smoke needs it on. It is safe: privileged-token-only, and the synthetic identity is strictly *less* privileged than the real one (can never escalate).

- [ ] **Step 1: Add the env var** to the `mayring-api` (and `mayring-mcp` if it shares auth) service environment block in `docker-compose.mayring.yml`:

```yaml
      MAYRING_ALLOW_ACT_AS: "1"
```

- [ ] **Step 2: Commit + push both repos**

```bash
# MayringCoder (Tasks 1-11)
cd /home/nileneb/Desktop/MayringCoder && git push origin master
# app.linn.games (compose)
cd /home/nileneb/Desktop/WebDev/app.linn.games && git add docker-compose.mayring.yml \
  && git commit -m "chore(mayring): enable MAYRING_ALLOW_ACT_AS for acceptance smokes" && git push origin main
```

- [ ] **Step 3: Wait for build + deploy**, then run the FULL smoke locally against prod:

```bash
TOK=$(ssh nileneb@u-server "grep '^MCP_SERVICE_TOKEN=' ~/app.linn.games/.env | cut -d= -f2-")
PYTHONPATH=.:core MCP_SERVICE_TOKEN="$TOK" python3 tools/smoke_test_production.py 2>&1 | tail -40
```
Expected: all checks pass, including the 9 new acceptance checks (private_isolation, public_visibility, user_cross_device, org_member_visibility, org_non_member_blocked, org_revoke_isolation, patch_visibility_authz, multi_org_membership, stats_workspaces_lists_all).

- [ ] **Step 4:** If any acceptance check is RED, that is a **real V2 isolation bug** (not a harness flake) — debug with superpowers:systematic-debugging, fix the root cause in the visibility/scope path, re-run. Do NOT loosen the assertion.

- [ ] **Step 5: Update memory** — append to `MEMORY.md` and the v2-workspaces project memory that V2 org/public memory is now acceptance-verified on prod (which checks, commit SHAs).

---

## Deferred / Out of Scope (explicit non-goals)

- **Token reissue of the hook.jwt** — the alias-aware `resolve_workspace_from_token` (`src/api/auth.py:62-70`) already bridges the stale `019d6933` token to `019e14d6` for the workspace dimension. Reissue only matters if the *hook* needs org visibility; defer until there's a concrete need.
- **`/memory/orgs*` proxy endpoints** — Laravel owns org CRUD (`/api/workspaces*` + Livewire); MayringCoder does not need proxies.
- **`GET /api/workspaces` list endpoint** — no current consumer (the Livewire Switcher queries directly).
- **Org-split / project→org reassignment** (DiakonieWhisper → Bergische-Diakonie) — separate later migration.
