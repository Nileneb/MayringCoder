#!/usr/bin/env python3
"""SessionStart hook: idempotent bootstrap + Memory-Kontext-Injection.

Two-phase per session:
  1) Bootstrap (only when something is missing): create venv, install
     requirements-client.txt, run OAuth-PKCE for the hook JWT.
  2) Memory inject: query mcp.linn.games /search and print the result block
     so Claude sees relevant context for the user's first prompt.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error


PLANS_DIR = os.path.expanduser("~/.claude/plans")
TASK_CONTEXT_BUDGET = 800
JWT_FILE = os.path.expanduser("~/.config/mayring/hook.jwt")


def _plugin_root() -> str:
    """Resolve the plugin's root directory.

    Prefers the runtime-provided CLAUDE_PLUGIN_ROOT; falls back to the
    grandparent of this file (claude-plugin/ when laid out via the marketplace).
    """
    env = os.environ.get("CLAUDE_PLUGIN_ROOT")
    if env:
        return env
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _repo_root(plugin_root: str) -> str:
    """The repo containing src/api/local_mcp.py.

    Searches in order:
      1. plugin_root/.. (dev clone where claude-plugin/ lives in repo)
      2. $MAYRING_REPO_ROOT (explicit override)

    No more hardcoded path probing — auto-sync just no-ops on machines without
    a local clone, which is the right default. Set MAYRING_REPO_ROOT if you
    want plugin-file auto-sync to work.
    """
    parent = os.path.abspath(os.path.join(plugin_root, ".."))
    if os.path.isfile(os.path.join(parent, "src", "api", "local_mcp.py")):
        return parent
    env_root = os.environ.get("MAYRING_REPO_ROOT", "")
    if env_root and os.path.isfile(os.path.join(env_root, "src", "api", "local_mcp.py")):
        return os.path.abspath(env_root)
    return plugin_root


def _venv_python(venv_dir: str) -> str:
    return os.path.join(venv_dir, "bin", "python")


def _venv_is_healthy(venv_dir: str) -> bool:
    py = _venv_python(venv_dir)
    pip = os.path.join(venv_dir, "bin", "pip")
    if not (os.path.isfile(py) and os.path.isfile(pip)):
        return False
    real = os.path.realpath(py)
    return os.path.isfile(real)


def _ensure_venv(plugin_root: str, repo_root: str) -> None:
    venv_dir = os.path.join(plugin_root, ".venv")
    requirements = os.path.join(repo_root, "requirements-client.txt")
    if _venv_is_healthy(venv_dir):
        return
    if not os.path.isfile(requirements):
        print(
            f"MayringCoder bootstrap: skipped (no {requirements}); is the marketplace clone complete?",
            file=sys.stderr,
        )
        return
    print(
        f"MayringCoder bootstrap: creating venv at {venv_dir} (one-time, ~30-60s)",
        file=sys.stderr,
    )
    try:
        # Trusted args: sys.executable is Python's own path; venv_dir derives
        # from CLAUDE_PLUGIN_ROOT (or this file's location). No user-controlled input.
        subprocess.run(  # nosec B603
            [sys.executable, "-m", "venv", "--clear", venv_dir],
            check=True,
            capture_output=True,
        )
        subprocess.run(  # nosec B603
            [os.path.join(venv_dir, "bin", "pip"), "install", "-q", "-r", requirements],
            check=True,
            capture_output=True,
        )
        print("MayringCoder bootstrap: venv ready", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(
            f"MayringCoder bootstrap: venv setup failed — {e.stderr.decode(errors='ignore')[:300]}",
            file=sys.stderr,
        )


def _have_jwt() -> bool:
    return os.path.isfile(JWT_FILE) and os.path.getsize(JWT_FILE) > 0


def _ensure_jwt(repo_root: str, python_executable: str) -> None:
    if _have_jwt():
        return
    oauth_script = os.path.join(repo_root, "tools", "oauth_install.py")
    if not os.path.isfile(oauth_script):
        print(
            "MayringCoder bootstrap: JWT setup skipped (oauth_install.py missing in repo)",
            file=sys.stderr,
        )
        return
    print(
        "MayringCoder bootstrap: opening browser for hook JWT (OAuth PKCE)",
        file=sys.stderr,
    )
    try:
        # Trusted argv: python_executable resolves from venv-or-sys.executable,
        # oauth_script + JWT_FILE are module constants — no user input.
        subprocess.run(  # nosec B603
            [python_executable, oauth_script, "--jwt-file", JWT_FILE],
            check=True,
            timeout=180,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print(
            "MayringCoder bootstrap: JWT setup failed — run tools/oauth_install.py manually",
            file=sys.stderr,
        )


def _bootstrap_if_needed() -> None:
    plugin_root = _plugin_root()
    repo_root = _repo_root(plugin_root)
    _ensure_venv(plugin_root, repo_root)
    venv_dir = os.path.join(plugin_root, ".venv")
    python_executable = _venv_python(venv_dir) if _venv_is_healthy(venv_dir) else sys.executable
    _ensure_jwt(repo_root, python_executable)


def _load_token() -> str:
    token = os.getenv("MCP_SERVICE_TOKEN", "")
    if token:
        return token
    try:
        with open(JWT_FILE) as f:
            content = f.read().strip()
            if content:
                return content
    except OSError:
        pass
    for env_file in [
        os.path.expanduser("~/app.linn.games/.env.mayring"),
        os.path.expanduser("~/.env.mayring"),
    ]:
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MCP_SERVICE_TOKEN="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


def _latest_plan_context() -> str:
    try:
        files = glob.glob(os.path.join(PLANS_DIR, "*.md"))
        if not files:
            return ""
        latest = max(files, key=os.path.getmtime)
        with open(latest) as f:
            content = f.read()
        if "## Context" in content:
            after = content.split("## Context", 1)[1]
            if "\n## " in after:
                after = after.split("\n## ", 1)[0]
            return after.strip()[:TASK_CONTEXT_BUDGET]
        return content[:TASK_CONTEXT_BUDGET]
    except Exception:
        return ""


def _inject_memory(payload: dict) -> None:
    token = _load_token()
    if not token:
        return

    api = os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
    initial_message = (
        payload.get("message")
        or payload.get("prompt")
        or payload.get("transcript", [{}])[0].get("content", "")
        or "session start"
    )
    body_dict: dict = {
        "query": str(initial_message)[:300],
        "top_k": 5,
        "include_text": True,
        "char_budget": 3000,
    }
    task_context = _latest_plan_context()
    if task_context:
        body_dict["task_context"] = task_context

    body = json.dumps(body_dict).encode()
    req = urllib.request.Request(
        f"{api}/search",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            result = json.loads(resp.read())
        context = result.get("prompt_context", "").strip()
        if context:
            print(
                f"## Automatisch injizierter Memory-Kontext\n\n{context}\n\n"
                "⚠️  PFLICHT: Gib am Ende jedes Tasks Feedback via "
                "`mcp__claude_ai_Memory__feedback` für jeden genutzten Chunk."
            )
    except Exception:
        pass


def _sync_plugin_files_from_repo(plugin_root: str, repo_root: str) -> None:
    """Mirror claude-plugin/hooks/*.py and hooks.json from the local repo into
    the marketplace plugin cache so a fresh `git push` is reflected without
    requiring the user to run `/plugin update mayring-coder` manually.

    Only runs when:
      1. plugin_root != repo_root (i.e. we ARE the marketplace cache copy, not
         the dev copy itself), AND
      2. The repo has a newer mtime on at least one hook file.

    No-op when the hook is invoked from a dev clone (plugin_root == repo_root)
    or the repo isn't accessible.
    """
    if os.path.abspath(plugin_root) == os.path.abspath(repo_root):
        return
    src_dir = os.path.join(repo_root, "claude-plugin", "hooks")
    dst_dir = os.path.join(plugin_root, "hooks")
    if not os.path.isdir(src_dir) or not os.path.isdir(dst_dir):
        return
    import shutil
    synced: list[str] = []
    try:
        for name in os.listdir(src_dir):
            if not (name.endswith(".py") or name.endswith(".json")):
                continue
            src = os.path.join(src_dir, name)
            dst = os.path.join(dst_dir, name)
            try:
                src_mtime = os.path.getmtime(src)
                dst_mtime = os.path.getmtime(dst) if os.path.isfile(dst) else 0
                if src_mtime > dst_mtime + 1.0:  # 1s slack vs filesystem precision
                    shutil.copy2(src, dst)
                    synced.append(name)
            except OSError:
                continue
    except OSError:
        return
    if synced:
        print(f"MayringCoder plugin hooks synced from repo: {', '.join(synced)}", file=sys.stderr)


if __name__ == "__main__":
    plugin_root = _plugin_root()
    repo_root = _repo_root(plugin_root)
    _sync_plugin_files_from_repo(plugin_root, repo_root)
    _bootstrap_if_needed()
    payload = json.loads(sys.stdin.read() or "{}")
    _inject_memory(payload)
