#!/usr/bin/env bash
# Bootstrap + run wrapper for the local memory-agents MCP server.
#
# Why a wrapper: the SessionStart hook cannot bootstrap the venv reliably
# because Claude Code starts plugin MCP servers BEFORE hooks fire. So when a
# fresh /plugin install lands, the canonical command
#   ${CLAUDE_PLUGIN_ROOT}/.venv/bin/python -m src.api.local_mcp
# fails immediately ("python not found"), the user sees a silent reconnect
# error, and there is no second chance.
#
# This script gives the MCP-server-start path itself the means to fix that:
# it checks the venv on every boot, builds it on first run (~30s), then
# exec's into the Python entrypoint so stdio JSON-RPC works exactly as before.
#
# Idempotent: a healthy venv is detected and the bootstrap is skipped.
# Quick-path: when nothing needs to be done, the script adds <50ms to start-up.

set -euo pipefail

# --- Paths --------------------------------------------------------------------
# CLAUDE_PLUGIN_ROOT is set by Claude Code when launching plugin commands.
# Fall back to the script's own directory layout for non-runtime invocations
# (e.g. local debugging: bash claude-plugin/bin/run_local_mcp.sh).
if [ -z "${CLAUDE_PLUGIN_ROOT:-}" ]; then
    CLAUDE_PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

REPO_ROOT="$(cd "$CLAUDE_PLUGIN_ROOT/.." && pwd)"
VENV_DIR="$CLAUDE_PLUGIN_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
REQ_FILE="$REPO_ROOT/requirements-client.txt"

# --- 1. venv (one-time bootstrap) ---------------------------------------------
needs_bootstrap=0
if [ ! -x "$VENV_PYTHON" ]; then
    needs_bootstrap=1
elif [ ! -x "$VENV_DIR/bin/pip" ]; then
    # Some users delete pip to free space — we need it to install deps.
    needs_bootstrap=1
fi

if [ "$needs_bootstrap" = "1" ]; then
    if [ ! -f "$REQ_FILE" ]; then
        echo "run_local_mcp: $REQ_FILE missing — marketplace clone incomplete?" >&2
        exit 1
    fi
    echo "run_local_mcp: first-run bootstrap (creating venv at $VENV_DIR, ~30s)" >&2
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -q -r "$REQ_FILE" >&2
    echo "run_local_mcp: bootstrap complete" >&2
fi

# --- 2. JWT (best-effort, do NOT block MCP-start) ---------------------------
JWT_FILE="${MAYRING_HOOK_JWT:-$HOME/.config/mayring/hook.jwt}"
if [ ! -s "$JWT_FILE" ] && [ -f "$REPO_ROOT/tools/oauth_install.py" ]; then
    echo "run_local_mcp: hook.jwt missing — run 'python3 $REPO_ROOT/tools/oauth_install.py' once to enable cloud features" >&2
fi

# --- 3. Hand off to the Python MCP server -------------------------------------
# `exec` keeps the same PID so Claude Code's stdio attached to the wrapper
# survives the transition into Python.
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_PYTHON" -m src.api.local_mcp "$@"
