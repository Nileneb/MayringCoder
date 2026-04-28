#!/bin/bash
set -e

# ── Repo-Erkennung ───────────────────────────────────────────────────────────
# Drei Fälle:
#   1. Klassisch: Script aus geklontem Repo (../src/api/local_mcp.py vorhanden)
#   2. Repo bereits unter ~/Desktop/MayringCoder vorhanden
#   3. Standalone-Download: Repo wird automatisch nach ~/Desktop/MayringCoder geklont
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
_DEFAULT_DIR="$HOME/Desktop/MayringCoder"

if [ -f "$_SCRIPT_DIR/../src/api/local_mcp.py" ]; then
    # Klassisch: aus geklontem Repo
    MAYRING_DIR="$(cd "$_SCRIPT_DIR/.." && pwd)"
elif [ -f "$_DEFAULT_DIR/src/api/local_mcp.py" ]; then
    # Repo bereits vorhanden
    MAYRING_DIR="$_DEFAULT_DIR"
else
    # Standalone: nur Client-Dateien via sparse checkout holen (kein Server-Code, kein whisper/gradio)
    echo "Klone MayringCoder (client-only, sparse) nach $_DEFAULT_DIR..."
    git clone --filter=blob:none --sparse https://github.com/Nileneb/MayringCoder "$_DEFAULT_DIR"
    git -C "$_DEFAULT_DIR" sparse-checkout set \
        claude-plugin \
        src/__init__.py \
        src/config.py \
        src/agents/__init__.py \
        src/agents/pi.py \
        src/api/__init__.py \
        src/api/local_mcp.py \
        src/api/mcp_agent_tools.py \
        src/api/mcp_auth.py \
        src/api/dependencies.py \
        src/api/memory_service.py \
        src/api/jwt_auth.py \
        src/memory \
        tools/memory_sync.py \
        tools/postcompact_hook.py \
        tools/oauth_install.py \
        requirements-client.txt
    MAYRING_DIR="$_DEFAULT_DIR"
fi

PLUGIN_DIR="$MAYRING_DIR/claude-plugin"
TOOLS_DIR="$MAYRING_DIR/tools"

# ── Plugin-Dateien kopieren ──────────────────────────────────────────────────
DEST="$HOME/.claude/plugins/mayring-coder"
mkdir -p "$DEST"
cp -r "$PLUGIN_DIR/." "$DEST/"
echo "Plugin installiert: $DEST"

# ── Python-Umgebung einrichten ───────────────────────────────────────────────
# venv liegt unter ~/.claude/plugins/mayring-coder/.venv (co-located mit Plugin).
# Überschreibbar via MAYRING_VENV_PYTHON.
VENV_PYTHON="${MAYRING_VENV_PYTHON:-"$DEST/.venv/bin/python"}"
VENV_DIR="$(dirname "$(dirname "$VENV_PYTHON")")"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Erstelle virtuelle Umgebung: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

if [ -f "$MAYRING_DIR/requirements-client.txt" ]; then
    echo "Installiere Client-Abhängigkeiten aus requirements-client.txt..."
    "$VENV_DIR/bin/pip" install -q -r "$MAYRING_DIR/requirements-client.txt"
else
    echo "Installiere Abhängigkeiten aus requirements.txt..."
    "$VENV_DIR/bin/pip" install -q -r "$MAYRING_DIR/requirements.txt"
fi
echo "Abhängigkeiten installiert."

# ── Skills in superpowers-Cache kopieren ─────────────────────────────────────
SP_SKILLS="$HOME/.claude/plugins/cache/claude-plugins-official/superpowers"
SP_VERSION=$(ls "$SP_SKILLS" 2>/dev/null | sort -V | tail -1)
if [ -n "$SP_VERSION" ]; then
    SKILL_DEST="$SP_SKILLS/$SP_VERSION/skills"
    for skill_dir in "$PLUGIN_DIR/skills"/*/; do
        skill_name=$(basename "$skill_dir")
        mkdir -p "$SKILL_DEST/$skill_name"
        cp "$skill_dir/SKILL.md" "$SKILL_DEST/$skill_name/SKILL.md"
        echo "Skill installiert: $skill_name → $SKILL_DEST/$skill_name"
    done
else
    echo "Warnung: superpowers-Plugin nicht gefunden — Skills nicht installiert"
fi

# ── Hooks in ~/.claude/settings.json eintragen ──────────────────────────────
SETTINGS="$HOME/.claude/settings.json"
HOOK_SCRIPT="$DEST/hooks/start_watcher.py"
COMPACT_SCRIPT="$TOOLS_DIR/postcompact_hook.py"
STOP_SCRIPT="$DEST/hooks/stop_hook.py"
SYNC_SCRIPT="$TOOLS_DIR/memory_sync.py"

python3 - <<PYEOF
import json, os

settings_path = os.path.expanduser("$SETTINGS")
hook_script = "$HOOK_SCRIPT"
compact_script = "$COMPACT_SCRIPT"
stop_script = "$STOP_SCRIPT"
sync_script = "$SYNC_SCRIPT"

try:
    with open(settings_path) as f:
        cfg = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    cfg = {}

hooks = cfg.setdefault("hooks", {})

# UserPromptSubmit — Watcher starten
hooks.setdefault("UserPromptSubmit", [])
if not any(any(h.get("command", "").endswith("start_watcher.py") for h in e.get("hooks", [])) for e in hooks["UserPromptSubmit"]):
    hooks["UserPromptSubmit"].append({"matcher": "", "hooks": [{"type": "command", "command": f"python3 {hook_script}"}]})
    print("Hook hinzugefügt: UserPromptSubmit → start_watcher.py")
else:
    print("Hook bereits vorhanden: UserPromptSubmit → start_watcher.py")

# UserPromptSubmit — memory_sync
if not any(any(h.get("command", "").endswith("memory_sync.py") for h in e.get("hooks", [])) for e in hooks["UserPromptSubmit"]):
    hooks["UserPromptSubmit"].append({"matcher": "", "hooks": [{"type": "command", "command": f"python3 {sync_script}"}]})
    print("Hook hinzugefügt: UserPromptSubmit → memory_sync.py")
else:
    print("Hook bereits vorhanden: UserPromptSubmit → memory_sync.py")

# PostCompact — Summary ingesten
hooks.setdefault("PostCompact", [])
if not any(any(h.get("command", "").endswith("postcompact_hook.py") for h in e.get("hooks", [])) for e in hooks["PostCompact"]):
    hooks["PostCompact"].append({"matcher": "", "hooks": [{"type": "command", "command": f"python3 {compact_script}"}]})
    print("Hook hinzugefügt: PostCompact → postcompact_hook.py")
else:
    print("Hook bereits vorhanden: PostCompact → postcompact_hook.py")

# Stop — unbeurteilte Chunks → neutral
hooks.setdefault("Stop", [])
if not any(any(h.get("command", "").endswith("stop_hook.py") for h in e.get("hooks", [])) for e in hooks["Stop"]):
    hooks["Stop"].append({"matcher": "", "hooks": [{"type": "command", "command": f"python3 {stop_script}"}]})
    print("Hook hinzugefügt: Stop → stop_hook.py")
else:
    print("Hook bereits vorhanden: Stop → stop_hook.py")

with open(settings_path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PYEOF

# ── MCP-Server in ~/.claude/.mcp.json eintragen ──────────────────────────────
# command: venv-Python aus Plugin-Verzeichnis
# cwd: MayringCoder-Repo (src/api/local_mcp.py liegt dort)
MCP_JSON="$HOME/.claude/.mcp.json"
python3 - <<MCPEOF
import json, os

mcp_json_path = os.path.expanduser("$MCP_JSON")
mayring_dir = "$MAYRING_DIR"
venv_python = "$VENV_PYTHON"

try:
    with open(mcp_json_path) as f:
        cfg = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    cfg = {}

mcp_servers = cfg.setdefault("mcpServers", {})
if "memory-agents" not in mcp_servers:
    mcp_servers["memory-agents"] = {
        "command": venv_python,
        "args": ["-m", "src.api.local_mcp"],
        "cwd": mayring_dir,
        "env": {"PYTHONPATH": mayring_dir},
    }
    print(f"MCP-Server hinzugefügt: memory-agents (cwd={mayring_dir})")
else:
    print("MCP-Server bereits vorhanden: memory-agents")

with open(mcp_json_path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
MCPEOF

# ── Auth-Token einrichten via OAuth PKCE ─────────────────────────────────────
HOOK_JWT="$HOME/.config/mayring/hook.jwt"
OAUTH_SCRIPT="$TOOLS_DIR/oauth_install.py"
echo ""
if [ -f "$HOOK_JWT" ] && [ -s "$HOOK_JWT" ]; then
    echo "Token bereits vorhanden: $HOOK_JWT"
else
    echo "=== Mayring JWT einrichten (OAuth-Login) ==="
    python3 "$OAUTH_SCRIPT" --jwt-file "$HOOK_JWT"
fi
echo ""
echo "Watcher-Log: ~/.cache/mayringcoder/watcher.log"
