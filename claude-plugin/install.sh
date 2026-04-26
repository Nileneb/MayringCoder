#!/bin/bash
set -e
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
MAYRING_DIR="$(dirname "$PLUGIN_DIR")"
TOOLS_DIR="$MAYRING_DIR/tools"

# Plugin-Dateien
DEST="$HOME/.claude/plugins/mayring-coder"
mkdir -p "$DEST"
cp -r "$PLUGIN_DIR/." "$DEST/"
echo "Plugin installiert: $DEST"

# Python-Umgebung einrichten (benötigt für memory-agents MCP-Server)
VENV_PYTHON="${MAYRING_VENV_PYTHON:-"$MAYRING_DIR/.venv/bin/python"}"
VENV_DIR="$(dirname "$(dirname "$VENV_PYTHON")")"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Erstelle virtuelle Umgebung: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

echo "Installiere Abhängigkeiten aus requirements.txt..."
"$VENV_DIR/bin/pip" install -q -r "$MAYRING_DIR/requirements.txt"
echo "Abhängigkeiten installiert."

# Skills in superpowers-Cache kopieren (überleben Superpowers-Updates nicht — daher hier)
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

# Hooks in ~/.claude/settings.json eintragen (UserPromptSubmit + PostCompact)
SETTINGS="$HOME/.claude/settings.json"
HOOK_SCRIPT="$DEST/hooks/start_watcher.py"
COMPACT_SCRIPT="$TOOLS_DIR/postcompact_hook.py"
STOP_SCRIPT="$DEST/hooks/stop_hook.py"

python3 - <<PYEOF
import json, os, sys

settings_path = os.path.expanduser("$SETTINGS")
hook_script = "$HOOK_SCRIPT"
compact_script = "$COMPACT_SCRIPT"
stop_script = "$STOP_SCRIPT"

try:
    with open(settings_path) as f:
        cfg = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    cfg = {}

hooks = cfg.setdefault("hooks", {})

# UserPromptSubmit — Watcher starten
hooks.setdefault("UserPromptSubmit", [])
watcher_hook = {"type": "command", "command": f"python3 {hook_script}"}
watcher_entry = {"matcher": "", "hooks": [watcher_hook]}
already = any(
    any(h.get("command", "").endswith("start_watcher.py")
        for h in e.get("hooks", []))
    for e in hooks["UserPromptSubmit"]
)
if not already:
    hooks["UserPromptSubmit"].append(watcher_entry)
    print("Hook hinzugefügt: UserPromptSubmit → start_watcher.py")
else:
    print("Hook bereits vorhanden: UserPromptSubmit")

# PostCompact — Summary ingesten
hooks.setdefault("PostCompact", [])
compact_hook = {"type": "command", "command": f"python3 {compact_script}"}
compact_entry = {"matcher": "", "hooks": [compact_hook]}
already_compact = any(
    any(h.get("command", "").endswith("postcompact_hook.py")
        for h in e.get("hooks", []))
    for e in hooks["PostCompact"]
)
if not already_compact:
    hooks["PostCompact"].append(compact_entry)
    print("Hook hinzugefügt: PostCompact → postcompact_hook.py")
else:
    print("Hook bereits vorhanden: PostCompact")

# Stop — unbeurteilte injizierte Chunks → signal=neutral
hooks.setdefault("Stop", [])
stop_hook = {"type": "command", "command": f"python3 {stop_script}"}
stop_entry = {"matcher": "", "hooks": [stop_hook]}
already_stop = any(
    any(h.get("command", "").endswith("stop_hook.py")
        for h in e.get("hooks", []))
    for e in hooks["Stop"]
)
if not already_stop:
    hooks["Stop"].append(stop_entry)
    print("Hook hinzugefügt: Stop → stop_hook.py")
else:
    print("Hook bereits vorhanden: Stop")

# UserPromptSubmit — memory_sync background hook
SYNC_SCRIPT = "$TOOLS_DIR/memory_sync.py"
sync_hook = {"type": "command", "command": f"python3 {SYNC_SCRIPT}"}
sync_entry = {"matcher": "", "hooks": [sync_hook]}
already_sync = any(
    any(h.get("command", "").endswith("memory_sync.py")
        for h in e.get("hooks", []))
    for e in hooks["UserPromptSubmit"]
)
if not already_sync:
    hooks["UserPromptSubmit"].append(sync_entry)
    print("Hook hinzugefügt: UserPromptSubmit → memory_sync.py")
else:
    print("Hook bereits vorhanden: memory_sync.py")

with open(settings_path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PYEOF

# Lokalen MCP-Agent-Server in ~/.claude/settings.json eintragen (memory-agents)
# Konfigurierbarer Pfad zum Python-Interpreter des virtuellen Environments:
# Standard ist "$MAYRING_DIR/.venv/bin/python", kann aber via MAYRING_VENV_PYTHON überschrieben werden.
VENV_PYTHON="${MAYRING_VENV_PYTHON:-"$MAYRING_DIR/.venv/bin/python"}"
python3 - <<MCPEOF
import json, os, sys

settings_path = os.path.expanduser("$SETTINGS")
mayring_dir = "$MAYRING_DIR"
venv_python = "$VENV_PYTHON"

try:
    with open(settings_path) as f:
        cfg = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    cfg = {}

mcp_servers = cfg.setdefault("mcpServers", {})
if "memory-agents" not in mcp_servers:
    mcp_servers["memory-agents"] = {
        "command": venv_python,
        "args": ["-m", "src.api.local_mcp"],
        "cwd": mayring_dir,
    }
    print("MCP-Server hinzugefügt: memory-agents (lokaler Agent-Server)")
else:
    print("MCP-Server bereits vorhanden: memory-agents")

with open(settings_path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
MCPEOF

# Auth-Token einrichten via OAuth PKCE (vollautomatisch, kein Copy-Paste)
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
