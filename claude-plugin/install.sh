#!/bin/bash
set -e
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"

# Plugin-Dateien
DEST="$HOME/.claude/plugins/mayring-coder"
mkdir -p "$DEST"
cp -r "$PLUGIN_DIR/." "$DEST/"
echo "Plugin installiert: $DEST"

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
COMPACT_SCRIPT="$HOME/Desktop/MayringCoder/tools/postcompact_hook.py"
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

with open(settings_path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PYEOF

# Auth-Token einrichten via OAuth PKCE (vollautomatisch, kein Copy-Paste)
HOOK_JWT="$HOME/.config/mayring/hook.jwt"
OAUTH_SCRIPT="$HOME/Desktop/MayringCoder/tools/oauth_install.py"
echo ""
if [ -f "$HOOK_JWT" ] && [ -s "$HOOK_JWT" ]; then
    echo "Token bereits vorhanden: $HOOK_JWT"
else
    echo "=== Mayring JWT einrichten (OAuth-Login) ==="
    python3 "$OAUTH_SCRIPT" --jwt-file "$HOOK_JWT"
fi
echo ""
echo "Watcher-Log: ~/.cache/mayringcoder/watcher.log"
