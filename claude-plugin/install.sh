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

echo ""
echo "MAYRING_JWT in der Shell setzen:"
echo "  export MAYRING_JWT=<Token von app.linn.games/mayring/watcher>"
