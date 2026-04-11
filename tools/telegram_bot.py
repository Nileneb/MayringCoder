#!/usr/bin/env python3
"""Telegram bot — remote interface to Claude Code.

Forwards messages to `claude -p` in the active workspace.
Only responds to whitelisted Telegram user IDs.

Setup:
    pip install python-telegram-bot
    Set env vars (see .env.telegram.example):
        TELEGRAM_BOT_TOKEN=...
        TELEGRAM_ALLOWED_IDS=123456789,987654321
        WORKSPACE_MAYRNG=/home/nileneb/Desktop/MayringCoder
        WORKSPACE_NEW=/home/nileneb/Desktop/NewProject    # optional

Commands:
    /ws list          — list available workspaces
    /ws <name>        — switch workspace
    /status           — show active workspace
    /cancel           — cancel running command
    Any other text    — forwarded to claude -p in active workspace

Start:
    .venv/bin/python tools/telegram_bot.py
    or: nohup .venv/bin/python tools/telegram_bot.py > /tmp/telegram_bot.log 2>&1 &
"""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

# Allow running from repo root or tools/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(".env.telegram")
load_dotenv()  # fallback to main .env

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
    from telegram.constants import ParseMode
except ImportError:
    print("Missing dependency: pip install python-telegram-bot")
    sys.exit(1)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("telegram_bot")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_raw_ids = os.getenv("TELEGRAM_ALLOWED_IDS", "")
ALLOWED_IDS: set[int] = {int(x.strip()) for x in _raw_ids.split(",") if x.strip().isdigit()}

WORKSPACES: dict[str, str] = {}
for key, val in os.environ.items():
    if key.startswith("WORKSPACE_") and val:
        name = key[len("WORKSPACE_"):].lower()
        WORKSPACES[name] = val

# Defaults
if "mayrng" not in WORKSPACES:
    WORKSPACES["mayrng"] = str(Path(__file__).parent.parent)

CLAUDE_BIN = os.getenv("CLAUDE_BIN", "claude")

# ---------------------------------------------------------------------------
# State (per user)
# ---------------------------------------------------------------------------

_active_workspace: dict[int, str] = {}
_running_procs: dict[int, asyncio.subprocess.Process] = {}


def _workspace_for(user_id: int) -> str:
    return _active_workspace.get(user_id, list(WORKSPACES.values())[0])


def _workspace_name_for(user_id: int) -> str:
    path = _workspace_for(user_id)
    for name, p in WORKSPACES.items():
        if p == path:
            return name
    return path


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _authorized(update: Update) -> bool:
    if not ALLOWED_IDS:
        log.warning("TELEGRAM_ALLOWED_IDS not set — bot is open to everyone!")
        return True
    user = update.effective_user
    if user and user.id in ALLOWED_IDS:
        return True
    log.warning("Unauthorized access attempt from user_id=%s", user.id if user else "?")
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _send_chunked(update: Update, text: str) -> None:
    """Send long text in ≤4096-char chunks."""
    if not text.strip():
        return
    for i in range(0, len(text), 4000):
        await update.message.reply_text(text[i:i + 4000])


async def _run_claude(update: Update, user_id: int, prompt: str) -> None:
    """Run claude -p in the active workspace and stream output back."""
    cwd = _workspace_for(user_id)
    ws_name = _workspace_name_for(user_id)

    await update.message.reply_text(f"⚙ `{ws_name}` — running…", parse_mode=ParseMode.MARKDOWN)

    try:
        proc = await asyncio.create_subprocess_exec(
            CLAUDE_BIN, "-p", prompt,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        _running_procs[user_id] = proc

        output_chunks: list[str] = []
        buffer = ""

        assert proc.stdout is not None
        async for line in proc.stdout:
            buffer += line.decode("utf-8", errors="replace")
            # Flush every ~3000 chars to stay under Telegram limit
            if len(buffer) >= 3000:
                output_chunks.append(buffer)
                await update.message.reply_text(buffer[:4000])
                buffer = ""

        await proc.wait()
        _running_procs.pop(user_id, None)

        if buffer.strip():
            await update.message.reply_text(buffer[:4000])

        if not output_chunks and not buffer.strip():
            await update.message.reply_text("✓ Done (no output)")

    except FileNotFoundError:
        await update.message.reply_text(
            f"❌ `claude` binary not found.\n"
            f"Set CLAUDE_BIN in .env.telegram or ensure `claude` is on PATH.",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as exc:
        await update.message.reply_text(f"❌ Error: {exc}")
        _running_procs.pop(user_id, None)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    ws = _workspace_name_for(update.effective_user.id)
    workspaces_list = "\n".join(f"  • `{n}` → `{p}`" for n, p in WORKSPACES.items())
    await update.message.reply_text(
        f"👋 Claude Code bot ready.\n\n"
        f"Active workspace: `{ws}`\n\n"
        f"Available workspaces:\n{workspaces_list}\n\n"
        f"Commands:\n"
        f"  `/ws <name>` — switch workspace\n"
        f"  `/ws list` — list workspaces\n"
        f"  `/status` — current workspace\n"
        f"  `/cancel` — cancel running command\n\n"
        f"Just type a prompt to talk to Claude Code.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    user_id = update.effective_user.id
    ws_name = _workspace_name_for(user_id)
    ws_path = _workspace_for(user_id)
    running = user_id in _running_procs
    await update.message.reply_text(
        f"Workspace: `{ws_name}`\nPath: `{ws_path}`\n{'⚙ Command running' if running else '✓ Idle'}",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_ws(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    user_id = update.effective_user.id
    args = context.args or []

    if not args or args[0] == "list":
        lines = "\n".join(f"  • `{n}` → `{p}`" for n, p in WORKSPACES.items())
        current = _workspace_name_for(user_id)
        await update.message.reply_text(
            f"Available workspaces (active: `{current}`):\n{lines}",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    name = args[0].lower()
    if name not in WORKSPACES:
        await update.message.reply_text(
            f"Unknown workspace `{name}`. Available: {', '.join(f'`{k}`' for k in WORKSPACES)}",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    _active_workspace[user_id] = WORKSPACES[name]
    await update.message.reply_text(
        f"✓ Switched to `{name}` (`{WORKSPACES[name]}`)",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    user_id = update.effective_user.id
    proc = _running_procs.pop(user_id, None)
    if proc:
        proc.terminate()
        await update.message.reply_text("✓ Command cancelled.")
    else:
        await update.message.reply_text("Nothing running.")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    if user_id in _running_procs:
        await update.message.reply_text("⚠ Command already running. Use /cancel to abort.")
        return

    await _run_claude(update, user_id, update.message.text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env.telegram")
        sys.exit(1)
    if not ALLOWED_IDS:
        print("WARNING: TELEGRAM_ALLOWED_IDS not set — bot responds to everyone!")

    print(f"Starting bot with workspaces: {list(WORKSPACES.keys())}")
    print(f"Allowed user IDs: {ALLOWED_IDS or 'ALL (no restriction!)'}")

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("ws", cmd_ws))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    print("Bot running. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
