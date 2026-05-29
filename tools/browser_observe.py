#!/usr/bin/env python3
"""Live browser observer for co-testing app.linn.games (Stufe 2).

Launches a HEADFUL persistent-profile Chromium that the USER drives. Logs every
JS console error/warning, page error, browser crash, and HTTP response >=400 to a
line-buffered log so the assistant can `Read` it on demand and see what broke while
the user tested — WITHOUT driving the browser itself.

Run with the /app-walkthrough venv (has playwright + the already-authed profile):
  ~/.cache/app-walkthrough/venv/bin/python tools/browser_observe.py [--base URL] [--log PATH]

Reuses ~/.cache/app-walkthrough/profile (same login as /app-walkthrough; if the
session expired, run `app_walkthrough.py --login` once first). Close the browser
window or Ctrl-C to stop. Pairs with the on-demand /app-walkthrough sweep (Stufe 3).
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

PROFILE = Path.home() / ".cache/app-walkthrough/profile"


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser(description="Live browser console/network observer")
    ap.add_argument("--base", default="https://app.linn.games")
    ap.add_argument("--log", default="/tmp/browser-observer/events.log")
    args = ap.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = log_path.open("a", buffering=1)  # line-buffered → readable live

    # File gets EVERYTHING (Read on demand); stdout gets only SERIOUS events so a
    # Monitor attached to stdout pings on real crashes, not on every console warning.
    _SERIOUS = {"PAGEERROR", "CRASH", "OBSERVER"}

    def emit(kind: str, detail: str, *, serious: bool = False) -> None:
        line = f"[{_ts()}] {kind}: {detail}"
        logf.write(line + "\n")
        if serious or kind in _SERIOUS:
            print(line, flush=True)

    if not PROFILE.exists():
        sys.exit("NOT_LOGGED_IN: erst `app_walkthrough.py --login` ausführen")

    from playwright.sync_api import sync_playwright

    emit("OBSERVER", f"start base={args.base} log={log_path}")
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE), headless=False,
            args=["--start-maximized"], no_viewport=True,
        )

        def wire(page) -> None:
            page.on("console", lambda m: (
                emit("CONSOLE", f"[{m.type}] {m.text}  @ {page.url}")
                if m.type in ("error", "warning") else None))
            page.on("pageerror", lambda e: emit("PAGEERROR", f"{e}  @ {page.url}"))
            page.on("crash", lambda: emit("CRASH", f"page crashed @ {page.url}"))
            page.on("response", lambda r: (
                emit("HTTP>=400", f"{r.status} {r.request.method} {r.url}")
                if r.status >= 400 else None))

        ctx.on("page", wire)
        for pg in ctx.pages:
            wire(pg)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.goto(args.base, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:  # noqa: BLE001 — surface, don't swallow
            emit("OBSERVER", f"initial goto failed: {e}")
        emit("OBSERVER", "bereit — teste im Fenster; Events landen hier. Fenster schließen zum Beenden.")

        # Block while the user drives. Exit when the last window closes (ctx access raises).
        try:
            while True:
                time.sleep(1)
                if not ctx.pages:
                    break
        except Exception:
            pass
    emit("OBSERVER", "stopped")


if __name__ == "__main__":
    main()
