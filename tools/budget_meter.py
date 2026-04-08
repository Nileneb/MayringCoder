#!/usr/bin/env python3
"""API-Budget-Ampel: Vergleicht Anthropic API-Kosten mit Flat-Rate.

Verwendung:
    # Token-Verbrauch einer Session loggen
    python tools/budget_meter.py log --input 45000 --output 12000 --model sonnet

    # Aktuellen Monat auswerten
    python tools/budget_meter.py status

    # Für Claude Code Statusline (~/.claude/settings.json → statusLine.command)
    python tools/budget_meter.py statusline
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Preise (USD per 1M Tokens, Stand 2025)
# ---------------------------------------------------------------------------

PRICES: dict[str, dict[str, float]] = {
    "sonnet": {"input": 3.00, "output": 15.00},   # claude-sonnet-4-6
    "opus":   {"input": 15.00, "output": 75.00},  # claude-opus-4-6
    "haiku":  {"input": 0.80,  "output": 4.00},   # claude-haiku-4-5
}

USD_TO_EUR = 0.92  # Näherungswert

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_DATA_FILE = Path.home() / ".claude" / "budget_meter.json"


def _load() -> dict:
    if _DATA_FILE.exists():
        try:
            return json.loads(_DATA_FILE.read_text())
        except Exception:
            pass
    return {"sessions": []}


def _save(data: dict) -> None:
    _DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    _DATA_FILE.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Kernlogik
# ---------------------------------------------------------------------------

def _month_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def log_session(input_tokens: int, output_tokens: int, model: str) -> None:
    data = _load()
    data["sessions"].append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "month": _month_key(),
        "model": model,
        "input": input_tokens,
        "output": output_tokens,
    })
    _save(data)
    cost = _session_cost_usd(input_tokens, output_tokens, model)
    print(f"Session geloggt — Kosten: ${cost:.4f} ({cost * USD_TO_EUR:.4f} €)")


def _session_cost_usd(input_t: int, output_t: int, model: str) -> float:
    p = PRICES.get(model, PRICES["sonnet"])
    return (input_t / 1_000_000) * p["input"] + (output_t / 1_000_000) * p["output"]


def _month_stats(month: str | None = None) -> dict:
    month = month or _month_key()
    data = _load()
    sessions = [s for s in data["sessions"] if s.get("month") == month]
    total_input = sum(s["input"] for s in sessions)
    total_output = sum(s["output"] for s in sessions)
    total_cost_usd = sum(
        _session_cost_usd(s["input"], s["output"], s.get("model", "sonnet"))
        for s in sessions
    )
    return {
        "month": month,
        "sessions": len(sessions),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cost_usd": total_cost_usd,
        "cost_eur": total_cost_usd * USD_TO_EUR,
    }


def _ampel(cost_eur: float, flat_rate_eur: float) -> tuple[str, str]:
    ratio = cost_eur / flat_rate_eur if flat_rate_eur > 0 else 0
    if ratio < 0.5:
        return "🟢", f"API günstiger ({cost_eur:.2f} € vs {flat_rate_eur:.0f} € Flat)"
    if ratio < 0.85:
        return "🟡", f"Flat lohnt sich ab ~{flat_rate_eur / max(cost_eur, 0.01):.1f}x mehr Nutzung"
    return "🔴", f"Flat-Rate günstiger! ({cost_eur:.2f} € ≥ {flat_rate_eur * 0.85:.0f} € Schwelle)"


def status(flat_rate_eur: float = 100.0) -> None:
    s = _month_stats()
    color, msg = _ampel(s["cost_eur"], flat_rate_eur)

    print(f"\n{color}  API-Budget {s['month']}")
    print(f"   Sessions:      {s['sessions']}")
    print(f"   Input Tokens:  {s['input_tokens']:,}")
    print(f"   Output Tokens: {s['output_tokens']:,}")
    print(f"   API-Kosten:    {s['cost_eur']:.4f} € (${s['cost_usd']:.4f})")
    print(f"   Flat-Rate:     {flat_rate_eur:.0f} €")
    print(f"   → {msg}\n")


def statusline(flat_rate_eur: float = 100.0) -> None:
    """Einzeilige Ausgabe für Claude Code Statusline."""
    s = _month_stats()
    color, _ = _ampel(s["cost_eur"], flat_rate_eur)
    pct = min(100, int(s["cost_eur"] / flat_rate_eur * 100)) if flat_rate_eur > 0 else 0
    print(f"{color} {s['cost_eur']:.2f}€/{flat_rate_eur:.0f}€ ({pct}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="API-Budget-Ampel")
    sub = p.add_subparsers(dest="cmd")

    log_p = sub.add_parser("log", help="Session-Verbrauch loggen")
    log_p.add_argument("--input", type=int, required=True, help="Input-Token-Anzahl")
    log_p.add_argument("--output", type=int, required=True, help="Output-Token-Anzahl")
    log_p.add_argument("--model", choices=list(PRICES), default="sonnet")

    stat_p = sub.add_parser("status", help="Monatsauswertung anzeigen")
    stat_p.add_argument("--flat", type=float, default=100.0, metavar="EUR",
                        help="Flat-Rate in Euro (Standard: 100)")

    sl_p = sub.add_parser("statusline", help="Einzeiler für Claude Code Statusline")
    sl_p.add_argument("--flat", type=float, default=100.0, metavar="EUR")

    args = p.parse_args()

    if args.cmd == "log":
        log_session(args.input, args.output, args.model)
    elif args.cmd == "status":
        status(args.flat)
    elif args.cmd == "statusline":
        statusline(args.flat)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
