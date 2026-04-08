"""GPU-Monitoring via nvidia-smi.

Misst VRAM, Auslastung, Wattverbrauch und Temperatur während Ingestion/Retrieval.
Graceful Fallback wenn nvidia-smi nicht verfügbar.

Verwendung:
    proc = start_monitoring("cache/gpu_metrics.csv", interval_s=1)
    # ... Arbeit ...
    csv_path = stop_monitoring(proc)
    summary = parse_metrics(csv_path)
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Any


# nvidia-smi CSV-Spalten in dieser Reihenfolge
_NVIDIA_SMI_QUERY = ",".join([
    "timestamp",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
])

_CSV_HEADERS = [
    "timestamp",
    "gpu_util_pct",
    "memory_used_mb",
    "memory_total_mb",
    "power_draw_w",
    "temperature_c",
]


def start_monitoring(output_path: str | Path, interval_s: int = 1) -> subprocess.Popen | None:
    """Startet nvidia-smi Monitoring als Hintergrundprozess.

    Args:
        output_path: Pfad zur CSV-Ausgabedatei
        interval_s: Messintervall in Sekunden (Standard: 1)

    Returns:
        Popen-Objekt oder None wenn nvidia-smi nicht verfügbar.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={_NVIDIA_SMI_QUERY}",
                "--format=csv,noheader,nounits",
                f"-l", str(interval_s),
                f"-f", str(output_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except FileNotFoundError:
        # nvidia-smi nicht installiert
        return None
    except Exception:
        return None


def stop_monitoring(proc: subprocess.Popen | None) -> Path | None:
    """Stoppt den Monitoring-Prozess.

    Args:
        proc: Popen-Objekt aus start_monitoring()

    Returns:
        Pfad zur CSV-Datei oder None wenn kein Prozess.
    """
    if proc is None:
        return None
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    return None  # Pfad wurde beim Start übergeben, nicht gespeichert intern


def parse_metrics(csv_path: str | Path) -> dict[str, Any]:
    """Liest nvidia-smi CSV und gibt zusammengefasste Metriken zurück.

    Returns:
        {
            peak_vram_mb: float,
            avg_gpu_util_pct: float,
            avg_power_w: float,
            peak_temp_c: float,
            sample_count: int,
        }
        Leeres Dict wenn Datei fehlt oder keine Daten.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}

    rows: list[dict] = []
    try:
        with csv_path.open(newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                try:
                    rows.append({
                        "gpu_util_pct": float(row[1]),
                        "memory_used_mb": float(row[2]),
                        "memory_total_mb": float(row[3]),
                        "power_draw_w": float(row[4]),
                        "temperature_c": float(row[5]),
                    })
                except (ValueError, IndexError):
                    continue
    except Exception:
        return {}

    if not rows:
        return {}

    return {
        "peak_vram_mb": max(r["memory_used_mb"] for r in rows),
        "avg_gpu_util_pct": sum(r["gpu_util_pct"] for r in rows) / len(rows),
        "avg_power_w": sum(r["power_draw_w"] for r in rows) / len(rows),
        "peak_temp_c": max(r["temperature_c"] for r in rows),
        "sample_count": len(rows),
    }


def format_summary(metrics: dict[str, Any]) -> str:
    """Gibt GPU-Metriken als lesbare Zeile zurück."""
    if not metrics:
        return "[GPU-Monitoring: keine Daten / nvidia-smi nicht verfügbar]"
    return (
        f"GPU: Peak VRAM {metrics['peak_vram_mb']:.0f} MB | "
        f"Avg Util {metrics['avg_gpu_util_pct']:.1f}% | "
        f"Avg Power {metrics['avg_power_w']:.1f} W | "
        f"Peak Temp {metrics['peak_temp_c']:.0f}°C "
        f"({metrics['sample_count']} Samples)"
    )
