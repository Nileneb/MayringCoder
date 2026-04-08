"""Tests for src/gpu_metrics.py."""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.gpu_metrics import (
    start_monitoring,
    stop_monitoring,
    parse_metrics,
    format_summary,
)


class TestStartMonitoring:
    def test_returns_none_when_nvidia_smi_missing(self, tmp_path):
        with patch("subprocess.Popen", side_effect=FileNotFoundError("nvidia-smi not found")):
            proc = start_monitoring(tmp_path / "metrics.csv")
        assert proc is None

    def test_returns_popen_on_success(self, tmp_path):
        mock_proc = MagicMock()
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            proc = start_monitoring(tmp_path / "metrics.csv", interval_s=2)
        assert proc is mock_proc
        # Verify nvidia-smi was called
        call_args = mock_popen.call_args[0][0]
        assert "nvidia-smi" in call_args
        assert "-l" in call_args
        assert "2" in call_args

    def test_creates_parent_directory(self, tmp_path):
        new_dir = tmp_path / "nested" / "dir"
        assert not new_dir.exists()
        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            start_monitoring(new_dir / "metrics.csv")
        assert new_dir.exists()

    def test_returns_none_on_generic_exception(self, tmp_path):
        with patch("subprocess.Popen", side_effect=OSError("permission denied")):
            proc = start_monitoring(tmp_path / "metrics.csv")
        assert proc is None


class TestStopMonitoring:
    def test_none_proc_returns_none(self):
        result = stop_monitoring(None)
        assert result is None

    def test_terminates_process(self):
        mock_proc = MagicMock()
        stop_monitoring(mock_proc)
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()

    def test_kills_if_terminate_fails(self):
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = Exception("timeout")
        stop_monitoring(mock_proc)
        mock_proc.kill.assert_called_once()


class TestParseMetrics:
    def _write_csv(self, tmp_path, rows: list[tuple]) -> Path:
        """Write nvidia-smi style CSV rows."""
        csv_path = tmp_path / "metrics.csv"
        with csv_path.open("w") as f:
            for row in rows:
                f.write(",".join(str(v) for v in row) + "\n")
        return csv_path

    def test_missing_file_returns_empty_dict(self, tmp_path):
        result = parse_metrics(tmp_path / "nonexistent.csv")
        assert result == {}

    def test_empty_file_returns_empty_dict(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        result = parse_metrics(csv_path)
        assert result == {}

    def test_single_row_parsed_correctly(self, tmp_path):
        # timestamp, gpu_util, mem_used, mem_total, power, temp
        csv_path = self._write_csv(tmp_path, [
            ("2026-04-08 10:00:00", 75, 4096, 8192, 120.5, 65),
        ])
        result = parse_metrics(csv_path)
        assert result["peak_vram_mb"] == 4096.0
        assert result["avg_gpu_util_pct"] == 75.0
        assert result["avg_power_w"] == pytest.approx(120.5)
        assert result["peak_temp_c"] == 65.0
        assert result["sample_count"] == 1

    def test_multiple_rows_peak_and_avg(self, tmp_path):
        csv_path = self._write_csv(tmp_path, [
            ("2026-04-08 10:00:00", 50, 3000, 8192, 100.0, 60),
            ("2026-04-08 10:00:01", 80, 5000, 8192, 150.0, 70),
            ("2026-04-08 10:00:02", 60, 4000, 8192, 120.0, 65),
        ])
        result = parse_metrics(csv_path)
        assert result["peak_vram_mb"] == 5000.0
        assert result["avg_gpu_util_pct"] == pytest.approx((50 + 80 + 60) / 3)
        assert result["peak_temp_c"] == 70.0
        assert result["sample_count"] == 3

    def test_invalid_rows_skipped(self, tmp_path):
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text(
            "2026-04-08 10:00:00,75,4096,8192,120.5,65\n"
            "INVALID LINE\n"
            "2026-04-08 10:00:01,80,5000,8192,150.0,70\n"
        )
        result = parse_metrics(csv_path)
        assert result["sample_count"] == 2


class TestFormatSummary:
    def test_empty_dict_returns_fallback_message(self):
        msg = format_summary({})
        assert "nicht verfügbar" in msg or "keine Daten" in msg

    def test_formats_metrics_correctly(self):
        metrics = {
            "peak_vram_mb": 4096.0,
            "avg_gpu_util_pct": 75.3,
            "avg_power_w": 120.5,
            "peak_temp_c": 65.0,
            "sample_count": 10,
        }
        msg = format_summary(metrics)
        assert "4096" in msg
        assert "75.3" in msg
        assert "120.5" in msg
        assert "65" in msg
