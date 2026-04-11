"""Tests for src/image_ingest.py."""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.image_ingest import (
    discover_images,
    run_image_ingest,
    _IMAGE_EXTENSIONS,
)


class TestDiscoverImages:
    def test_finds_png_jpg_svg(self, tmp_path):
        (tmp_path / "diagram.png").write_bytes(b"x" * 100)
        (tmp_path / "photo.jpg").write_bytes(b"x" * 200)
        (tmp_path / "icon.svg").write_text("<svg/>")
        (tmp_path / "readme.md").write_text("text")

        found = discover_images(tmp_path, max_file_bytes=1024, max_images=50)
        names = {p.name for p in found}
        assert names == {"diagram.png", "photo.jpg", "icon.svg"}

    def test_skips_files_over_size_limit(self, tmp_path):
        (tmp_path / "big.png").write_bytes(b"x" * 6_000_000)  # 6 MB
        (tmp_path / "small.png").write_bytes(b"x" * 100)

        found = discover_images(tmp_path, max_file_bytes=5_000_000, max_images=50)
        assert len(found) == 1
        assert found[0].name == "small.png"

    def test_respects_max_images_limit(self, tmp_path):
        for i in range(10):
            (tmp_path / f"img{i}.png").write_bytes(b"x" * 10)

        found = discover_images(tmp_path, max_file_bytes=1024 * 1024, max_images=3)
        assert len(found) == 3

    def test_returns_empty_for_no_images(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hi')")
        found = discover_images(tmp_path, max_file_bytes=1024 * 1024, max_images=50)
        assert found == []


class TestRunImageIngest:
    def test_clones_repo_and_ingests_images(self, tmp_path):
        fake_clone_dir = tmp_path / "clone"
        fake_clone_dir.mkdir()
        (fake_clone_dir / "docs").mkdir()
        (fake_clone_dir / "docs" / "erd.png").write_bytes(b"PNG" * 10)

        mock_ingest_image = MagicMock(return_value={"chunk_ids": ["c1"], "deduped": 0})
        mock_conn = MagicMock()
        mock_chroma = MagicMock()

        with (
            patch("src.image_ingest.subprocess.run") as mock_run,
            patch("src.image_ingest.tempfile.mkdtemp", return_value=str(fake_clone_dir.parent)),
            patch("src.image_ingest.shutil.rmtree") as mock_rmtree,
            patch("src.image_ingest.ingest_image", mock_ingest_image),
            patch("src.image_ingest.init_memory_db", return_value=mock_conn),
            patch("src.image_ingest.get_or_create_chroma_collection", return_value=mock_chroma),
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = run_image_ingest(
                repo_url="https://github.com/Nileneb/app.linn.games",
                ollama_url="http://localhost:11434",
                vision_model="qwen2.5vl:3b",
                max_images=50,
                max_file_bytes=5 * 1024 * 1024,
            )

        assert result["images_found"] >= 0
        mock_rmtree.assert_called_once()  # cleanup always runs

    def test_cleanup_runs_even_on_error(self, tmp_path):
        with (
            patch("src.image_ingest.subprocess.run", side_effect=RuntimeError("clone failed")),
            patch("src.image_ingest.tempfile.mkdtemp", return_value=str(tmp_path)),
            patch("src.image_ingest.shutil.rmtree") as mock_rmtree,
        ):
            with pytest.raises(RuntimeError):
                run_image_ingest(
                    repo_url="https://github.com/Nileneb/app.linn.games",
                    ollama_url="http://localhost:11434",
                )

        mock_rmtree.assert_called_once_with(str(tmp_path), ignore_errors=True)

    def test_returns_correct_counts(self, tmp_path):
        fake_clone_dir = tmp_path / "clone"
        fake_clone_dir.mkdir()
        (fake_clone_dir / "a.png").write_bytes(b"x" * 100)
        (fake_clone_dir / "b.svg").write_text("<svg/>")

        mock_ingest_image = MagicMock(side_effect=[
            {"chunk_ids": ["c1"], "deduped": 0},   # a.png: ingested
            {"chunk_ids": [], "deduped": 1},        # b.svg: deduped
        ])

        with (
            patch("src.image_ingest.subprocess.run", return_value=MagicMock(returncode=0)),
            patch("src.image_ingest.tempfile.mkdtemp", return_value=str(tmp_path)),
            patch("src.image_ingest.shutil.rmtree"),
            patch("src.image_ingest.ingest_image", mock_ingest_image),
            patch("src.image_ingest.init_memory_db", return_value=MagicMock()),
            patch("src.image_ingest.get_or_create_chroma_collection", return_value=MagicMock()),
        ):
            result = run_image_ingest(
                repo_url="https://github.com/Nileneb/app.linn.games",
                ollama_url="http://localhost:11434",
            )

        assert result["images_found"] == 2
        assert result["images_captioned"] == 1
        assert result["images_skipped"] == 1
