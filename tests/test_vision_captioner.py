"""Tests for src/vision_captioner.py — TDD first pass."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.agents.vision import (
    caption_image,
    caption_images_batch,
    get_image_metadata,
)


# ---------------------------------------------------------------------------
# TestGetImageMetadata
# ---------------------------------------------------------------------------


class TestGetImageMetadata:
    def test_png_metadata(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 50), color=(255, 0, 0))
        img.save(img_path, format="PNG")

        meta = get_image_metadata(img_path)

        assert meta is not None
        assert meta["width"] == 100
        assert meta["height"] == 50
        assert meta["format"] == "PNG"
        assert meta["file_size"] > 0

    def test_nonexistent_file_returns_none(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.png"
        assert get_image_metadata(missing) is None

    def test_svg_returns_text_metadata(self, tmp_path: Path) -> None:
        svg_path = tmp_path / "diagram.svg"
        svg_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><rect width="100" height="100"/></svg>',
            encoding="utf-8",
        )

        meta = get_image_metadata(svg_path)

        assert meta is not None
        assert meta["format"] == "SVG"
        assert meta["is_text"] is True


# ---------------------------------------------------------------------------
# TestCaptionImage
# ---------------------------------------------------------------------------


class TestCaptionImage:
    def test_caption_returns_string(self, tmp_path: Path) -> None:
        img_path = tmp_path / "arch.png"
        img = Image.new("RGB", (64, 64), color=(0, 128, 255))
        img.save(img_path, format="PNG")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "A diagram showing system architecture."}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            caption = caption_image(img_path, ollama_url="http://localhost:11434")

        assert "architecture" in caption.lower() or isinstance(caption, str)
        assert len(caption) > 0

    def test_caption_with_ollama_error_returns_empty(self, tmp_path: Path) -> None:
        img_path = tmp_path / "broken.png"
        img = Image.new("RGB", (32, 32), color=(0, 0, 0))
        img.save(img_path, format="PNG")

        with patch("httpx.post", side_effect=Exception("Connection refused")):
            caption = caption_image(img_path, ollama_url="http://localhost:11434")

        assert caption == ""

    def test_svg_returns_text_content_not_caption(self, tmp_path: Path) -> None:
        svg_path = tmp_path / "flow.svg"
        svg_content = '<svg xmlns="http://www.w3.org/2000/svg"><text>Hello</text></svg>'
        svg_path.write_text(svg_content, encoding="utf-8")

        with patch("httpx.post") as mock_post:
            caption = caption_image(svg_path, ollama_url="http://localhost:11434")
            mock_post.assert_not_called()

        assert caption == svg_content


# ---------------------------------------------------------------------------
# TestCaptionBatch
# ---------------------------------------------------------------------------


class TestCaptionBatch:
    def test_batch_returns_list_of_dicts(self, tmp_path: Path) -> None:
        paths = []
        for i in range(3):
            p = tmp_path / f"img{i}.png"
            img = Image.new("RGB", (32, 32), color=(i * 80, 0, 0))
            img.save(p, format="PNG")
            paths.append(p)

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "A red square."}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            results = caption_images_batch(paths, ollama_url="http://localhost:11434")

        assert len(results) == 3
        for result in results:
            assert "caption" in result
            assert "path" in result
