"""Tests fuer die Visual-Model-Pipeline (Issue #91).

Abgedeckt:
- caption_image() fuer SVG (kein Ollama-Call)
- caption_image() fuer PNG (Ollama wird aufgerufen)
- Ollama-Fehler -> leerer String
- ingest_image() erzeugt Chunk mit chunk_level='image_caption'
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.agents.vision import caption_image
from src.memory.schema import Source
from src.memory.store import init_memory_db, _init_schema


# ---------------------------------------------------------------------------
# caption_image — SVG
# ---------------------------------------------------------------------------


def test_caption_svg_returns_raw_text_no_ollama(tmp_path: Path) -> None:
    svg_path = tmp_path / "diagram.svg"
    svg_content = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="10"/></svg>'
    svg_path.write_text(svg_content, encoding="utf-8")

    with patch("src.ollama_client.generate") as mock_gen:
        result = caption_image(svg_path, ollama_url="http://localhost:11434")

    mock_gen.assert_not_called()
    assert result == svg_content


# ---------------------------------------------------------------------------
# caption_image — PNG via Ollama
# ---------------------------------------------------------------------------


def test_caption_png_calls_ollama(tmp_path: Path) -> None:
    img_path = tmp_path / "chart.png"
    img = Image.new("RGB", (32, 32), color=(0, 100, 200))
    img.save(img_path, format="PNG")

    with patch("src.ollama_client.generate", return_value="A blue square.") as mock_gen:
        result = caption_image(img_path, ollama_url="http://localhost:11434", model="qwen2.5vl:3b")

    mock_gen.assert_called_once()
    assert result == "A blue square."


def test_caption_png_ollama_error_returns_empty(tmp_path: Path) -> None:
    img_path = tmp_path / "broken.png"
    img = Image.new("RGB", (16, 16), color=(0, 0, 0))
    img.save(img_path, format="PNG")

    with patch("src.ollama_client.generate", side_effect=Exception("Connection refused")):
        result = caption_image(img_path, ollama_url="http://localhost:11434")

    assert result == ""


# ---------------------------------------------------------------------------
# ingest_image — chunk mit chunk_level='image_caption'
# ---------------------------------------------------------------------------


def test_ingest_image_produces_image_caption_chunk(tmp_path: Path) -> None:
    from src.memory.ingestion.image import ingest_image

    img_path = tmp_path / "arch.png"
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(img_path, format="PNG")

    db_path = tmp_path / "test.db"
    conn = init_memory_db(db_path)

    source = Source(
        source_id="repo:test/project:arch.png",
        source_type="repo_file",
        repo="test/project",
        path="arch.png",
    )

    fake_caption = "A red architecture diagram."

    with (
        patch("src.agents.vision.caption_image", return_value=fake_caption),
        patch("src.analysis.context._embed_texts", return_value=[[0.1] * 768]),
        patch("src.memory.ingestion.core.resolve_dedup") as mock_dedup,
    ):
        mock_dedup.return_value = (MagicMock(), False)

        result = ingest_image(
            source=source,
            image_path=img_path,
            conn=conn,
            chroma_collection=None,
            ollama_url="http://localhost:11434",
            model="nomic-embed-text",
            vision_model="qwen2.5vl:3b",
            workspace_id="test-ws",
        )

    from src.memory.store import get_chunk

    assert len(result["chunk_ids"]) == 1
    chunk_id = result["chunk_ids"][0]
    stored = get_chunk(conn, chunk_id)
    assert stored is not None
    assert stored.chunk_level == "image_caption"
    assert result["deduped"] == 0


def test_ingest_image_svg_no_ollama_call(tmp_path: Path) -> None:
    from src.memory.ingestion.image import ingest_image

    svg_path = tmp_path / "flow.svg"
    svg_content = '<svg xmlns="http://www.w3.org/2000/svg"><text>Flow</text></svg>'
    svg_path.write_text(svg_content, encoding="utf-8")

    db_path = tmp_path / "test.db"
    conn = init_memory_db(db_path)

    source = Source(
        source_id="repo:test/project:flow.svg",
        source_type="repo_file",
        repo="test/project",
        path="flow.svg",
    )

    with (
        patch("src.ollama_client.generate") as mock_gen,
        patch("src.analysis.context._embed_texts", return_value=[[0.1] * 768]),
        patch("src.memory.ingestion.core.resolve_dedup") as mock_dedup,
    ):
        mock_dedup.return_value = (MagicMock(), False)

        result = ingest_image(
            source=source,
            image_path=svg_path,
            conn=conn,
            chroma_collection=None,
            ollama_url="http://localhost:11434",
            model="nomic-embed-text",
            vision_model="qwen2.5vl:3b",
            workspace_id="test-ws",
        )

    mock_gen.assert_not_called()
    assert len(result["chunk_ids"]) == 1


def test_ingest_image_fallback_caption_on_empty_response(tmp_path: Path) -> None:
    from src.memory.ingestion.image import ingest_image

    img_path = tmp_path / "empty_caption.png"
    img = Image.new("RGB", (16, 16), color=(128, 128, 128))
    img.save(img_path, format="PNG")

    db_path = tmp_path / "test.db"
    conn = init_memory_db(db_path)

    source = Source(
        source_id="repo:test/project:empty_caption.png",
        source_type="repo_file",
        repo="test/project",
        path="empty_caption.png",
    )

    with (
        patch("src.agents.vision.caption_image", return_value=""),
        patch("src.analysis.context._embed_texts", return_value=[[0.1] * 768]),
        patch("src.memory.ingestion.core.resolve_dedup") as mock_dedup,
    ):
        mock_dedup.return_value = (MagicMock(), False)

        result = ingest_image(
            source=source,
            image_path=img_path,
            conn=conn,
            chroma_collection=None,
            ollama_url="http://localhost:11434",
            model="nomic-embed-text",
            workspace_id="test-ws",
        )

    assert len(result["chunk_ids"]) == 1
