"""Integration tests: ModelRouter wired into ingest() and mayring_categorize()."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_source(path: str, source_type: str = "repo_file"):
    from src.memory.schema import Source
    return Source(
        source_id=f"test:{path}",
        source_type=source_type,
        repo="test",
        path=path,
        branch="main",
        commit="abc",
        content_hash="",
        captured_at="2026-04-17T00:00:00+00:00",
    )


def _make_router(vision_available: bool = True, vision_model: str = "qwen2.5vl:3b"):
    router = MagicMock()
    router.is_available.side_effect = lambda task: task == "vision" and vision_available
    router.resolve.side_effect = lambda task: vision_model if task == "vision" else "llama3.1:8b"
    return router


class TestIngestVisualPipeline:
    """ingest() auto-detects image extensions and routes to vision model."""

    def test_image_extension_triggers_vision_when_available(self, tmp_path):
        """PNG path → ingest_image() called when vision is available."""
        from src.memory.ingest import ingest
        from src.memory.store import init_memory_db

        conn = init_memory_db(tmp_path / "mem.db")
        router = _make_router(vision_available=True)
        source = _make_source("docs/arch.png")

        fake_image_result = {
            "source_id": source.source_id,
            "chunk_ids": ["chk_fake"],
            "indexed": False,
            "deduped": 0,
            "superseded": 0,
        }

        with patch("src.memory.ingest.ingest_image", return_value=fake_image_result) as mock_img:
            result = ingest(
                source=source,
                content="<binary>",
                conn=conn,
                chroma_collection=None,
                ollama_url="http://localhost:11434",
                model="",
                router=router,
            )

        mock_img.assert_called_once()
        call_kwargs = mock_img.call_args
        assert call_kwargs.kwargs.get("vision_model") == "qwen2.5vl:3b"
        assert result["source_id"] == source.source_id

    def test_image_extension_fallback_when_vision_unavailable(self, tmp_path):
        """PNG path → falls through to generic pipeline when vision not available."""
        from src.memory.ingest import ingest
        from src.memory.store import init_memory_db

        conn = init_memory_db(tmp_path / "mem.db")
        router = _make_router(vision_available=False)
        source = _make_source("docs/arch.png")

        with (
            patch("src.memory.ingest.ingest_image") as mock_img,
            patch("src.analysis.context._embed_texts", return_value=[[0.1, 0.2, 0.3]]),
        ):
            result = ingest(
                source=source,
                content="placeholder content for image",
                conn=conn,
                chroma_collection=None,
                ollama_url="http://localhost:11434",
                model="",
                router=router,
            )

        # Vision not available → ingest_image NOT called via router path
        mock_img.assert_not_called()
        assert result.get("source_id") == source.source_id

    def test_non_image_extension_not_affected(self, tmp_path):
        """Python file → vision not triggered even with router."""
        from src.memory.ingest import ingest
        from src.memory.store import init_memory_db

        conn = init_memory_db(tmp_path / "mem.db")
        router = _make_router(vision_available=True)
        source = _make_source("src/app.py")

        with (
            patch("src.memory.ingest.ingest_image") as mock_img,
            patch("src.analysis.context._embed_texts", return_value=[[0.1, 0.2, 0.3]]),
            patch("src.analysis.analyzer._ollama_generate", return_value="domain"),
        ):
            result = ingest(
                source=source,
                content="def main(): pass",
                conn=conn,
                chroma_collection=None,
                ollama_url="http://localhost:11434",
                model="llama3.1:8b",
                router=router,
            )

        mock_img.assert_not_called()
        assert result.get("source_id") == source.source_id

    def test_ingest_without_router_unchanged_behavior(self, tmp_path):
        """ingest() with router=None behaves exactly as before."""
        from src.memory.ingest import ingest
        from src.memory.store import init_memory_db

        conn = init_memory_db(tmp_path / "mem.db")
        source = _make_source("src/app.py")

        with patch("src.analysis.context._embed_texts", return_value=[[0.1, 0.2, 0.3]]):
            result = ingest(
                source=source,
                content="def main(): pass",
                conn=conn,
                chroma_collection=None,
                ollama_url="http://localhost:11434",
                model="",
                router=None,
            )

        assert result.get("source_id") == source.source_id

    def test_image_source_type_set_on_image_extension(self, tmp_path):
        """PNG source with source_type='repo_file' → source_type corrected to 'image'."""
        from src.memory.ingest import ingest
        from src.memory.store import init_memory_db

        conn = init_memory_db(tmp_path / "mem.db")
        router = _make_router(vision_available=True)
        source = _make_source("docs/diagram.png", source_type="repo_file")

        captured_source = {}

        def fake_ingest_image(source, image_path, conn, chroma_collection,
                              ollama_url, model, vision_model, workspace_id):
            captured_source["source_type"] = source.source_type
            return {
                "source_id": source.source_id,
                "chunk_ids": [],
                "indexed": False,
                "deduped": 0,
                "superseded": 0,
            }

        with patch("src.memory.ingest.ingest_image", side_effect=fake_ingest_image):
            ingest(
                source=source,
                content="<binary>",
                conn=conn,
                chroma_collection=None,
                ollama_url="http://localhost:11434",
                model="",
                router=router,
            )

        assert captured_source.get("source_type") == "image"


class TestMayringCategorizeRouter:
    """mayring_categorize() uses router for model selection."""

    def test_router_model_used_when_no_explicit_model(self):
        """Router provides model when model='' passed."""
        from src.memory.schema import Chunk
        from src.memory.ingest import mayring_categorize

        router = MagicMock()
        router.is_available.return_value = True
        router.resolve.return_value = "mayringqwen:latest"

        chunk = Chunk(
            chunk_id="chk_test_001",
            source_id="repo:test:app.py",
            chunk_level="file",
            ordinal=0,
            text="def authenticate(user): return validate(user)",
            text_hash="abc",
            dedup_key="abc",
            created_at="2026-04-17T00:00:00+00:00",
        )

        captured_models: list[str] = []

        def fake_generate(prompt, ollama_url, model, label, *, system_prompt=None):
            captured_models.append(model)
            return "auth, security"

        with patch("src.analysis.analyzer._ollama_generate", side_effect=fake_generate):
            mayring_categorize(
                chunks=[chunk],
                ollama_url="http://localhost:11434",
                model="",
                mode="hybrid",
                codebook="code",
                source_type="repo_file",
                router=router,
            )

        router.is_available.assert_called_with("mayring_code")
        router.resolve.assert_called_with("mayring_code")
        assert captured_models == ["mayringqwen:latest"]

    def test_explicit_model_not_overridden_by_router(self):
        """Explicit model wins over router."""
        from src.memory.schema import Chunk
        from src.memory.ingest import mayring_categorize

        router = MagicMock()
        router.is_available.return_value = True
        router.resolve.return_value = "mayringqwen:latest"

        chunk = Chunk(
            chunk_id="chk_test_002",
            source_id="repo:test:app.py",
            chunk_level="file",
            ordinal=0,
            text="class UserService: pass",
            text_hash="def",
            dedup_key="def",
            created_at="2026-04-17T00:00:00+00:00",
        )

        captured_models: list[str] = []

        def fake_generate(prompt, ollama_url, model, label, *, system_prompt=None):
            captured_models.append(model)
            return "domain"

        with patch("src.analysis.analyzer._ollama_generate", side_effect=fake_generate):
            mayring_categorize(
                chunks=[chunk],
                ollama_url="http://localhost:11434",
                model="explicit-model:latest",
                mode="hybrid",
                codebook="code",
                source_type="repo_file",
                router=router,
            )

        # Router.resolve should NOT have been called (explicit model wins)
        router.resolve.assert_not_called()
        assert captured_models == ["explicit-model:latest"]

    def test_no_router_passes_through_unchanged(self):
        """router=None → existing behavior, no router calls."""
        from src.memory.schema import Chunk
        from src.memory.ingest import mayring_categorize

        chunk = Chunk(
            chunk_id="chk_test_003",
            source_id="repo:test:app.py",
            chunk_level="file",
            ordinal=0,
            text="SELECT * FROM users",
            text_hash="ghi",
            dedup_key="ghi",
            created_at="2026-04-17T00:00:00+00:00",
        )

        captured_models: list[str] = []

        def fake_generate(prompt, ollama_url, model, label, *, system_prompt=None):
            captured_models.append(model)
            return "data_access"

        with patch("src.analysis.analyzer._ollama_generate", side_effect=fake_generate):
            mayring_categorize(
                chunks=[chunk],
                ollama_url="http://localhost:11434",
                model="some-model:latest",
                router=None,
            )

        assert captured_models == ["some-model:latest"]
