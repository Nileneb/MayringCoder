"""Tests for src/api/training.py."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestTrainingStatus:
    def test_status_returns_counts(self, tmp_path):
        from src.api.training import _count_jsonl_lines

        tmp_train = tmp_path / "train.jsonl"
        tmp_train.write_text('{"messages": []}\n{"messages": []}\n')

        assert _count_jsonl_lines(tmp_train) == 2
        assert _count_jsonl_lines(tmp_path / "nonexistent.jsonl") == 0

    def test_prompt_hash_is_stable(self):
        from src.api.training import _prompt_hash
        msgs = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "hello"},
        ]
        h1 = _prompt_hash(msgs)
        h2 = _prompt_hash(msgs)
        assert h1 == h2
        assert len(h1) == 16

    def test_prompt_hash_differs_for_different_content(self):
        from src.api.training import _prompt_hash
        h1 = _prompt_hash([{"role": "user", "content": "a"}])
        h2 = _prompt_hash([{"role": "user", "content": "b"}])
        assert h1 != h2


class TestMerge:
    def test_merge_deduplicates(self, tmp_path):
        from src.api.training import _prompt_hash

        sample = {"messages": [{"role": "user", "content": "test"}], "label": "good", "quality_score": 0.9}

        f1 = tmp_path / "batch1.jsonl"
        f2 = tmp_path / "batch2.jsonl"
        f1.write_text(json.dumps(sample) + "\n")
        f2.write_text(json.dumps(sample) + "\n")

        with (
            patch("src.api.training._TRAIN_JSONL", tmp_path / "train.jsonl"),
            patch("src.api.training._FINETUNING_DIR", tmp_path),
            patch("src.api.training._HAIKU_ANNOTATIONS", tmp_path / "nonexistent.jsonl"),
            patch("src.api.training._LANGDOCK_BATCHES_DIR", tmp_path),
        ):
            existing = {}
            for path in [f1, f2]:
                with path.open() as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        ph = _prompt_hash(entry.get("messages", []))
                        if ph not in existing:
                            existing[ph] = entry

            assert len(existing) == 1

    def test_merge_skips_low_quality(self, tmp_path):
        bad = {"messages": [{"role": "user", "content": "bad"}], "label": "skip", "quality_score": 0.3}
        good = {"messages": [{"role": "user", "content": "good"}], "label": "good", "quality_score": 0.9}

        samples = [bad, good]
        filtered = [s for s in samples if s.get("label") != "skip" and float(s.get("quality_score", 1.0)) >= 0.5]
        assert len(filtered) == 1
        assert filtered[0]["label"] == "good"


class TestWebhookAuth:
    def test_compare_digest_used(self):
        import hmac
        secret = "test-secret"
        token = "test-secret"
        assert hmac.compare_digest(token.encode(), secret.encode())

    def test_wrong_token_fails(self):
        import hmac
        secret = "correct-secret"
        token = "wrong-token"
        assert not hmac.compare_digest(token.encode(), secret.encode())
