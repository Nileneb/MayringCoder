"""Tests for src/benchmark_retrieval.py — mrr() and recall_at_k()."""
import pytest

from src.benchmark_retrieval import mrr, recall_at_k


class TestMrr:
    def test_empty_input_returns_zero(self):
        assert mrr([], []) == 0.0

    def test_perfect_first_hit(self):
        ranked = [["issue/7", "issue/12"]]
        relevant = [["issue/7"]]
        assert mrr(ranked, relevant) == 1.0

    def test_hit_at_rank_2(self):
        ranked = [["issue/1", "issue/7"]]
        relevant = [["issue/7"]]
        assert mrr(ranked, relevant) == pytest.approx(0.5)

    def test_hit_at_rank_3(self):
        ranked = [["issue/1", "issue/2", "issue/7"]]
        relevant = [["issue/7"]]
        assert mrr(ranked, relevant) == pytest.approx(1 / 3)

    def test_no_hit_returns_zero(self):
        ranked = [["issue/1", "issue/2"]]
        relevant = [["issue/99"]]
        assert mrr(ranked, relevant) == 0.0

    def test_multiple_queries_averaged(self):
        ranked = [
            ["issue/7", "issue/1"],   # hit at rank 1 → RR=1.0
            ["issue/1", "issue/7"],   # hit at rank 2 → RR=0.5
        ]
        relevant = [["issue/7"], ["issue/7"]]
        assert mrr(ranked, relevant) == pytest.approx((1.0 + 0.5) / 2)

    def test_first_matching_relevant_path_counts(self):
        """If multiple relevant paths exist, first hit determines RR."""
        ranked = [["issue/1", "issue/23", "issue/7"]]
        relevant = [["issue/7", "issue/23"]]
        # issue/23 is at rank 2 → RR = 0.5
        assert mrr(ranked, relevant) == pytest.approx(0.5)


class TestRecallAtK:
    def test_empty_input_returns_zero(self):
        assert recall_at_k([], []) == 0.0

    def test_full_recall_when_all_relevant_in_top_k(self):
        ranked = [["issue/7", "issue/12", "issue/19"]]
        relevant = [["issue/7", "issue/12"]]
        assert recall_at_k(ranked, relevant, k=5) == 1.0

    def test_partial_recall(self):
        ranked = [["issue/7", "issue/1", "issue/2", "issue/3", "issue/4"]]
        relevant = [["issue/7", "issue/12"]]  # issue/12 not in top-5
        assert recall_at_k(ranked, relevant, k=5) == pytest.approx(0.5)

    def test_zero_recall_when_no_relevant_in_top_k(self):
        ranked = [["issue/1", "issue/2", "issue/3"]]
        relevant = [["issue/99"]]
        assert recall_at_k(ranked, relevant, k=5) == 0.0

    def test_k_cutoff_respected(self):
        """Result at rank 6 should not count when k=5."""
        ranked = [["issue/1", "issue/2", "issue/3", "issue/4", "issue/5", "issue/7"]]
        relevant = [["issue/7"]]
        assert recall_at_k(ranked, relevant, k=5) == 0.0

    def test_empty_relevant_returns_full_recall(self):
        """No ground truth → trivially perfect (nothing to miss)."""
        ranked = [["issue/1"]]
        relevant = [[]]
        assert recall_at_k(ranked, relevant, k=5) == 1.0

    def test_multiple_queries_averaged(self):
        ranked = [
            ["issue/7"],           # 1/1 = 1.0
            ["issue/1", "issue/2"],  # 0/1 = 0.0
        ]
        relevant = [["issue/7"], ["issue/99"]]
        assert recall_at_k(ranked, relevant, k=5) == pytest.approx(0.5)

    def test_default_k_is_5(self):
        ranked = [["issue/7", "issue/1", "issue/2", "issue/3", "issue/4", "issue/99"]]
        relevant = [["issue/99"]]
        # issue/99 is at rank 6 — beyond default k=5
        assert recall_at_k(ranked, relevant) == 0.0
