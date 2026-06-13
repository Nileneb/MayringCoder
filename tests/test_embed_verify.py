import math
import pytest
from mayring_core.embed_verify import cosine, verify


def test_cosine_identical_is_one():
    v = [0.1, 0.2, 0.3, 0.4]
    assert cosine(v, v) == pytest.approx(1.0, abs=1e-9)


def test_cosine_orthogonal_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-9)


def test_fp_noise_still_agrees():
    a = [0.50, 0.50, 0.50, 0.50]
    b = [0.5000001, 0.4999999, 0.50, 0.5000002]
    assert verify(a, b, threshold=0.9999) is True


def test_real_divergence_fails():
    a = [1.0, 0.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0, 0.0]
    assert verify(a, b, threshold=0.9999) is False


def test_length_mismatch_is_divergence_not_crash():
    assert verify([1.0, 2.0], [1.0, 2.0, 3.0], threshold=0.9999) is False


def test_zero_vector_is_divergence_not_crash():
    assert verify([0.0, 0.0], [0.0, 0.0], threshold=0.9999) is False
