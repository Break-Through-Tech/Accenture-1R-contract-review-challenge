import numpy as np
import pytest

from scripts.evaluate_tfidf_reranker_fusion import _fuse


def test_zero_weight_preserves_tfidf_probabilities():
    sparse = np.full((2, 41), 0.35, dtype=np.float32)
    reranker = np.zeros_like(sparse)
    reranker[0, 0] = 0.8

    assert np.allclose(_fuse(sparse, reranker, 0.0), sparse, atol=1e-6)


def test_non_candidate_scores_fall_back_exactly_to_sparse_model():
    sparse = np.full((2, 41), 0.35, dtype=np.float32)
    reranker = np.zeros_like(sparse)
    reranker[0, 0] = 0.8
    fused = _fuse(sparse, reranker, 1.0)

    assert fused[1, 0] == pytest.approx(sparse[1, 0])
    assert fused[0, 1] == pytest.approx(sparse[0, 1])
    assert fused[0, 0] == pytest.approx(reranker[0, 0])


def test_fusion_rejects_weight_outside_unit_interval():
    scores = np.full((1, 41), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="alpha"):
        _fuse(scores, scores, 1.1)
