import numpy as np
import pytest

from scripts.evaluate_category_cue_boost import _adjust_logits, _match_matrix
from src.labels.categories import CATEGORIES


def test_category_cues_are_matched_to_their_own_category_and_as_whole_words():
    cues = {"governinglaw": ["law"], "parties": ["party"]}
    matches = _match_matrix(["The law governs. One party may assign.", "lawful conduct"], cues)

    assert matches.shape == (2, len(CATEGORIES))
    assert matches[0, CATEGORIES.index("Governing Law")]
    assert matches[0, CATEGORIES.index("Parties")]
    assert not matches[1, CATEGORIES.index("Governing Law")]


def test_zero_boost_preserves_sparse_scores_exactly():
    scores = np.asarray([[0.1, 0.8], [0.5, 0.3]], dtype=np.float32)
    matches = np.asarray([[True, False], [False, True]])

    assert np.array_equal(_adjust_logits(scores, matches, 0), scores)


def test_positive_boost_only_adjusts_matching_category_scores():
    scores = np.asarray([[0.2, 0.7]], dtype=np.float32)
    matches = np.asarray([[True, False]])
    adjusted = _adjust_logits(scores, matches, 1.0)

    assert adjusted[0, 0] > scores[0, 0]
    assert adjusted[0, 1] == scores[0, 1]


def test_adjustment_rejects_misaligned_matrices():
    with pytest.raises(ValueError, match="identical shape"):
        _adjust_logits(np.zeros((2, 2)), np.zeros((2, 1), dtype=bool), 1.0)
