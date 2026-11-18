import numpy as np
import pytest

from clausetriage.threshold_strategies import (
    DEFAULT_THRESHOLD,
    global_threshold,
    per_category_f1,
    pooled_threshold,
    rare_fallback,
    recall_floor,
    shrunk_thresholds,
)

CATS = ["common", "rare"]


@pytest.fixture
def toy() -> tuple[np.ndarray, np.ndarray]:
    """Eight calibration chunks: 'common' has 4 positives, 'rare' has 1."""
    y_true = np.array([[1, 0], [1, 0], [1, 0], [1, 1],
                       [0, 0], [0, 0], [0, 0], [0, 0]])
    y_score = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.95],
                        [0.4, 0.1], [0.3, 0.2], [0.2, 0.9], [0.1, 0.1]])
    return y_true, y_score


# ----- per_category_f1 (current method) -----
def test_per_category_f1_picks_each_category_best_cutoff(toy):
    """Checks 'common' separates perfectly at 0.6 and 'rare' peaks at its one positive (0.95)."""
    y_true, y_score = toy

    thresholds = per_category_f1(y_true, y_score, CATS)

    assert thresholds["common"] == pytest.approx(0.6)
    assert thresholds["rare"] == pytest.approx(0.95)


def test_per_category_f1_defaults_when_no_positives():
    """Checks a category without calibration positives gets the 0.5 fallback."""
    thresholds = per_category_f1(np.array([[1, 0], [0, 0]]), np.array([[0.9, 0.7], [0.1, 0.2]]), CATS)

    assert thresholds["rare"] == DEFAULT_THRESHOLD


# ----- pooled / global -----
def test_pooled_threshold_maximizes_micro_f1_hand_computed(toy):
    """Checks the pooled cutoff over all 16 cells.

    At 0.6: 5 TP (common 0.9/0.8/0.7/0.6, rare 0.95), 1 FP (rare 0.9), 0 FN -> F1 = 10/11, beating
    0.7 (F1 0.8) and every other cutoff.
    """
    y_true, y_score = toy

    assert pooled_threshold(y_true, y_score) == pytest.approx(0.6)
    assert list(global_threshold(y_true, y_score, CATS).values()) == pytest.approx([0.6, 0.6])


# ----- shrinkage -----
def test_shrunk_thresholds_weights_by_calibration_support(toy):
    """Checks w = n / (n + prior): 'rare' (n=1, prior=1) lands halfway to the pooled 0.6."""
    y_true, y_score = toy

    thresholds = shrunk_thresholds(y_true, y_score, CATS, prior=1)

    assert thresholds["rare"] == pytest.approx(0.5 * 0.95 + 0.5 * 0.6)
    assert thresholds["common"] == pytest.approx(0.6)  # own and pooled agree


def test_shrunk_thresholds_with_zero_prior_equals_current_method(toy):
    """Checks prior 0 gives back the per-category cutoffs."""
    y_true, y_score = toy

    assert shrunk_thresholds(y_true, y_score, CATS, prior=0) == pytest.approx(per_category_f1(y_true, y_score, CATS))


# ----- rare fallback -----
def test_rare_fallback_only_replaces_categories_below_support(toy):
    """Checks 'rare' (1 positive) takes the pooled cutoff and 'common' (4) keeps its own."""
    y_true, y_score = toy

    thresholds = rare_fallback(y_true, y_score, CATS, rare_support=2)

    assert thresholds["rare"] == pytest.approx(0.6)
    assert thresholds["common"] == pytest.approx(0.6)


# ----- recall floor -----
def test_recall_floor_lowers_cutoff_to_reach_recall():
    """Checks a cutoff catching 1 of 4 positives drops to the 2nd-highest positive for 50% recall."""
    y_true = np.array([[1], [1], [1], [1], [0]])
    y_score = np.array([[0.9], [0.5], [0.4], [0.3], [0.85]])

    thresholds = recall_floor(y_true, y_score, ["A"], floor=0.5)

    assert per_category_f1(y_true, y_score, ["A"])["A"] == pytest.approx(0.3)  # F1-optimal keeps all
    assert thresholds["A"] == pytest.approx(0.3)  # already above the floor, so unchanged


def test_recall_floor_never_raises_the_cutoff(toy):
    """Checks recall-floor cutoffs are at or below the F1-optimal ones."""
    y_true, y_score = toy

    floor = recall_floor(y_true, y_score, CATS, floor=0.75)
    own = per_category_f1(y_true, y_score, CATS)

    assert all(floor[c] <= own[c] + 1e-12 for c in CATS)


def test_recall_floor_lowers_a_conservative_cutoff():
    """Checks a high F1-optimal cutoff (1 of 3 positives) drops to 0.3 so 2 of 3 are caught.

    F1 at 0.95 = 2/4 = 0.50; at 0.3 = 4/12 (2 TP, 7 FP); at 0.25 = 6/13. So F1 picks 0.95,
    and a 0.6 recall floor needs ceil(1.8) = 2 positives -> the 2nd-highest positive, 0.3.
    """
    y_true = np.array([[1], [1], [1], [0], [0], [0], [0], [0], [0], [0]])
    y_score = np.array([[0.95], [0.3], [0.25], [0.9], [0.8], [0.7], [0.6], [0.5], [0.4], [0.35]])

    own = per_category_f1(y_true, y_score, ["A"])["A"]
    floor = recall_floor(y_true, y_score, ["A"], floor=0.6)["A"]

    assert own == pytest.approx(0.95)
    assert floor == pytest.approx(0.3)
