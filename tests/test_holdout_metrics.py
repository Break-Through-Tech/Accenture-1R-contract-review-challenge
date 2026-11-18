import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from src.holdout_metrics import (
    bootstrap_replicates, category_counts, compare_with_reference, contract_groups, contract_presence,
    detection_table, f1_from_counts, macro_aupr, macro_f1, per_category_aupr, percentile_interval,
    predict_at_thresholds, spearman, support_bucket, support_status,
)

# Two categories, four chunks; hand-computed below
Y_TRUE = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
SCORES = np.array([[0.9, 0.6], [0.4, 0.7], [0.2, 0.1], [0.1, 0.8]])
THRESHOLDS = np.array([0.5, 0.6])


# ----- Thresholding and counts -----
def test_predict_at_thresholds_counts_a_probability_equal_to_the_threshold_as_detected():
    """A probability exactly at the threshold is a detection (>=), the rule ClauseModelBundle.predict uses."""
    assert predict_at_thresholds(np.array([[0.5, 0.59]]), np.array([0.5, 0.6])).tolist() == [[1, 0]]


def test_predict_at_thresholds_rejects_a_threshold_vector_of_the_wrong_length():
    """One threshold per column is required."""
    with pytest.raises(ValueError, match="one value per"):
        predict_at_thresholds(np.zeros((2, 3)), np.zeros(2))


def test_category_counts_and_f1_match_hand_computed_values():
    """Column 0: TP1 FP0 FN1 -> F1 2/3; column 1: TP1 FP2 FN1 -> F1 2/5."""
    predictions = predict_at_thresholds(SCORES, THRESHOLDS)  # [[1,1],[0,1],[0,0],[0,1]]
    tp, fp, fn = category_counts(Y_TRUE, predictions)
    assert (tp.tolist(), fp.tolist(), fn.tolist()) == ([1, 1], [0, 2], [1, 1])
    assert f1_from_counts(tp, fp, fn) == pytest.approx([2 / 3, 2 / 5])


def test_f1_from_counts_is_zero_when_a_category_has_no_positives_and_no_predictions():
    """0/0 counts as 0, as sklearn's zero_division=0 does."""
    assert f1_from_counts(np.array([0]), np.array([0]), np.array([0])).tolist() == [0.0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_f1_from_counts_matches_sklearn_on_random_labels(seed):
    """Per-category F1 equals precision_recall_fscore_support(average=None, zero_division=0)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = rng.integers(0, 2, (50, 6)), rng.integers(0, 2, (50, 6))
    y_true[:, 0] = 0  # one category without positives
    expected = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)[2]
    assert f1_from_counts(*category_counts(y_true, y_pred)) == pytest.approx(expected)


# ----- AUPR and macro conventions -----
def test_per_category_aupr_is_nan_for_a_category_without_positives():
    """A column with no positives has undefined AUPR; the others match sklearn."""
    y_true = np.array([[1, 0], [0, 0], [1, 0]])
    scores = np.array([[0.9, 0.1], [0.5, 0.2], [0.3, 0.3]])
    result = per_category_aupr(y_true, scores)
    assert result[0] == pytest.approx(average_precision_score(y_true[:, 0], scores[:, 0]))
    assert np.isnan(result[1])


def test_per_category_aupr_golden_value():
    """Scores 0.9(+), 0.5(-), 0.3(+): AP = 1/2 * 1 + 1/2 * 2/3 = 5/6."""
    assert per_category_aupr(np.array([[1], [0], [1]]), np.array([[0.9], [0.5], [0.3]]))[0] == pytest.approx(5 / 6)


def test_macro_f1_keeps_zero_categories_and_macro_aupr_skips_undefined_ones():
    """Macro-F1 averages over every category; macro-AUPR only over categories with positives."""
    assert macro_f1(np.array([1.0, 0.0, 0.5])) == pytest.approx(0.5)
    assert macro_aupr(np.array([1.0, np.nan, 0.5])) == pytest.approx(0.75)
    assert np.isnan(macro_aupr(np.array([np.nan, np.nan])))


def test_detection_table_golden_rows_in_category_order():
    """Rows keep the given category order and hold the hand-computed counts, precision and recall."""
    table = detection_table(Y_TRUE, SCORES, THRESHOLDS, categories=["A", "B"])
    assert table["category"].tolist() == ["A", "B"]
    assert table[["support", "tp", "fp", "fn"]].values.tolist() == [[2, 1, 0, 1], [2, 1, 2, 1]]
    assert table["precision"].tolist() == pytest.approx([1.0, 1 / 3])
    assert table["recall"].tolist() == pytest.approx([0.5, 0.5])
    assert table["f1"].tolist() == pytest.approx([2 / 3, 2 / 5])


# ----- Contract grouping -----
def test_contract_groups_keep_first_appearance_order():
    """Contracts are listed in the order their first chunk appears, with all their row indices."""
    order, groups = contract_groups(["b", "b", "a", "c", "a"])
    assert order == ["b", "a", "c"]
    assert [g.tolist() for g in groups] == [[0, 1], [2, 4], [3]]


def test_contract_presence_takes_any_label_and_the_highest_score():
    """Contract gold = any labeled chunk; contract score = max chunk probability."""
    gold, score = contract_presence(Y_TRUE, SCORES, [np.array([0, 1]), np.array([2, 3])])
    assert gold.tolist() == [[1, 1], [0, 1]]
    assert score.tolist() == [[0.9, 0.7], [0.2, 0.8]]


# ----- Bootstrap -----
def test_bootstrap_replicates_uses_the_documented_draws():
    """Replicate i sees default_rng(seed).integers(0, n, size=n), the project's resampling rule."""
    seen = bootstrap_replicates(5, 3, 123, lambda selected: {"first": float(selected[0]), "all": selected.copy()})
    rng = np.random.default_rng(123)
    expected = [rng.integers(0, 5, size=5) for _ in range(3)]
    assert seen["all"].tolist() == [draw.tolist() for draw in expected]
    assert seen["first"].tolist() == [draw[0] for draw in expected]


def test_bootstrap_replicates_is_deterministic_for_a_seed():
    """The same seed gives identical replicates."""
    def statistic(selected):
        """Mean of the selected indices."""
        return {"mean": float(np.mean(selected))}
    assert np.array_equal(bootstrap_replicates(10, 20, 7, statistic)["mean"],
                          bootstrap_replicates(10, 20, 7, statistic)["mean"])


@pytest.mark.parametrize("n_contracts, replicates", [(0, 5), (5, 0)])
def test_bootstrap_replicates_rejects_empty_input(n_contracts, replicates):
    """Zero contracts or zero replicates is an error."""
    with pytest.raises(ValueError):
        bootstrap_replicates(n_contracts, replicates, 1, lambda selected: {})


def test_percentile_interval_golden_values_and_skips_nan():
    """0..100 gives 2.5 and 97.5; NaN replicates are dropped and counted out."""
    result = percentile_interval(np.append(np.arange(101.0), np.nan))
    assert (result["lower_95"], result["upper_95"], result["replicates_defined"]) == pytest.approx((2.5, 97.5, 101))
    assert percentile_interval(np.array([np.nan]))["replicates_defined"] == 0


# ----- Spearman -----
@pytest.mark.parametrize("a, b, expected", [([1, 2, 3], [1, 2, 3], 1.0), ([1, 2, 3], [3, 2, 1], -1.0),
                                            ([1, 2, 3, 4], [1, 1, 2, 3], 0.9486832980505138)])
def test_spearman_golden_values(a, b, expected):
    """Average ranks for ties: [1,1,2,3] ranks to [1.5,1.5,3,4]."""
    assert spearman(a, b) == pytest.approx(expected)


def test_spearman_matches_pandas_on_tied_data():
    """Equals pandas' DataFrame.corr(method='spearman'), which scripts/evaluate_triage.py uses."""
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 5, 40), rng.integers(0, 5, 40)
    expected = pd.DataFrame({"a": a, "b": b}).corr(method="spearman").iloc[0, 1]
    assert spearman(a, b) == pytest.approx(expected)


def test_spearman_is_nan_for_a_constant_input():
    """A constant vector has no ranking to correlate."""
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


# ----- Pre-registered reading against development -----
@pytest.mark.parametrize("point, lower, upper, verdict, position", [
    (0.50, 0.45, 0.55, "consistent_with_development", "within_development_range"),
    (0.44, 0.40, 0.48, "lower_than_development", "below_every_development_split"),
    (0.49, 0.47, 0.499, "lower_than_development", "within_development_range"),
    (0.56, 0.51, 0.60, "higher_than_development", "above_every_development_split"),
    (0.504, 0.45, 0.504, "consistent_with_development", "within_development_range"),
])
def test_compare_with_reference_wording_rule(point, lower, upper, verdict, position):
    """Consistent when the development mean (0.504) is inside the interval; position against the range 0.47-0.53."""
    result = compare_with_reference(point, lower, upper, reference=0.504, reference_min=0.47, reference_max=0.53)
    assert (result["verdict"], result["position"]) == (verdict, position)
    assert result["gap_to_development_mean"] == pytest.approx(point - 0.504)


# ----- Support labels -----
@pytest.mark.parametrize("support, status, bucket", [(0, "unscorable", "rare"), (9, "insufficient support", "rare"),
                                                     (10, "reported", "uncommon"), (29, "reported", "uncommon"),
                                                     (30, "reported", "common")])
def test_support_status_and_bucket_cut_offs(support, status, bucket):
    """0 is unscorable, under 10 is insufficient support; buckets are <10, 10-29, 30+."""
    assert (support_status(support), support_bucket(support)) == (status, bucket)
