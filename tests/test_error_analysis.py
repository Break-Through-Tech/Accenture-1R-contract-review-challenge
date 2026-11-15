import numpy as np
import pandas as pd
import pytest

from src.error_analysis import (
    NO_LABEL,
    apply_category_thresholds,
    category_error_table,
    false_positive_sources,
    model_disagreement,
    oracle_f1,
    seed_stability,
    support_bucket,
)
from src.metrics import add_aupr_column, compute_aupr, compute_base_metrics

CATS = ["A", "B", "C"]


@pytest.fixture
def toy() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Five chunks, three categories; chunk 4 has no gold label."""
    y_true = np.array([[1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 0]])
    y_score = np.array([[0.9, 0.1, 0.1],
                        [0.4, 0.8, 0.1],
                        [0.7, 0.6, 0.1],
                        [0.1, 0.2, 0.9],
                        [0.6, 0.7, 0.2]])
    return y_true, y_score, {"A": 0.5, "B": 0.5, "C": 0.5}


# ----- add_aupr_column: regression test for the sorted-report misalignment -----
def test_add_aupr_column_matches_by_name_after_support_sort():
    """Checks AUPR lands on the right row after compute_base_metrics sorts by support."""
    y_true = np.array([[0, 1], [0, 1], [1, 1], [0, 0]])  # "A" support 1, "B" support 3
    y_score = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.9, 0.1]])
    report = compute_base_metrics(y_true, (y_score >= 0.5).astype(int), ["A", "B"])

    result = add_aupr_column(report, y_true, y_score, ["A", "B"])

    expected = dict(zip(["A", "B"], compute_aupr(y_true, y_score)))
    for _, row in result.iterrows():
        assert row["aupr"] == pytest.approx(expected[row["category"]])


def test_add_aupr_column_differs_from_positional_assignment_when_order_changes():
    """Checks the old positional assignment would have been wrong on this sorted report."""
    y_true = np.array([[0, 1], [0, 1], [1, 1], [0, 0]])  # second category has more support
    y_score = np.array([[0.9, 0.9], [0.8, 0.8], [0.2, 0.7], [0.1, 0.1]])
    report = compute_base_metrics(y_true, (y_score >= 0.5).astype(int), ["X", "Y"])

    assert list(report["category"]) == ["Y", "X"]  # sorted by support
    fixed = add_aupr_column(report, y_true, y_score, ["X", "Y"])
    assert fixed.set_index("category").loc["Y", "aupr"] == pytest.approx(1.0)
    assert fixed.set_index("category").loc["X", "aupr"] == pytest.approx(1 / 3)


# ----- thresholds and oracle F1 -----
def test_apply_category_thresholds_uses_each_category_cutoff():
    """Checks each column is binarized against its own threshold, inclusive."""
    scores = np.array([[0.5, 0.5], [0.3, 0.9]])

    result = apply_category_thresholds(scores, {"A": 0.5, "B": 0.95}, ["A", "B"])

    assert result.tolist() == [[1, 0], [0, 0]]


def test_oracle_f1_finds_best_cutoff_hand_computed():
    """Checks best F1 over cutoffs: cutting at 0.7 keeps 2 TP, 1 FP, 0 FN -> F1 = 4/5."""
    assert oracle_f1(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1])) == pytest.approx(0.8)


def test_oracle_f1_without_positives_is_nan():
    """Checks a category with no positives has no oracle F1."""
    assert np.isnan(oracle_f1(np.array([0, 0]), np.array([0.2, 0.8])))


@pytest.mark.parametrize("support, expected", [
    (0, "rare (<10)"), (9, "rare (<10)"), (10, "uncommon (10-29)"),
    (29, "uncommon (10-29)"), (30, "common (30+)"),
])
def test_support_bucket_boundaries(support, expected):
    """Checks bucket edges are inclusive on both ends."""
    assert support_bucket(support) == expected


# ----- category_error_table: hand-computed golden counts -----
def test_category_error_table_counts_match_hand_computed(toy):
    """Checks TP/FP/FN, F1 and unlabeled-chunk FPs per category on the toy data."""
    y_true, y_score, thresholds = toy

    table = category_error_table(y_true, y_score, thresholds, CATS).set_index("category")

    # A: predicted on chunks 0, 2, 4; gold on 0, 1 -> TP 1, FP 2 (one unlabeled), FN 1
    assert table.loc["A", ["tp", "fp", "fn", "fp_on_unlabeled"]].tolist() == [1, 2, 1, 1]
    assert table.loc["A", "f1"] == pytest.approx(2 / 5)
    # B: predicted on 1, 2, 4; gold on 1, 2 -> TP 2, FP 1 (unlabeled), FN 0
    assert table.loc["B", ["tp", "fp", "fn", "fp_on_unlabeled"]].tolist() == [2, 1, 0, 1]
    assert table.loc["B", "f1"] == pytest.approx(4 / 5)
    # C is perfect
    assert table.loc["C", "f1"] == pytest.approx(1.0)
    assert table.loc["C", "threshold_gap"] == pytest.approx(0.0)


def test_category_error_table_threshold_gap_is_oracle_minus_actual(toy):
    """Checks threshold_gap equals oracle_f1 - f1 and is never negative."""
    y_true, y_score, thresholds = toy

    table = category_error_table(y_true, y_score, thresholds, CATS)

    assert np.allclose(table["threshold_gap"], table["oracle_f1"] - table["f1"])
    assert (table["threshold_gap"] >= -1e-12).all()


# ----- false_positive_sources -----
def test_false_positive_sources_attributes_gold_labels_and_unlabeled(toy):
    """Checks A's two false positives come from one B-labeled chunk and one unlabeled chunk."""
    y_true, y_score, thresholds = toy
    y_pred = apply_category_thresholds(y_score, thresholds, CATS)

    sources = false_positive_sources(y_true, y_pred, CATS)
    a_rows = sources[sources["predicted_category"] == "A"].set_index("gold_label")

    assert a_rows.loc["B", "count"] == 1
    assert a_rows.loc[NO_LABEL, "count"] == 1
    assert a_rows["share"].sum() == pytest.approx(1.0)
    assert "C" not in set(sources["predicted_category"])  # C has no false positives


# ----- model_disagreement -----
def test_model_disagreement_splits_hits_and_false_alarms():
    """Checks the four positive outcomes and the false alarms for one category."""
    y_true = np.array([[1], [1], [1], [1], [0], [0]])
    pred_a = np.array([[1], [1], [0], [0], [1], [0]])
    pred_b = np.array([[1], [0], [1], [0], [1], [1]])

    row = model_disagreement(y_true, pred_a, pred_b, ["A"]).iloc[0]

    assert row[["both_caught", "only_a_caught", "only_b_caught", "both_missed"]].tolist() == [1, 1, 1, 1]
    assert row[["fp_only_a", "fp_only_b", "fp_both"]].tolist() == [0, 1, 1]
    assert row["either_recall"] == pytest.approx(3 / 4)


# ----- seed_stability -----
def test_seed_stability_reports_mean_std_and_range():
    """Checks mean, sample std and range of F1 across three seeds."""
    per_seed = pd.DataFrame({"model": ["m"] * 3, "category": ["A"] * 3,
                             "seed": [1, 2, 3], "f1": [0.2, 0.4, 0.6]})

    row = seed_stability(per_seed).iloc[0]

    assert row["mean_f1"] == pytest.approx(0.4)
    assert row["std_f1"] == pytest.approx(0.2)
    assert row["range_f1"] == pytest.approx(0.4)
    assert row["n_seeds"] == 3
