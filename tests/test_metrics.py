import numpy as np
import pytest
from transformers import EvalPrediction
from src.metrics import compute_aupr, compute_transformer_metrics


def column(values: list[float]) -> np.ndarray:
    """Turn a flat list into a single column (n, 1) matrix."""
    return np.array(values).reshape(-1, 1)


def make_eval_pred(logits: list[list[float]], labels: list[list[int]]) -> EvalPrediction:
    """Build the float32 EvalPrediction the Trainer passes to compute_metrics."""
    return EvalPrediction(predictions=np.array(logits, dtype=np.float32),
                          label_ids=np.array(labels, dtype=np.float32))


# ----- compute_aupr: hand-computed golden values -----
def test_compute_aupr_two_positives_at_ranks_one_and_three_gives_five_sixths():
    """Checks AP = mean of precision at each positive = (1/1 + 2/3) / 2 = 5/6."""
    y_true = column([1, 0, 1, 0])
    y_prob = column([0.9, 0.8, 0.7, 0.1])

    assert compute_aupr(y_true, y_prob)[0] == pytest.approx(5 / 6)


@pytest.mark.parametrize("y_true, y_prob, expected", [
    ([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], 1.0),        # every positive above every negative
    ([0, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], 1 / 2),      # one positive at rank 2 -> precision 1/2
    ([0, 0, 1], [0.9, 0.8, 0.1], 1 / 3),              # one positive at rank 3 -> precision 1/3
    ([1, 1, 1], [0.5, 0.4, 0.3], 1.0),                # every chunk positive
])
def test_compute_aupr_single_category_matches_hand_computed_value(y_true, y_prob, expected):
    """Checks AUPR for one category against values worked out by hand from the ranking."""
    assert compute_aupr(column(y_true), column(y_prob))[0] == pytest.approx(expected)


# ----- compute_aupr: shape, order and unscorable categories -----
def test_compute_aupr_category_without_positives_is_nan_not_zero():
    """Checks a category with no positives is NaN, since it cannot be scored."""
    result = compute_aupr(column([0, 0, 0]), column([0.9, 0.5, 0.1]))

    assert np.isnan(result[0])


def test_compute_aupr_multiple_columns_keep_column_order_and_score_independently():
    """Checks each output entry matches its own column: golden, perfect, and no positives."""
    y_true = np.array([[1, 1, 0],
                       [0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]])
    y_prob = np.array([[0.9, 0.9, 0.5],
                       [0.8, 0.8, 0.5],
                       [0.7, 0.2, 0.5],
                       [0.1, 0.1, 0.5]])

    result = compute_aupr(y_true, y_prob)

    assert result.shape == (3,)
    assert result[0] == pytest.approx(5 / 6)
    assert result[1] == pytest.approx(1.0)
    assert np.isnan(result[2])


def test_compute_aupr_unscorable_column_does_not_affect_other_columns():
    """Checks skipping a column with no positives leaves the neighboring columns' scores unchanged."""
    scorable = compute_aupr(column([1, 0, 1, 0]), column([0.9, 0.8, 0.7, 0.1]))[0]
    y_true = np.array([[0, 1], [0, 0], [0, 1], [0, 0]])
    y_prob = np.array([[0.5, 0.9], [0.5, 0.8], [0.5, 0.7], [0.5, 0.1]])

    assert compute_aupr(y_true, y_prob)[1] == pytest.approx(scorable)


def test_compute_aupr_output_has_one_float_entry_per_column():
    """Checks the result is a float array whose length equals the number of columns."""
    result = compute_aupr(np.zeros((5, 41), dtype=int), np.full((5, 41), 0.5))

    assert result.shape == (41,)
    assert result.dtype == np.float64


# ----- compute_transformer_metrics: Trainer hook -----
def test_compute_transformer_metrics_logits_give_golden_macro_aupr():
    """Checks logits ranking the golden example give macro_aupr 5/6; the empty column is ignored."""
    # Column 0 ranks like probs [0.9, 0.8, 0.7, 0.1]; column 1 has no positives
    logits = [[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [-2.0, 0.0]]
    labels = [[1, 0], [0, 0], [1, 0], [0, 0]]

    result = compute_transformer_metrics(make_eval_pred(logits, labels))

    assert result["macro_aupr"] == pytest.approx(5 / 6)


def test_compute_transformer_metrics_averages_over_scorable_categories():
    """Checks the macro average is the plain mean of per-category AUPR: (5/6 + 1) / 2 = 11/12."""
    logits = [[3.0, 2.0], [2.0, 1.0], [1.0, -1.0], [-2.0, -2.0]]
    labels = [[1, 1], [0, 1], [1, 0], [0, 0]]

    result = compute_transformer_metrics(make_eval_pred(logits, labels))

    assert result["macro_aupr"] == pytest.approx(11 / 12)


def test_compute_transformer_metrics_monotone_rescaling_of_logits_does_not_change_score():
    """Checks AUPR depends only on ranking, so scaling the logits leaves macro_aupr unchanged."""
    logits = np.array([[3.0], [2.0], [1.0], [-2.0]])
    labels = [[1], [0], [1], [0]]

    original = compute_transformer_metrics(make_eval_pred(logits.tolist(), labels))
    scaled = compute_transformer_metrics(make_eval_pred((logits * 3).tolist(), labels))

    assert scaled["macro_aupr"] == pytest.approx(original["macro_aupr"])


def test_compute_transformer_metrics_returns_plain_float_under_macro_aupr_key():
    """Checks the dict has only the macro_aupr key, holding a Python float for JSON logging."""
    result = compute_transformer_metrics(make_eval_pred([[1.0], [-1.0]], [[1], [0]]))

    assert list(result.keys()) == ["macro_aupr"]
    assert type(result["macro_aupr"]) is float


def test_compute_transformer_metrics_no_positives_anywhere_gives_nan_with_warning():
    """Checks an eval set with no positives yields NaN (with a RuntimeWarning), not a fake score."""
    eval_pred = make_eval_pred([[1.0, 0.0], [-1.0, 0.0]], [[0, 0], [0, 0]])

    with pytest.warns(RuntimeWarning):
        result = compute_transformer_metrics(eval_pred)

    assert np.isnan(result["macro_aupr"])
