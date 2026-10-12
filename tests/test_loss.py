import numpy as np
import pytest
import torch
from src.labels.categories import CATEGORIES
from src.models.encoder.loss import compute_pos_weight


def matrix_with_positives(num_rows: int, *pos_counts: int) -> np.ndarray:
    """Build a (num_rows, len(pos_counts)) label matrix whose column j has pos_counts[j] ones."""
    labels = np.zeros((num_rows, len(pos_counts)), dtype=int)
    for column, count in enumerate(pos_counts):
        labels[:count, column] = 1
    return labels


# ----- compute_pos_weight: golden values -----
def test_compute_pos_weight_hand_computed_matrix_gives_golden_weights():
    """Checks a 4x3 matrix: 1 pos / 3 neg -> 3.0 twice, and a column with no positives -> 1.0."""
    labels = np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])

    result = compute_pos_weight(labels, cap=50.0)

    assert result.tolist() == [3.0, 3.0, 1.0]


@pytest.mark.parametrize("num_rows, pos_count, expected", [
    (8, 1, 7.0),    # 1 pos, 7 neg -> 7/1
    (8, 2, 3.0),    # 2 pos, 6 neg -> 6/2
    (8, 4, 1.0),    # 4 pos, 4 neg -> 4/4 (balanced)
    (10, 5, 1.0),   # 5 pos, 5 neg -> 5/5 (balanced)
    (5, 1, 4.0),    # 1 pos, 4 neg -> 4/1
])
def test_compute_pos_weight_single_column_gives_neg_over_pos(num_rows, pos_count, expected):
    """Checks the raw weight of one category is its negative count divided by its positive count."""
    result = compute_pos_weight(matrix_with_positives(num_rows, pos_count), cap=100.0)

    assert result.tolist() == [expected]


# ----- compute_pos_weight: cap and floor -----
@pytest.mark.parametrize("cap, expected", [
    (5.0, 5.0),     # cap below the raw weight of 9 clamps it
    (9.0, 9.0),     # cap exactly equal to the raw weight leaves it unchanged
    (50.0, 9.0),    # cap above the raw weight leaves it unchanged
    (1.0, 1.0),     # smallest allowed cap flattens every weight to 1
])
def test_compute_pos_weight_cap_limits_raw_weight(cap, expected):
    """Checks weights are clipped at the cap but never raised to it."""
    # 1 positive out of 10 chunks -> raw weight 9/1 = 9.0
    result = compute_pos_weight(matrix_with_positives(10, 1), cap=cap)

    assert result.tolist() == [expected]


def test_compute_pos_weight_category_in_every_chunk_is_floored_at_one():
    """Checks a category positive everywhere (raw weight 0) is raised to 1.0, never below."""
    result = compute_pos_weight(matrix_with_positives(4, 4), cap=50.0)

    assert result.tolist() == [1.0]


def test_compute_pos_weight_category_without_positives_gets_one_not_inf():
    """Checks a category with no positives gets 1.0 instead of infinity or NaN."""
    result = compute_pos_weight(matrix_with_positives(6, 0), cap=50.0)

    assert result.tolist() == [1.0]
    assert torch.isfinite(result).all()


def test_compute_pos_weight_mixed_columns_are_weighted_independently():
    """Checks each column gets its own weight, including a clipped one and a zero-positive one."""
    # 10 chunks; positives per column: 1 (raw 9, capped to 4), 5 (raw 1), 0 (filled 1), 2 (raw 4)
    result = compute_pos_weight(matrix_with_positives(10, 1, 5, 0, 2), cap=4.0)

    assert result.tolist() == [4.0, 1.0, 1.0, 4.0]


# ----- compute_pos_weight: output type and shape -----
def test_compute_pos_weight_returns_float32_tensor_with_one_weight_per_category():
    """Checks the result is a float32 torch tensor of shape (num_categories,)."""
    result = compute_pos_weight(np.zeros((5, len(CATEGORIES)), dtype=int), cap=50.0)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.shape == (len(CATEGORIES),) == (41,)


def test_compute_pos_weight_accepts_list_of_label_vectors_as_array():
    """Checks a matrix built the way train.py will build it, np.array of label lists, works."""
    label_lists = [[1, 0], [0, 0], [0, 0], [0, 1]]

    result = compute_pos_weight(np.array(label_lists), cap=50.0)

    assert result.tolist() == [3.0, 3.0]
