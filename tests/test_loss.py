import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from src.labels.categories import CATEGORIES
from src.models.encoder.loss import WeightedBCETrainer, compute_pos_weight


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


class FixedLogitsModel(nn.Module):
    """Stand-in model that ignores its inputs and returns fixed logits, so no weights are downloaded."""

    def __init__(self, logits: torch.Tensor) -> None:
        """Store the logits to return, plus a dummy parameter so the Trainer sees a trainable model."""
        super().__init__()
        self.logits = logits
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None) -> SimpleNamespace:
        """Return an output object with the fixed logits; raises TypeError if labels are passed in."""
        return SimpleNamespace(logits=self.logits)


def make_trainer(tmp_path, logits: list[list[float]], pos_weight: list[float]) -> WeightedBCETrainer:
    """Build a WeightedBCETrainer around a FixedLogitsModel, on CPU with logging disabled."""
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True)
    model = FixedLogitsModel(torch.tensor(logits))
    return WeightedBCETrainer(pos_weight=torch.tensor(pos_weight), model=model, args=args)


def make_inputs(labels: list[list[float]]) -> dict:
    """Build a Trainer-style inputs dict for a batch of the given float labels."""
    batch_size = len(labels)
    return {
        "input_ids": torch.zeros(batch_size, 3, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        "labels": torch.tensor(labels),
    }


# ----- WeightedBCETrainer: loss values -----
def test_weighted_bce_trainer_zero_logits_with_weight_three_gives_golden_loss(tmp_path):
    """Checks logits 0 (p=0.5), labels [1,0], weights [3,1] -> mean(3*ln2, ln2) = 2*ln2."""
    trainer = make_trainer(tmp_path, logits=[[0.0, 0.0]], pos_weight=[3.0, 1.0])

    loss = trainer.compute_loss(trainer.model, make_inputs([[1.0, 0.0]]))

    assert loss.item() == pytest.approx(2 * math.log(2))


def test_weighted_bce_trainer_unit_weights_give_plain_bce(tmp_path):
    """Checks pos_weight of all ones reduces to ordinary BCE: logits 0 -> ln2 per element."""
    trainer = make_trainer(tmp_path, logits=[[0.0, 0.0]], pos_weight=[1.0, 1.0])

    loss = trainer.compute_loss(trainer.model, make_inputs([[1.0, 0.0]]))

    assert loss.item() == pytest.approx(math.log(2))


def test_weighted_bce_trainer_nonzero_logits_match_hand_computed_loss(tmp_path):
    """Checks logits [2,-1], labels [1,1], weights [2,5] against the hand formula w*log(1+e^-x)."""
    # Both labels are positive, so loss = mean(2*log(1+e^-2), 5*log(1+e^1))
    expected = (2 * math.log1p(math.exp(-2.0)) + 5 * math.log1p(math.exp(1.0))) / 2
    trainer = make_trainer(tmp_path, logits=[[2.0, -1.0]], pos_weight=[2.0, 5.0])

    loss = trainer.compute_loss(trainer.model, make_inputs([[1.0, 1.0]]))

    assert loss.item() == pytest.approx(expected)


def test_weighted_bce_trainer_larger_weight_makes_missed_positive_cost_more(tmp_path):
    """Checks a heavier pos_weight raises the loss when a positive is predicted as negative."""
    # Logit -3 for a true positive is a confident miss
    light = make_trainer(tmp_path, logits=[[-3.0]], pos_weight=[1.0])
    heavy = make_trainer(tmp_path, logits=[[-3.0]], pos_weight=[10.0])
    inputs = make_inputs([[1.0]])

    assert heavy.compute_loss(heavy.model, dict(inputs)) > light.compute_loss(light.model, dict(inputs))


def test_weighted_bce_trainer_weight_does_not_change_loss_for_a_negative_label(tmp_path):
    """Checks pos_weight only scales the positive term, so an all-negative batch is unaffected."""
    light = make_trainer(tmp_path, logits=[[1.5, -0.5]], pos_weight=[1.0, 1.0])
    heavy = make_trainer(tmp_path, logits=[[1.5, -0.5]], pos_weight=[40.0, 40.0])
    inputs = make_inputs([[0.0, 0.0]])

    assert heavy.compute_loss(heavy.model, dict(inputs)).item() == pytest.approx(
        light.compute_loss(light.model, dict(inputs)).item()
    )


# ----- WeightedBCETrainer: Trainer interface -----
def test_weighted_bce_trainer_return_outputs_gives_loss_and_model_outputs(tmp_path):
    """Checks return_outputs=True returns (loss, outputs) with the model's logits."""
    trainer = make_trainer(tmp_path, logits=[[0.5, -0.5]], pos_weight=[1.0, 1.0])

    loss, outputs = trainer.compute_loss(trainer.model, make_inputs([[1.0, 0.0]]), return_outputs=True)

    assert loss.ndim == 0
    assert torch.equal(outputs.logits, torch.tensor([[0.5, -0.5]]))


def test_weighted_bce_trainer_default_returns_scalar_loss_tensor(tmp_path):
    """Checks the default return is a single scalar loss tensor, not a tuple."""
    trainer = make_trainer(tmp_path, logits=[[0.0, 0.0]], pos_weight=[1.0, 1.0])

    loss = trainer.compute_loss(trainer.model, make_inputs([[1.0, 0.0]]))

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_weighted_bce_trainer_removes_labels_from_the_model_inputs(tmp_path):
    """Checks labels are popped so the model never computes its own unweighted loss."""
    trainer = make_trainer(tmp_path, logits=[[0.0]], pos_weight=[1.0])
    inputs = make_inputs([[1.0]])

    # FixedLogitsModel.forward would raise TypeError if "labels" reached it
    trainer.compute_loss(trainer.model, inputs)

    assert "labels" not in inputs


def test_weighted_bce_trainer_accepts_num_items_in_batch_and_extra_kwargs(tmp_path):
    """Checks the arguments newer Trainer versions pass are accepted and do not change the loss."""
    trainer = make_trainer(tmp_path, logits=[[0.0, 0.0]], pos_weight=[3.0, 1.0])

    loss = trainer.compute_loss(trainer.model, make_inputs([[1.0, 0.0]]),
                                num_items_in_batch=torch.tensor(2), unused_option=True)

    assert loss.item() == pytest.approx(2 * math.log(2))


def test_weighted_bce_trainer_is_a_trainer_and_stores_pos_weight(tmp_path):
    """Checks the subclass is a real Trainer and keeps the pos_weight it was given."""
    trainer = make_trainer(tmp_path, logits=[[0.0, 0.0]], pos_weight=[3.0, 1.0])

    assert isinstance(trainer, Trainer)
    assert trainer.pos_weight.tolist() == [3.0, 1.0]
