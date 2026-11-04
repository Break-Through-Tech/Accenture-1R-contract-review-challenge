import numpy as np
from torch import Tensor, tensor, float32
from torch.nn import BCEWithLogitsLoss, Module
from transformers import Trainer
from typing import Any


def compute_pos_weight(labels: np.ndarray, cap: float) -> Tensor:
    """
    Compute per-category pos_weight (neg / pos) for BCEWithLogitsLoss from a
    (num_chunks, num_categories) train label matrix, clipped to [1, cap].
    A category with no positives gets 1.0. Returns a float32 tensor of length num_categories.
    """
    # Count positive and negative chunks per category
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos

    # Divide only where a category has positives; the rest keep the pre-filled 1.0
    weights = np.divide(neg, pos, out=np.ones_like(pos, dtype=float), where=pos > 0)

    # Limit the weights to [1, cap] and convert for the loss function
    weights = np.clip(weights, 1.0, cap)
    return tensor(weights, dtype=float32)


class WeightedBCETrainer(Trainer):
    """
    Hugging Face Trainer that trains with class-weighted BCE for multi-label imbalance.
    The stock multi-label loss is plain BCE, so only compute_loss is overridden to apply
    pos_weight (one weight per category, see compute_pos_weight). All other arguments
    are passed through to Trainer unchanged.
    """

    def __init__(self, pos_weight: Tensor, *args, **kwargs) -> None:
        """Store the per-category pos_weight tensor, then build the Trainer as usual."""
        self.pos_weight = pos_weight
        super().__init__(*args, **kwargs)

    def compute_loss(self, model: Module, inputs: dict, return_outputs: bool = False,
                     num_items_in_batch: Tensor | int | None = None,
                     **kwargs) -> Tensor | tuple[Tensor, Any]:
        """
        Return the pos_weight-weighted BCE loss of the model's logits against the labels.
        Pops labels from inputs so the model does not compute its own unweighted loss.
        Returns (loss, outputs) when return_outputs is True. num_items_in_batch and
        extra kwargs are accepted for Trainer compatibility and unused.
        """
        # Separate the labels from the model inputs
        labels = inputs.pop("labels")

        # Run the model to get one raw score (logit) per category
        outputs = model(**inputs)
        logits = outputs.logits

        # Weighted BCE; the weights move to the logits' device (CPU or GPU)
        loss_fn = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss