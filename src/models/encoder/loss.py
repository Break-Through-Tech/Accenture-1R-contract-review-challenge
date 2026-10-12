import numpy as np
from torch import Tensor, tensor, float32


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
