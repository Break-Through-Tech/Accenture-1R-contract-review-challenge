# NOTE: Encoder metrics were added here, so consider splitting into baseline/encoder metrics.py modules
# The baseline now needs torch to import this, which would add a few seconds of overhead per baseline run

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from transformers import EvalPrediction
import torch


# Computes precision, recall, f1, and support for each category sorted by support
def compute_base_metrics(y_true, y_pred, categories):
    # Computes metrics
    precision, recall, fl, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Returns df with metrics sorted by support
    report_df = pd.DataFrame({
        "category": categories,
        "precision": precision.round(5),
        "recall": recall.round(5),
        "f1": fl.round(5),
        "support": support,
    })

    return report_df.sort_values("support", ascending=False).reset_index(drop=True)


def compute_aupr(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """
    Compute the area under the precision-recall curve (average precision) per category.
    y_true is a 0/1 matrix and y_prob the matching probabilities, both (num_chunks,
    num_categories). A category with no positives is NaN, not 0, since it cannot be scored.
    """
    # Start every category as NaN so unscorable ones stay NaN
    output = np.full(y_true.shape[1], np.nan)

    # Score each category that has at least one positive
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        output[i] = average_precision_score(y_true[:, i], y_prob[:, i])
    return output


def compute_transformer_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """
    Trainer compute_metrics hook: macro-average AUPR over the categories in the eval set.
    eval_pred holds (logits, labels); logits are converted to probabilities with a sigmoid.
    Categories with no positives are ignored, and the result is NaN if none can be scored.
    Returns {"macro_aupr": value}, the key used to select the best checkpoint.
    """
    # Unpack the eval output, turning logits into probabilities and labels into 0/1 ints
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    labels = labels.astype(int)

    # Average the per-category AUPR, skipping NaN categories
    macro_aupr = float(np.nanmean(compute_aupr(labels, probs)))
    return {"macro_aupr": macro_aupr}