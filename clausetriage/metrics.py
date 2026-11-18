import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

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
    """Compute per-category average precision; categories without positives are NaN."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.ndim != 2 or y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must be matching two-dimensional arrays")
    scores = np.full(y_true.shape[1], np.nan, dtype=float)
    for index in range(y_true.shape[1]):
        if np.any(y_true[:, index] == 1):
            scores[index] = average_precision_score(y_true[:, index], y_prob[:, index])
    return scores


def add_aupr_column(report: pd.DataFrame, y_true: np.ndarray, y_score: np.ndarray,
                    categories: list[str]) -> pd.DataFrame:
    """Attach per-category AUPR to a metrics report, matched by category name.

    `compute_base_metrics` sorts rows by support, so assigning the raw AUPR array (which is in
    `categories` order) row by row would give each category another category's AUPR. Mapping by
    name keeps rows correct whatever order the report is in.
    """
    aupr_by_category = pd.Series(compute_aupr(y_true, y_score), index=categories)
    report = report.copy()
    report["aupr"] = report["category"].map(aupr_by_category)
    return report


def compute_transformer_metrics(eval_pred) -> dict[str, float]:
    """Trainer hook returning macro AUPR from multilabel logits and labels."""
    logits, labels = eval_pred
    logits = np.asarray(logits, dtype=float)
    # Numerically stable sigmoid without importing PyTorch in baseline commands.
    probs = np.empty_like(logits)
    positive = logits >= 0
    probs[positive] = 1 / (1 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probs[~positive] = exp_logits / (1 + exp_logits)
    per_category = compute_aupr(np.asarray(labels).astype(int), probs)
    return {"macro_aupr": float(np.nanmean(per_category))}