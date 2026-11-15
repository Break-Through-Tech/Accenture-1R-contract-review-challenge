"""Alternative ways to set per-category decision thresholds from calibration scores.

The current method picks each category's F1-optimal cutoff on calibration contracts. With only
a few calibration positives that cutoff is noisy and transfers badly (see docs/error-analysis.md).
Every strategy here is fit on calibration rows only and returns `{category: threshold}`, so it
drops into `src.error_analysis.apply_category_thresholds` unchanged.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

DEFAULT_THRESHOLD = 0.5  # same fallback the current method uses for categories with no positives
SHRINKAGE_PRIOR = 10  # pseudo-positives behind the pooled cutoff; predeclared, not tuned
RARE_SUPPORT = 10  # calibration positives below which a category counts as rare
RECALL_FLOOR = 0.5  # minimum calibration recall the recall-floor strategy guarantees


def per_category_f1(y_true: np.ndarray, y_score: np.ndarray, categories: list[str]) -> dict[str, float]:
    """Current method: each category's F1-optimal calibration cutoff (0.5 if it has no positives).

    Mirrors `calibrate_thresholds` without importing PyTorch, so the comparison runs anywhere.
    """
    thresholds = {}
    for index, category in enumerate(categories):
        if not np.any(y_true[:, index] == 1):
            thresholds[category] = DEFAULT_THRESHOLD
            continue
        precision, recall, cutoffs = precision_recall_curve(y_true[:, index], y_score[:, index])
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        best = int(np.argmax(f1))
        thresholds[category] = float(cutoffs[best]) if best < len(cutoffs) else DEFAULT_THRESHOLD
    return thresholds


def pooled_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """One cutoff for every category: the one that maximizes pooled (micro) F1 on calibration."""
    precision, recall, cutoffs = precision_recall_curve(np.ravel(y_true), np.ravel(y_score))
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    best = int(np.argmax(f1[:-1]))  # the last PR point has no cutoff
    return float(cutoffs[best])


def global_threshold(y_true: np.ndarray, y_score: np.ndarray, categories: list[str]) -> dict[str, float]:
    """The pooled cutoff applied to every category."""
    cutoff = pooled_threshold(y_true, y_score)
    return {category: cutoff for category in categories}


def shrunk_thresholds(y_true: np.ndarray, y_score: np.ndarray, categories: list[str],
                      prior: float = SHRINKAGE_PRIOR) -> dict[str, float]:
    """Pull each category's cutoff toward the pooled one, more strongly when it has few positives.

    t = w * t_category + (1 - w) * t_pooled, with w = n / (n + prior) and n the number of
    calibration positives. A category with 10 positives and prior 10 lands halfway between.
    """
    own = per_category_f1(y_true, y_score, categories)
    pooled = pooled_threshold(y_true, y_score)
    support = np.asarray(y_true).sum(axis=0)
    thresholds = {}
    for index, category in enumerate(categories):
        weight = support[index] / (support[index] + prior)
        thresholds[category] = float(weight * own[category] + (1 - weight) * pooled)
    return thresholds


def rare_fallback(y_true: np.ndarray, y_score: np.ndarray, categories: list[str],
                  rare_support: int = RARE_SUPPORT) -> dict[str, float]:
    """Keep each category's own cutoff, but use the pooled one when it has < `rare_support` positives."""
    own = per_category_f1(y_true, y_score, categories)
    pooled = pooled_threshold(y_true, y_score)
    support = np.asarray(y_true).sum(axis=0)
    return {category: (pooled if support[index] < rare_support else own[category])
            for index, category in enumerate(categories)}


def recall_floor(y_true: np.ndarray, y_score: np.ndarray, categories: list[str],
                 floor: float = RECALL_FLOOR) -> dict[str, float]:
    """Lower a category's cutoff only when it would catch under `floor` of calibration positives.

    The cutoff becomes the highest one still reaching that recall, so it never rises above the
    F1-optimal cutoff. Categories with no calibration positives keep the default.
    """
    own = per_category_f1(y_true, y_score, categories)
    thresholds = {}
    for index, category in enumerate(categories):
        positives = np.sort(y_score[y_true[:, index] == 1, index])[::-1]
        if positives.size == 0:
            thresholds[category] = own[category]
            continue
        # Highest cutoff that keeps at least `floor` of positives at or above it
        needed = int(np.ceil(floor * positives.size))
        thresholds[category] = float(min(own[category], positives[needed - 1]))
    return thresholds


STRATEGIES = {
    "per_category_f1 (current)": per_category_f1,
    "global": global_threshold,
    "shrunk": shrunk_thresholds,
    "rare_fallback": rare_fallback,
    "recall_floor": recall_floor,
}
