"""Error analysis for multi-label clause classifiers, computed from saved scores.

Every function works on the arrays the experiment scripts already save
(`y_true`, `y_score` and per-category thresholds), so no model needs to be rerun.
Column i of every matrix means `categories[i]`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.metrics import compute_aupr

NO_LABEL = "(no gold label)"
SUPPORT_BUCKETS = [(0, 9, "rare (<10)"), (10, 29, "uncommon (10-29)"), (30, 10**9, "common (30+)")]


def _f1(tp: int, fp: int, fn: int) -> float:
    """F1 from counts, defined as 0.0 when there is nothing to score."""
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def apply_category_thresholds(y_score: np.ndarray, thresholds: dict[str, float],
                              categories: list[str]) -> np.ndarray:
    """Binarize scores with each category's own threshold, looked up by name."""
    cutoffs = np.asarray([thresholds[category] for category in categories], dtype=float)
    return (np.asarray(y_score) >= cutoffs).astype(np.int8)


def oracle_f1(y_true_column: np.ndarray, y_score_column: np.ndarray) -> float:
    """Best F1 any single threshold could reach on these labels (an upper bound, not a result).

    The gap between this and the F1 at the calibration threshold measures how much F1 is
    lost because the threshold chosen on calibration contracts does not transfer.
    """
    if not np.any(y_true_column == 1):
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true_column, y_score_column)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.max())


def support_bucket(support: int) -> str:
    """Name the support bucket a category falls in."""
    for low, high, name in SUPPORT_BUCKETS:
        if low <= support <= high:
            return name
    raise ValueError(f"support must be non-negative, got {support}")


def category_error_table(y_true: np.ndarray, y_score: np.ndarray, thresholds: dict[str, float],
                         categories: list[str]) -> pd.DataFrame:
    """Per-category error counts, F1, AUPR and threshold-transfer loss, in `categories` order.

    `fp_on_unlabeled` counts false positives on chunks that carry no gold label at all,
    i.e. boilerplate text the model mistook for a clause rather than a different clause.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = apply_category_thresholds(y_score, thresholds, categories)
    unlabeled = y_true.sum(axis=1) == 0
    aupr = compute_aupr(y_true, y_score)

    rows = []
    for index, category in enumerate(categories):
        # Confusion counts for this category
        gold, pred = y_true[:, index] == 1, y_pred[:, index] == 1
        tp, fp, fn = int(np.sum(gold & pred)), int(np.sum(~gold & pred)), int(np.sum(gold & ~pred))
        support = int(gold.sum())
        f1 = _f1(tp, fp, fn) if support else float("nan")
        best = oracle_f1(y_true[:, index], np.asarray(y_score)[:, index])
        rows.append({
            "category": category,
            "support": support,
            "support_bucket": support_bucket(support),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / support if support else float("nan"),
            "f1": f1,
            "aupr": float(aupr[index]),
            "threshold": float(thresholds[category]),
            "oracle_f1": best,
            "threshold_gap": best - f1 if support else float("nan"),
            "fp_on_unlabeled": int(np.sum(~gold & pred & unlabeled)),
        })
    return pd.DataFrame(rows)


def false_positive_sources(y_true: np.ndarray, y_pred: np.ndarray, categories: list[str]) -> pd.DataFrame:
    """For each predicted category, which gold labels its false-positive chunks actually had.

    A chunk with several gold labels counts once for each of them, so `share` is the
    fraction of that category's false-positive chunks carrying that gold label. Chunks
    with no gold label are reported as `NO_LABEL`.
    """
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    rows = []
    for index, category in enumerate(categories):
        # False-positive chunks for this category
        fp_rows = np.flatnonzero((y_pred[:, index] == 1) & (y_true[:, index] == 0))
        if fp_rows.size == 0:
            continue
        gold = y_true[fp_rows]
        # Gold labels those chunks really carried
        counts = {categories[j]: int(gold[:, j].sum()) for j in range(len(categories)) if gold[:, j].any()}
        counts[NO_LABEL] = int(np.sum(gold.sum(axis=1) == 0))
        for source, count in counts.items():
            if count:
                rows.append({"predicted_category": category, "gold_label": source,
                             "count": count, "share": count / fp_rows.size,
                             "category_false_positives": int(fp_rows.size)})
    columns = ["predicted_category", "gold_label", "count", "share", "category_false_positives"]
    return pd.DataFrame(rows, columns=columns)


def model_disagreement(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                       categories: list[str]) -> pd.DataFrame:
    """Per-category overlap of two models' hits and false alarms on the same chunks.

    `either_recall` is the recall if a positive counted when either model caught it; a large
    gap between it and each model's own recall means the two models make different mistakes.
    """
    y_true = np.asarray(y_true).astype(int)
    pred_a, pred_b = np.asarray(pred_a).astype(int), np.asarray(pred_b).astype(int)
    rows = []
    for index, category in enumerate(categories):
        # Split positives and negatives by which model flagged them
        gold = y_true[:, index] == 1
        a, b = pred_a[:, index] == 1, pred_b[:, index] == 1
        support = int(gold.sum())
        rows.append({
            "category": category,
            "support": support,
            "both_caught": int(np.sum(gold & a & b)),
            "only_a_caught": int(np.sum(gold & a & ~b)),
            "only_b_caught": int(np.sum(gold & ~a & b)),
            "both_missed": int(np.sum(gold & ~a & ~b)),
            "fp_only_a": int(np.sum(~gold & a & ~b)),
            "fp_only_b": int(np.sum(~gold & ~a & b)),
            "fp_both": int(np.sum(~gold & a & b)),
            "either_recall": float(np.sum(gold & (a | b)) / support) if support else float("nan"),
        })
    return pd.DataFrame(rows)


def seed_stability(per_seed: pd.DataFrame, value: str = "f1") -> pd.DataFrame:
    """Mean, spread and range of one metric per (model, category) across seeds.

    `per_seed` needs columns `model`, `category`, `seed` and `value`.
    """
    grouped = per_seed.groupby(["model", "category"])[value]
    summary = grouped.agg(["mean", "std", "min", "max", "count"]).reset_index()
    summary = summary.rename(columns={"mean": f"mean_{value}", "std": f"std_{value}",
                                      "min": f"min_{value}", "max": f"max_{value}", "count": "n_seeds"})
    summary[f"range_{value}"] = summary[f"max_{value}"] - summary[f"min_{value}"]
    return summary
