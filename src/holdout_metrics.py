"""
Metrics for the one-time sealed evaluation (docs/final-test-evaluation.md): detection quality at
fixed thresholds, contract-level presence, and a seeded contract bootstrap. Pure functions on
arrays; nothing here reads data or fits a model.

How everything comes together:
predict_at_thresholds(): probability >= its category's threshold, the rule ClauseModelBundle.predict uses
category_counts() / f1_from_counts(): per-category TP/FP/FN and F1 (0 when undefined, as zero_division=0)
per_category_aupr(): average precision per category, NaN for a category with no positives
macro_f1() / macro_aupr(): the project conventions (fixed 41 categories / supported categories only)
detection_table(): per-category precision, recall, F1, AUPR, support and counts, in CATEGORIES order
contract_groups() / contract_presence(): chunk rows per contract, and chunks collapsed to contracts
bootstrap_replicates() / jackknife_replicates(): seeded contract resampling, and leave-one-contract-out values
bca_interval() / percentile_interval(): BCa 95% bounds (the registered interval) and plain percentile bounds
spearman(): rank correlation with average ranks for ties (the same value pandas' DataFrame.corr gives)
compare_with_reference(): the pre-registered wording rule for a test result against development
support_status(): "unscorable", "insufficient support" or "reported" from test support
"""
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score

from src.labels.categories import CATEGORIES

# Support cut-offs shared with docs/error-analysis.md and scripts/compare_encoder_tfidf.py
RARE_SUPPORT, COMMON_SUPPORT = 10, 30


def predict_at_thresholds(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Return 0/1 predictions: probability >= the category's threshold (columns in CATEGORIES order)."""
    probabilities, thresholds = np.asarray(probabilities), np.asarray(thresholds, dtype=np.float64)
    if probabilities.ndim != 2 or thresholds.shape != (probabilities.shape[1],):
        raise ValueError("thresholds must have one value per probability column")
    return (probabilities >= thresholds[None, :]).astype(np.uint8)


def category_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-category true positives, false positives and false negatives (each one value per column)."""
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return ((y_true & y_pred).sum(axis=0), (~y_true & y_pred).sum(axis=0), (y_true & ~y_pred).sum(axis=0))


def f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """F1 = 2TP / (2TP + FP + FN) per category, 0 where the denominator is 0 (sklearn's zero_division=0)."""
    tp, fp, fn = (np.asarray(x, dtype=np.float64) for x in (tp, fp, fn))
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)


def per_category_aupr(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Average precision per category; NaN for a category with no positives (it is undefined there)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    if y_true.ndim != 2 or y_true.shape != scores.shape:
        raise ValueError("y_true and scores must be matching two-dimensional arrays")
    result = np.full(y_true.shape[1], np.nan)
    active = np.any(y_true == 1, axis=0)
    if active.any():
        result[active] = average_precision_score(y_true[:, active], scores[:, active], average=None)
    return result


def macro_f1(per_category_f1: np.ndarray) -> float:
    """Mean F1 over every category, so a category with no positives contributes 0 (fixed-41 rule)."""
    return float(np.mean(per_category_f1))


def macro_aupr(per_category: np.ndarray) -> float:
    """Mean AUPR over categories that have at least one positive (NaN entries are skipped)."""
    per_category = np.asarray(per_category, dtype=np.float64)
    return float(np.nanmean(per_category)) if np.any(~np.isnan(per_category)) else float("nan")


def detection_table(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray,
                    categories: Sequence[str] = CATEGORIES) -> pd.DataFrame:
    """
    One row per category, in `categories` order: support, TP/FP/FN, precision, recall, F1 (all 0 when
    undefined) and AUPR (NaN without positives), at the given fixed thresholds.
    """
    tp, fp, fn = category_counts(y_true, predict_at_thresholds(scores, thresholds))
    tp_f, fp_f, fn_f = (x.astype(np.float64) for x in (tp, fp, fn))
    precision = np.divide(tp_f, tp_f + fp_f, out=np.zeros_like(tp_f), where=(tp + fp) != 0)
    recall = np.divide(tp_f, tp_f + fn_f, out=np.zeros_like(tp_f), where=(tp + fn) != 0)
    return pd.DataFrame({
        "category": list(categories), "support": (tp + fn).astype(int),
        "tp": tp.astype(int), "fp": fp.astype(int), "fn": fn.astype(int),
        "precision": precision, "recall": recall, "f1": f1_from_counts(tp, fp, fn),
        "aupr": per_category_aupr(y_true, scores),
    })


def contract_groups(contract_ids: Sequence[str]) -> tuple[list[str], list[np.ndarray]]:
    """Return (contract order by first appearance, row indices of each contract), for chunk-ordered rows."""
    ids = np.asarray(list(contract_ids), dtype=object)
    order = list(dict.fromkeys(ids.tolist()))
    return order, [np.flatnonzero(ids == contract_id) for contract_id in order]


def contract_presence(y_true: np.ndarray, scores: np.ndarray,
                      groups: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse chunks to contracts: gold = 1 if any of the contract's chunks is labeled, score = the
    highest chunk probability. Thresholding that score is the same as "any chunk detected".
    """
    gold = np.stack([np.asarray(y_true)[rows].max(axis=0) for rows in groups]).astype(np.uint8)
    best = np.stack([np.asarray(scores)[rows].max(axis=0) for rows in groups])
    return gold, best


def bootstrap_replicates(n_contracts: int, replicates: int, seed: int,
                         statistic: Callable[[np.ndarray], dict[str, np.ndarray | float]]) -> dict[str, np.ndarray]:
    """
    Resample contracts with replacement `replicates` times (numpy default_rng(seed), one
    rng.integers(0, n, size=n) draw per replicate) and collect statistic(selected) for each draw.
    Every statistic sees the same draws, so differences between models are paired.
    """
    if replicates < 1 or n_contracts < 1:
        raise ValueError("replicates and n_contracts must be positive")
    rng = np.random.default_rng(seed)
    collected: dict[str, list] = {}
    for _ in range(replicates):
        selected = rng.integers(0, n_contracts, size=n_contracts)
        for key, value in statistic(selected).items():
            collected.setdefault(key, []).append(value)
    return {key: np.asarray(values, dtype=np.float64) for key, values in collected.items()}


def percentile_interval(values: np.ndarray) -> dict[str, float | int]:
    """2.5th and 97.5th percentiles over replicates (NaN replicates skipped), with the count used."""
    values = np.asarray(values, dtype=np.float64)
    defined = values[~np.isnan(values)]
    if defined.size == 0:
        return {"lower_95": float("nan"), "upper_95": float("nan"), "replicates_defined": 0}
    lower, upper = np.quantile(defined, [0.025, 0.975])
    return {"lower_95": float(lower), "upper_95": float(upper), "replicates_defined": int(defined.size)}


def jackknife_replicates(n_contracts: int,
                        statistic: Callable[[np.ndarray], dict[str, np.ndarray | float]]) -> dict[str, np.ndarray]:
    """statistic() with each contract left out once (n_contracts evaluations), for the BCa acceleration."""
    everyone = np.arange(n_contracts)
    collected: dict[str, list] = {}
    for left_out in range(n_contracts):
        for key, value in statistic(np.delete(everyone, left_out)).items():
            collected.setdefault(key, []).append(value)
    return {key: np.asarray(values, dtype=np.float64) for key, values in collected.items()}


def bca_interval(replicates: np.ndarray, point: float, jackknife: np.ndarray) -> dict:
    """
    Bias-corrected and accelerated 95% bootstrap interval (Efron 1987), with the plain percentile
    bounds alongside. z0 = Phi^-1(share of replicates below the point estimate, ties counted half);
    acceleration a = sum(d^3) / (6 sum(d^2)^1.5) with d = jackknife mean - jackknife value. Falls back
    to the percentile interval (method "percentile") when z0 or the adjusted levels are not finite.
    NaN replicates and jackknife values are skipped.
    """
    percentile = percentile_interval(replicates)
    result = {**percentile, "method": "percentile", "percentile_lower_95": percentile["lower_95"],
              "percentile_upper_95": percentile["upper_95"]}
    replicates = np.asarray(replicates, dtype=np.float64)
    defined = replicates[~np.isnan(replicates)]
    if defined.size == 0 or point is None or np.isnan(point):
        return result
    below = (np.sum(defined < point) + 0.5 * np.sum(defined == point)) / defined.size
    jackknife = np.asarray(jackknife, dtype=np.float64)
    jackknife = jackknife[~np.isnan(jackknife)]
    deviations = jackknife.mean() - jackknife if jackknife.size else np.zeros(1)
    denominator = 6 * np.sum(deviations ** 2) ** 1.5
    acceleration = float(np.sum(deviations ** 3) / denominator) if denominator > 0 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z0 = float(norm.ppf(below))
        levels = [norm.cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z))) for z in norm.ppf([0.025, 0.975])]
    if not (np.isfinite(z0) and np.all(np.isfinite(levels))):
        return result
    lower, upper = np.quantile(defined, levels)
    return {**result, "lower_95": float(lower), "upper_95": float(upper), "method": "bca",
            "z0": z0, "acceleration": acceleration}


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman correlation: Pearson correlation of average ranks (ties share their mean rank)."""
    ranks_a = pd.Series(np.asarray(a, dtype=np.float64)).rank(method="average").to_numpy()
    ranks_b = pd.Series(np.asarray(b, dtype=np.float64)).rank(method="average").to_numpy()
    if np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
        return float("nan")
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def compare_with_reference(point: float, lower: float, upper: float,
                           reference: float, reference_min: float, reference_max: float) -> dict:
    """
    Pre-registered wording rule: "consistent" when the development mean lies inside the test 95%
    interval, otherwise "lower" or "higher"; plus where the point estimate falls against the range
    of the development splits.
    """
    if upper < reference:
        verdict = "lower_than_development"
    elif lower > reference:
        verdict = "higher_than_development"
    else:
        verdict = "consistent_with_development"
    if point < reference_min:
        position = "below_every_development_split"
    elif point > reference_max:
        position = "above_every_development_split"
    else:
        position = "within_development_range"
    return {"verdict": verdict, "position": position, "gap_to_development_mean": point - reference}


def support_status(support: int) -> str:
    """'unscorable' with no positives, 'insufficient support' under RARE_SUPPORT, else 'reported'."""
    if support == 0:
        return "unscorable"
    return "insufficient support" if support < RARE_SUPPORT else "reported"


def support_bucket(support: float) -> str:
    """Bucket by positives: rare (<10), uncommon (10-29), common (30+), as in docs/error-analysis.md."""
    if support < RARE_SUPPORT:
        return "rare"
    return "uncommon" if support < COMMON_SUPPORT else "common"
