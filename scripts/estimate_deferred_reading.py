"""Compute the time-saved inputs of the final report from the sealed run's saved outputs.

The report's estimate assumes a reviewer reads every window in which the model detects at least
one category and defers the rest. This script measures, post hoc, how much of each contract's text
lies outside the flagged windows (d) and what share of annotated (window, category) labels sits in
that deferred text (m), then prints the deferred-reading grid for the report's assumed reading
speeds. It reads only `predictions.npz`, `chunks.csv`, `per_category_metrics.csv` and
`contract_level_metrics.csv` from the results folder, never any file under `data/cuad/`, and it
checks that its detections reproduce the saved per-category TP/FP/FN before printing anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "final_test_evaluation"
FORBIDDEN_DATA_DIR = ROOT / "data" / "cuad"
# Median words per contract in the training file, quoted from data/cuad/EDA_SUMMARY.md (not read here)
MEDIAN_CONTRACT_WORDS = 5357.5
READING_SPEEDS_WPM = (100.0, 150.0, 250.0)


def merged_length(intervals: list[tuple[int, int]]) -> int:
    """Total length covered by half-open character intervals, counting overlaps once."""
    total = 0
    current_start = current_end = None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def load_detections(results_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load labels, threshold decisions and window positions, checked against the saved counts.

    Probability columns follow the row order of `per_category_metrics.csv`; the check that every
    category's TP, FP and FN equal the saved ones confirms that order and the >= threshold rule.
    """
    # Refuse anything inside the CUAD data folder
    resolved = results_dir.resolve()
    if resolved == FORBIDDEN_DATA_DIR or FORBIDDEN_DATA_DIR in resolved.parents:
        raise SystemExit(f"refusing to read from {FORBIDDEN_DATA_DIR}")

    # Saved probabilities, labels, thresholds and windows
    saved = np.load(results_dir / "predictions.npz")
    y_true = saved["y_true"].astype(bool)
    probabilities = saved["word_char_tfidf"].astype(np.float64)
    metrics = pd.read_csv(results_dir / "per_category_metrics.csv")
    chunks = pd.read_csv(results_dir / "chunks.csv")
    if len(chunks) != len(y_true) or len(metrics) != y_true.shape[1]:
        raise SystemExit("predictions.npz does not match chunks.csv or per_category_metrics.csv")
    detected = probabilities >= metrics["threshold"].to_numpy(dtype=np.float64)

    # Reproduce the saved per-category counts
    counts = {
        "tp": (detected & y_true).sum(axis=0),
        "fp": (detected & ~y_true).sum(axis=0),
        "fn": (~detected & y_true).sum(axis=0),
    }
    for name, values in counts.items():
        if not np.array_equal(values, metrics[name].to_numpy()):
            raise SystemExit(f"recomputed {name} differ from per_category_metrics.csv")
    return y_true, detected, chunks


def deferred_reading(results_dir: Path) -> dict[str, float]:
    """Measure d (text outside flagged windows) and m (annotated labels in that text)."""
    y_true, detected, chunks = load_detections(results_dir)
    flagged = detected.any(axis=1)

    # Share of each contract's text outside the merged flagged windows
    lengths, flagged_lengths, unflagged_contracts = [], [], 0
    for _, windows in chunks.groupby("contract_id", sort=False):
        rows = windows.index.to_numpy()
        length = int(windows["end"].max())
        covered = merged_length(list(zip(windows["start"][flagged[rows]], windows["end"][flagged[rows]])))
        lengths.append(length)
        flagged_lengths.append(covered)
        unflagged_contracts += int(not flagged[rows].any())
    lengths_array = np.asarray(lengths, dtype=np.float64)
    deferred_share = 1.0 - np.asarray(flagged_lengths, dtype=np.float64) / lengths_array

    # Annotated labels in deferred text, and contract-level misses
    contract_level = pd.read_csv(results_dir / "contract_level_metrics.csv")
    contract_tp, contract_fn = int(contract_level["tp"].sum()), int(contract_level["fn"].sum())
    return {
        "contracts": len(lengths),
        "detections": int(detected.sum()),
        "false_detections": int((detected & ~y_true).sum()),
        "d_median": float(np.percentile(deferred_share, 50)),
        "d_lower_quartile": float(np.percentile(deferred_share, 25)),
        "d_upper_quartile": float(np.percentile(deferred_share, 75)),
        "d_pooled": float(1.0 - sum(flagged_lengths) / lengths_array.sum()),
        "labels": int(y_true.sum()),
        "labels_deferred": int(y_true[~flagged].sum()),
        "m": float(y_true[~flagged].sum() / y_true.sum()),
        "contracts_without_flags": unflagged_contracts,
        "contract_presences": contract_tp + contract_fn,
        "contract_presences_missed": contract_fn,
        "median_contract_characters": float(np.median(lengths_array)),
    }


def deferred_minutes(share: float, words: float, speed_wpm: float) -> float:
    """Minutes of reading deferred: share of text times words, divided by reading speed."""
    return share * words / speed_wpm


def main(argv: list[str] | None = None) -> None:
    """Print d, m and the deferred-reading grid quoted in docs/final-report.md."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    values = deferred_reading(args.results)

    # Measured inputs
    print(f"contracts {values['contracts']}, detections {values['detections']} "
          f"({values['false_detections']} false)")
    print(f"d median {values['d_median']:.1%}, quartiles {values['d_lower_quartile']:.1%} to "
          f"{values['d_upper_quartile']:.1%}, pooled {values['d_pooled']:.1%}")
    print(f"m {values['m']:.1%} ({values['labels_deferred']} of {values['labels']} labels)")
    print(f"contracts without a flagged window: {values['contracts_without_flags']}")
    print(f"contract-level misses: {values['contract_presences_missed']} of {values['contract_presences']}")
    print(f"median test contract length: {values['median_contract_characters']:,.0f} characters")

    # Deferred minutes per median contract at the assumed reading speeds
    print(f"deferred minutes for {MEDIAN_CONTRACT_WORDS} words at {', '.join(f'{r:g}' for r in READING_SPEEDS_WPM)} wpm")
    for label in ("d_lower_quartile", "d_median", "d_upper_quartile"):
        minutes = [deferred_minutes(values[label], MEDIAN_CONTRACT_WORDS, r) for r in READING_SPEEDS_WPM]
        print(f"  {label} {values[label]:.1%}: " + ", ".join(f"{m:.1f}" for m in minutes))
    print("full read: " + ", ".join(f"{MEDIAN_CONTRACT_WORDS / r:.1f}" for r in READING_SPEEDS_WPM))


if __name__ == "__main__":
    main()
