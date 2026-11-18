"""Rewrite per-category AUPR in saved result tables from the saved evaluation scores.

Before the `add_aupr_column` fix, four scripts assigned an AUPR array in CATEGORIES order to a
report already sorted by support, so each row got another category's AUPR. Macro AUPR was
unaffected (a mean does not depend on order). This script recomputes the column from the saved
`*_evaluation.npz` / `evaluation_predictions.npz` files and matches it by category name, so no
model is rerun. It is idempotent and never reads CUAD test.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clausetriage.labels.categories import CATEGORIES
from clausetriage.metrics import compute_aupr

ROOT = Path(__file__).resolve().parents[1]
SUITE_SEEDS = [7, 17, 23, 42]
SINGLE_MODEL_DIRS = ["category_query_seed42", "long_context_transformer_seed42"]


def _aupr_map(npz_path: Path) -> dict[str, float]:
    """Per-category AUPR keyed by category name from one saved score file."""
    saved = np.load(npz_path)
    return dict(zip(CATEGORIES, compute_aupr(saved["y_true"], saved["y_score"])))


def _repair(csv_path: Path, aupr_for_row) -> int:
    """Rewrite the aupr column of one CSV; returns how many rows changed."""
    table = pd.read_csv(csv_path)
    corrected = table.apply(aupr_for_row, axis=1)
    changed = int((~np.isclose(table["aupr"], corrected, equal_nan=True)).sum())
    table["aupr"] = corrected
    table.to_csv(csv_path, index=False)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    # Architecture-suite tables: one npz per (seed, model)
    maps: dict[tuple[int, str], dict[str, float]] = {}
    for seed in SUITE_SEEDS:
        directory = args.results / f"architecture_suite_seed{seed}"
        table = pd.read_csv(directory / "per_category_metrics.csv")
        for model in table["model"].unique():
            maps[(seed, model)] = _aupr_map(directory / f"{model}_evaluation.npz")
        changed = _repair(directory / "per_category_metrics.csv",
                          lambda row, s=seed: maps[(s, row["model"])][row["category"]])
        print(f"{directory.name}/per_category_metrics.csv: {changed} rows corrected")

    # Multi-seed table built from the four suite runs
    by_seed = args.results / "architecture_suite_seed42/per_category_metrics_by_seed.csv"
    changed = _repair(by_seed, lambda row: maps[(int(row["seed"]), row["model"])][row["category"]])
    print(f"{by_seed.parent.name}/{by_seed.name}: {changed} rows corrected")

    # Single-model experiments
    for name in SINGLE_MODEL_DIRS:
        directory = args.results / name
        aupr = _aupr_map(directory / "evaluation_predictions.npz")
        changed = _repair(directory / "per_category_metrics.csv", lambda row, aupr=aupr: aupr[row["category"]])
        print(f"{name}/per_category_metrics.csv: {changed} rows corrected")


if __name__ == "__main__":
    main()
