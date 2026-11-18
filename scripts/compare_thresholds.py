"""Compare threshold strategies for the word+char TF-IDF control and the retrieval reranker.

Every strategy is fit on calibration contracts only and scored on the held-out evaluation
contracts of the training file, using the saved scores in
`results/retrieval_reranker_seed{N}/`. Nothing is retrained and test.json is never read.

The strategy settings (shrinkage prior 10, rare cut-off 10 calibration positives, recall floor
0.5) are fixed in `clausetriage/threshold_strategies.py` before any evaluation score was looked at.
These evaluation contracts were already inspected in earlier experiments, so any winner here is
a candidate for the untouched-holdout gate in docs/upgrade-validation-gate.md, not a final result.

Writes `results/threshold_robustness/`:
  per_seed.csv      macro-F1 overall and by calibration-support bucket, per model/strategy/seed
  summary.csv       the same averaged over seeds, with change vs. the current method and wins
  per_category.csv  mean F1 per category for each strategy, and change vs. the current method
  summary.json      headline numbers quoted in docs/threshold-robustness.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clausetriage.error_analysis import category_error_table
from clausetriage.labels.categories import CATEGORIES
from clausetriage.threshold_strategies import RARE_SUPPORT, STRATEGIES

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [7, 17, 23, 42, 101]
MODELS = {"word_char_tfidf": "retriever_score", "reranker": "reranker_score"}
CURRENT = "per_category_f1 (current)"


def calibration_bucket(support: int) -> str:
    """Bucket a category by its calibration positives, the quantity the strategies react to."""
    if support < RARE_SUPPORT:
        return "rare"
    return "uncommon" if support < 30 else "common"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    output = args.results / "threshold_robustness"
    output.mkdir(parents=True, exist_ok=True)

    seed_rows, category_rows = [], []
    for seed in args.seeds:
        directory = args.results / f"retrieval_reranker_seed{seed}"
        calibration = np.load(directory / "calibration_predictions.npz")
        evaluation = np.load(directory / "evaluation_predictions.npz")
        buckets = [calibration_bucket(int(n)) for n in calibration["y_true"].sum(axis=0)]

        for model, score_key in MODELS.items():
            for strategy, fit in STRATEGIES.items():
                # Fit on calibration, score on evaluation
                thresholds = fit(calibration["y_true"], calibration[score_key], CATEGORIES)
                table = category_error_table(evaluation["y_true"], evaluation[score_key], thresholds, CATEGORIES)
                table["f1"] = table["f1"].fillna(0.0)  # unscorable categories count as 0, as in saved reports
                table["calibration_bucket"] = buckets

                row = {"model": model, "strategy": strategy, "seed": seed,
                       "macro_f1": float(table["f1"].mean())}
                for bucket, group in table.groupby("calibration_bucket"):
                    row[f"{bucket}_f1"] = float(group["f1"].mean())
                    row[f"{bucket}_categories"] = int(len(group))
                seed_rows.append(row)

                for _, item in table.iterrows():
                    category_rows.append({"model": model, "strategy": strategy, "seed": seed,
                                          "category": item["category"], "f1": item["f1"],
                                          "precision": item["precision"], "recall": item["recall"],
                                          "threshold": item["threshold"],
                                          "calibration_bucket": item["calibration_bucket"]})

    per_seed = pd.DataFrame(seed_rows)
    per_seed.round(4).to_csv(output / "per_seed.csv", index=False)

    # Mean over seeds, change vs. the current method, and per-seed wins
    metrics = ["macro_f1", "rare_f1", "uncommon_f1", "common_f1"]
    current = per_seed[per_seed["strategy"] == CURRENT].set_index(["model", "seed"])[metrics]
    deltas = per_seed.set_index(["model", "seed"])
    deltas = deltas[metrics].sub(current, axis=0).add_prefix("delta_").reset_index()
    deltas["strategy"] = per_seed["strategy"].values
    summary = per_seed.groupby(["model", "strategy"])[metrics].mean()
    summary = summary.join(deltas.groupby(["model", "strategy"])[[f"delta_{m}" for m in metrics]].mean())
    wins = deltas.assign(win=deltas["delta_macro_f1"] > 1e-9).groupby(["model", "strategy"])["win"].sum()
    summary["seeds_beating_current"] = wins
    summary["worst_seed_delta_macro_f1"] = deltas.groupby(["model", "strategy"])["delta_macro_f1"].min()
    summary = summary.reset_index()
    summary.round(4).to_csv(output / "summary.csv", index=False)

    # Per-category mean F1 and change vs. current
    categories = pd.DataFrame(category_rows)
    by_category = categories.groupby(["model", "strategy", "category"])[["f1", "precision", "recall"]].mean()
    baseline = by_category.xs(CURRENT, level="strategy")["f1"]
    by_category["delta_f1"] = by_category["f1"] - baseline.reindex(
        by_category.index.droplevel("strategy")).values
    by_category.reset_index().round(4).to_csv(output / "per_category.csv", index=False)

    report = {
        "seeds": args.seeds,
        "fit_on": "calibration contracts only",
        "evaluated_on": "held-out evaluation contracts of the CUAD training file (previously inspected)",
        "summary": summary.round(4).to_dict("records"),
        "test_data_used": False,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    columns = ["model", "strategy", "macro_f1", "delta_macro_f1", "delta_rare_f1", "delta_common_f1",
               "seeds_beating_current", "worst_seed_delta_macro_f1"]
    print(summary[columns].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
