"""Error analysis of the word+char TF-IDF control and the retrieval reranker.

Reads only saved scores and thresholds from `results/retrieval_reranker_seed{N}/` (the
retriever there is the word+char TF-IDF control, scored on the same chunks as the reranker),
so nothing is retrained. Thresholds were fit on calibration contracts; everything here is
measured on the held-out evaluation contracts of the training file. Never reads test.json.

Writes `results/error_analysis/`:
  category_errors_by_seed.csv   per model, seed and category: counts, F1, AUPR, threshold gap
  category_summary.csv          the same averaged over seeds, with F1 spread across seeds
  bucket_summary.csv            macro metrics by category support bucket
  false_positive_sources.csv    which gold labels each category's false positives really had
  model_disagreement.csv        where the two models catch or miss different positives
  summary.json                  headline numbers quoted in docs/error-analysis.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clausetriage.error_analysis import (
    NO_LABEL,
    apply_category_thresholds,
    category_error_table,
    false_positive_sources,
    model_disagreement,
    seed_stability,
)
from clausetriage.labels.categories import CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [7, 17, 23, 42, 101]
MODELS = {"word_char_tfidf": ("retriever_score", "retriever"),
          "reranker": ("reranker_score", "reranker")}


def _load_seed(directory: Path) -> dict:
    """Saved evaluation/calibration scores and per-model thresholds for one seed."""
    evaluation = np.load(directory / "evaluation_predictions.npz")
    calibration = np.load(directory / "calibration_predictions.npz")
    thresholds = json.loads((directory / "thresholds.json").read_text())
    return {"evaluation": evaluation, "calibration_support": calibration["y_true"].sum(axis=0),
            "thresholds": thresholds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    output = args.results / "error_analysis"
    output.mkdir(parents=True, exist_ok=True)

    tables, sources, disagreements, macro = [], [], [], []
    for seed in args.seeds:
        saved = _load_seed(args.results / f"retrieval_reranker_seed{seed}")
        y_true = saved["evaluation"]["y_true"].astype(int)
        predictions = {}

        # Per-category error table and false-positive sources for each model
        for model, (score_key, threshold_key) in MODELS.items():
            scores = saved["evaluation"][score_key]
            thresholds = saved["thresholds"][threshold_key]
            table = category_error_table(y_true, scores, thresholds, CATEGORIES)
            table.insert(0, "seed", seed)
            table.insert(0, "model", model)
            table["calibration_support"] = saved["calibration_support"].astype(int)
            tables.append(table)

            predictions[model] = apply_category_thresholds(scores, thresholds, CATEGORIES)
            fp = false_positive_sources(y_true, predictions[model], CATEGORIES)
            fp.insert(0, "seed", seed)
            fp.insert(0, "model", model)
            sources.append(fp)

            # Macro F1 counting unscorable categories as 0, matching the saved reports
            macro.append({"model": model, "seed": seed,
                          "macro_f1": float(table["f1"].fillna(0).mean()),
                          "macro_aupr": float(table["aupr"].mean())})

        # Where the two models make different mistakes
        agreement = model_disagreement(y_true, predictions["word_char_tfidf"], predictions["reranker"],
                                       CATEGORIES)
        agreement.insert(0, "seed", seed)
        disagreements.append(agreement)

    by_seed = pd.concat(tables, ignore_index=True)
    by_seed.to_csv(output / "category_errors_by_seed.csv", index=False)

    # Average over seeds; F1 spread shows how much a category depends on the split
    numeric = ["support", "calibration_support", "tp", "fp", "fn", "f1", "aupr", "threshold_gap",
               "fp_on_unlabeled"]
    summary = by_seed.groupby(["model", "category"])[numeric].mean().reset_index()
    stability = seed_stability(by_seed, "f1")[["model", "category", "std_f1", "min_f1", "max_f1"]]
    summary = summary.merge(stability, on=["model", "category"])
    summary["fp_unlabeled_share"] = by_seed.groupby(["model", "category"]).apply(
        lambda g: g["fp_on_unlabeled"].sum() / max(g["fp"].sum(), 1), include_groups=False).values
    summary["support_bucket"] = summary["support"].round().astype(int).map(
        lambda s: "rare (<10)" if s < 10 else ("uncommon (10-29)" if s < 30 else "common (30+)"))
    summary = summary.sort_values(["model", "f1"]).reset_index(drop=True)
    summary.round(4).to_csv(output / "category_summary.csv", index=False)

    # Macro metrics by support bucket
    buckets = (summary.groupby(["model", "support_bucket"])
               .agg(categories=("category", "count"), mean_f1=("f1", "mean"),
                    mean_aupr=("aupr", "mean"), mean_threshold_gap=("threshold_gap", "mean"),
                    mean_f1_std_across_seeds=("std_f1", "mean"))
               .reset_index())
    buckets.round(4).to_csv(output / "bucket_summary.csv", index=False)

    # False-positive sources summed over seeds
    fp_all = pd.concat(sources, ignore_index=True)
    fp_sum = fp_all.groupby(["model", "predicted_category", "gold_label"])["count"].sum().reset_index()
    totals = by_seed.groupby(["model", "category"])["fp"].sum().rename("category_false_positives")
    fp_sum = fp_sum.merge(totals, left_on=["model", "predicted_category"], right_index=True)
    fp_sum["share"] = fp_sum["count"] / fp_sum["category_false_positives"]
    fp_sum = fp_sum.sort_values(["model", "predicted_category", "count"], ascending=[True, True, False])
    fp_sum.round(4).to_csv(output / "false_positive_sources.csv", index=False)

    # Disagreement summed over seeds
    count_columns = ["support", "both_caught", "only_a_caught", "only_b_caught", "both_missed",
                     "fp_only_a", "fp_only_b", "fp_both"]
    disagreement = pd.concat(disagreements).groupby("category")[count_columns].sum().reset_index()
    disagreement = disagreement.rename(columns=lambda c: c.replace("_a_", "_tfidf_").replace("_b_", "_reranker_")
                                       .replace("fp_only_a", "fp_only_tfidf").replace("fp_only_b", "fp_only_reranker"))
    disagreement.to_csv(output / "model_disagreement.csv", index=False)

    # Headline numbers
    macro_frame = pd.DataFrame(macro)
    total = by_seed.groupby("model")[["tp", "fp", "fn", "fp_on_unlabeled"]].sum()
    fp_cross = fp_sum[fp_sum["gold_label"] != NO_LABEL]
    top_confusions = (fp_cross.sort_values("count", ascending=False)
                      .groupby("model").head(8)[["model", "predicted_category", "gold_label", "count", "share"]])
    caught = disagreement[["only_tfidf_caught", "only_reranker_caught", "both_caught", "both_missed"]].sum()
    report = {
        "seeds": args.seeds,
        "macro_by_seed": macro_frame.to_dict("records"),
        "macro_mean": macro_frame.groupby("model")[["macro_f1", "macro_aupr"]].mean().to_dict("index"),
        "error_totals": total.to_dict("index"),
        "fp_share_on_unlabeled_chunks": (total["fp_on_unlabeled"] / total["fp"]).to_dict(),
        "fn_share_of_errors": (total["fn"] / (total["fn"] + total["fp"])).to_dict(),
        "top_cross_category_confusions": top_confusions.round(4).to_dict("records"),
        "positives_caught": {key: int(value) for key, value in caught.items()},
        "test_data_used": False,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["macro_mean", "fp_share_on_unlabeled_chunks",
                                             "fn_share_of_errors", "positives_caught"]}, indent=2))


if __name__ == "__main__":
    main()
