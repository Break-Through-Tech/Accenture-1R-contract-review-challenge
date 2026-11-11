"""Evaluate category-description cue matches as a small TF-IDF logit adjustment.

A single nonnegative boost is selected on calibration macro-AUPR from a predeclared grid.
The cue lexicon is built from category descriptions, not evaluation labels. This is an
exploratory ablation over saved TF-IDF predictions and never reads test.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from scripts.analyze_reranker_multiseed import (
    SEEDS, _bootstrap_delta, _macro_aupr, _macro_f1, _prepare_evaluation,
    _run_dir, _threshold_vector,
)
from scripts.train_retrieval_reranker import _metrics_at_thresholds
from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.evidence.cues import load_category_cues, normalize_category
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.models.encoder.experiment_utils import ROOT, TRAIN_PATH, calibrate_thresholds
from src.pipeline import prepare_contracts, to_model_inputs

OUTPUT = ROOT / "results/category_cue_boost"
BOOST_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def _match_matrix(texts: list[str], cues: dict[str, list[str]]) -> np.ndarray:
    """Binary chunk/category matrix: 1 if any category cue occurs as a whole phrase."""
    patterns = []
    for category in CATEGORIES:
        category_cues = cues.get(normalize_category(category), [])
        patterns.append([
            re.compile(rf"(?<!\w){re.escape(cue.casefold())}(?!\w)")
            for cue in category_cues if cue.strip()
        ])
    matches = np.zeros((len(texts), len(CATEGORIES)), dtype=bool)
    for row, text in enumerate(texts):
        normalized = text.casefold()
        for column, category_patterns in enumerate(patterns):
            matches[row, column] = any(pattern.search(normalized) for pattern in category_patterns)
    return matches


def _adjust_logits(probabilities: np.ndarray, matches: np.ndarray, boost: float) -> np.ndarray:
    if probabilities.shape != matches.shape:
        raise ValueError("probabilities and cue matches must have identical shape")
    clipped = np.clip(probabilities.astype(np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(clipped) - np.log1p(-clipped)
    logits += boost * matches
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))).astype(np.float32)


def evaluate(bootstrap_replicates: int = 1000) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )
    contracts = load_cuad(TRAIN_PATH)
    cues = load_category_cues()
    results, alpha_rows, category_rows = [], [], []
    for seed in SEEDS:
        run_dir = _run_dir(seed)
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        if report.get("test_data_used") is not False:
            raise ValueError(f"Seed {seed}: test-data provenance is not explicitly false")
        y_eval, ids, contract_rows = _prepare_evaluation(seed, tokenizer, contracts)
        calibration = np.load(run_dir / "calibration_predictions.npz")
        evaluation = np.load(run_dir / "evaluation_predictions.npz")
        if not np.array_equal(y_eval, evaluation["y_true"]):
            raise ValueError(f"Seed {seed}: evaluation labels do not match saved predictions")
        if {name: hashlib.sha256("\n".join(values).encode()).hexdigest()
            for name, values in ids.items()} != report["split"]["contract_id_sha256"]:
            raise ValueError(f"Seed {seed}: contract split differs from report")

        def chunk_texts(contract_group):
            chunks = prepare_contracts(contract_group, tokenizer, max_tokens=512, overlap_tokens=64)
            return [item.chunk.text for item in chunks]

        # Reconstruct split groups in the exact seeded order validated above.
        from src.labels.splits import split_by_contract
        development, eval_contracts = split_by_contract(contracts, val_fraction=0.2, seed=seed)
        train_contracts, calibration_contracts = split_by_contract(
            development, val_fraction=0.2, seed=seed,
        )
        ordered_ids = {
            "training": [item.contract_id for item in train_contracts],
            "calibration": [item.contract_id for item in calibration_contracts],
            "evaluation": [item.contract_id for item in eval_contracts],
        }
        if ordered_ids != ids:
            raise ValueError(f"Seed {seed}: reconstructed contract order mismatch")
        cal_texts, eval_texts = chunk_texts(calibration_contracts), chunk_texts(eval_contracts)
        cal_matches, eval_matches = _match_matrix(cal_texts, cues), _match_matrix(eval_texts, cues)
        sparse_cal, sparse_eval = calibration["retriever_score"], evaluation["retriever_score"]
        if cal_matches.shape != sparse_cal.shape or eval_matches.shape != sparse_eval.shape:
            raise ValueError(f"Seed {seed}: reconstructed chunks do not align with saved scores")

        score_by_boost = {
            boost: _adjust_logits(sparse_cal, cal_matches, boost) for boost in BOOST_GRID
        }
        cal_aupr = {boost: _macro_aupr(calibration["y_true"], scores)
                    for boost, scores in score_by_boost.items()}
        selected = max(BOOST_GRID, key=lambda boost: (cal_aupr[boost], -boost))
        adjusted_cal = score_by_boost[selected]
        adjusted_eval = _adjust_logits(sparse_eval, eval_matches, selected)
        baseline_thresholds = calibrate_thresholds(calibration["y_true"], sparse_cal)
        adjusted_thresholds = calibrate_thresholds(calibration["y_true"], adjusted_cal)
        base_vector = _threshold_vector(baseline_thresholds)
        adjusted_vector = _threshold_vector(adjusted_thresholds)
        base_f1 = _macro_f1(y_eval, sparse_eval, base_vector)
        adjusted_f1 = _macro_f1(y_eval, adjusted_eval, adjusted_vector)
        base_aupr = _macro_aupr(y_eval, sparse_eval)
        adjusted_aupr = _macro_aupr(y_eval, adjusted_eval)
        interval = _bootstrap_delta(
            y_eval, sparse_eval, adjusted_eval, base_vector, adjusted_vector,
            contract_rows, seed, bootstrap_replicates,
        )
        results.append({
            "seed": seed, "selected_boost": selected,
            "tfidf_macro_f1": base_f1, "cue_macro_f1": adjusted_f1,
            "delta_macro_f1": adjusted_f1 - base_f1,
            "tfidf_macro_aupr": base_aupr, "cue_macro_aupr": adjusted_aupr,
            "delta_macro_aupr": adjusted_aupr - base_aupr,
            "paired_contract_bootstrap_95pct": interval,
            "cue_match_fraction": float(eval_matches.mean()),
        })
        for boost in BOOST_GRID:
            alpha_rows.append({"seed": seed, "boost": boost,
                               "calibration_macro_aupr": cal_aupr[boost]})
        baseline_metrics = _metrics_at_thresholds(y_eval, sparse_eval, baseline_thresholds).set_index("category")
        cue_metrics = _metrics_at_thresholds(y_eval, adjusted_eval, adjusted_thresholds).set_index("category")
        for category in CATEGORIES:
            category_rows.append({
                "seed": seed, "category": category,
                "support": int(baseline_metrics.loc[category, "support"]),
                "tfidf_f1": float(baseline_metrics.loc[category, "f1"]),
                "cue_f1": float(cue_metrics.loc[category, "f1"]),
                "delta_f1": float(cue_metrics.loc[category, "f1"] - baseline_metrics.loc[category, "f1"]),
                "tfidf_aupr": float(baseline_metrics.loc[category, "aupr"]),
                "cue_aupr": float(cue_metrics.loc[category, "aupr"]),
                "delta_aupr": float(cue_metrics.loc[category, "aupr"] - baseline_metrics.loc[category, "aupr"]),
            })

    summary = {
        "data_sha256": json.loads((_run_dir(SEEDS[0]) / "report.json").read_text())["data_sha256"],
        "method": "per-category binary cue-match indicator adds one shared nonnegative logit offset; cues come from category descriptions",
        "boost_grid": list(BOOST_GRID),
        "selection": "one global boost selected independently per seed using calibration macro-AUPR; per-category thresholds also fit on calibration only",
        "seeds": results,
        "mean_delta_macro_f1": float(np.mean([item["delta_macro_f1"] for item in results])),
        "mean_delta_macro_aupr": float(np.mean([item["delta_macro_aupr"] for item in results])),
        "interpretation": "exploratory ablation on previously inspected contract evaluation splits; cue matches are not risk policy",
        "test_data_used": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(results).to_csv(OUTPUT / "per_seed.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(OUTPUT / "calibration_grid.csv", index=False)
    pd.DataFrame(category_rows).to_csv(OUTPUT / "per_category_by_seed.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.bootstrap_replicates), indent=2))


if __name__ == "__main__":
    main()
