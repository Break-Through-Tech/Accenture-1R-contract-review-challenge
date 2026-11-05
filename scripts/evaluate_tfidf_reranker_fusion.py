"""Exploratory score fusion: preserve TF-IDF scores and add reranker evidence on candidates.

Uses only saved predictions from CUAD training-file splits. A single global logit-blend
weight is chosen from a predeclared grid by calibration macro-AUPR; per-category thresholds
are then calibrated on the same calibration partition. Evaluation is diagnostic because
these splits have already been inspected in prior experiments. Never reads test.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from scripts.analyze_reranker_multiseed import (
    SEEDS, _bootstrap_delta, _macro_aupr, _macro_f1, _prepare_evaluation,
    _run_dir, _threshold_vector,
)
from scripts.train_retrieval_reranker import _metrics_at_thresholds
from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.models.encoder.experiment_utils import ROOT, TRAIN_PATH, calibrate_thresholds

OUTPUT = ROOT / "results/tfidf_reranker_fusion"
ALPHA_GRID = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(probabilities.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def _fuse(sparse: np.ndarray, reranker: np.ndarray, alpha: float) -> np.ndarray:
    """Blend logits on reranker candidates; retain sparse scores everywhere else."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    candidate = reranker > 0
    sparse_logits = _logit(sparse)
    reranker_logits = _logit(np.maximum(reranker, 1e-6))
    fused_logits = sparse_logits.copy()
    fused_logits[candidate] += alpha * (reranker_logits[candidate] - sparse_logits[candidate])
    return _sigmoid(fused_logits).astype(np.float32)


def evaluate(seeds: tuple[int, ...] = SEEDS, bootstrap_replicates: int = 1000) -> dict:
    if bootstrap_replicates < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("bootstrap_replicates must be positive and seeds must be unique")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )
    contracts = load_cuad(TRAIN_PATH)
    per_seed, alpha_rows, category_rows, intervals = [], [], [], {}

    for seed in seeds:
        run_dir = _run_dir(seed)
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        if report.get("test_data_used") is not False:
            raise ValueError(f"Seed {seed}: test-data provenance is not explicitly false")
        y_eval, ids, contract_rows = _prepare_evaluation(seed, tokenizer, contracts)
        if {name: hashlib.sha256("\n".join(values).encode()).hexdigest()
            for name, values in ids.items()} != report["split"]["contract_id_sha256"]:
            raise ValueError(f"Seed {seed}: split identity does not match saved report")

        cal = np.load(run_dir / "calibration_predictions.npz")
        evaluation = np.load(run_dir / "evaluation_predictions.npz")
        if not np.array_equal(y_eval, evaluation["y_true"]):
            raise ValueError(f"Seed {seed}: evaluation labels do not align with saved scores")
        sparse_cal, cross_cal = cal["retriever_score"], cal["reranker_score"]
        sparse_eval, cross_eval = evaluation["retriever_score"], evaluation["reranker_score"]
        expected_candidates = report["candidate_generation"]["evaluation_candidate_pairs_scored"]
        if int(np.count_nonzero(cross_eval > 0)) != expected_candidates:
            raise ValueError(f"Seed {seed}: candidate mask cannot be recovered exactly from saved scores")
        cal_aupr = {}
        fused_cal = {}
        fused_eval = {}
        for alpha in ALPHA_GRID:
            fused_cal[alpha] = _fuse(sparse_cal, cross_cal, alpha)
            fused_eval[alpha] = _fuse(sparse_eval, cross_eval, alpha)
            cal_aupr[alpha] = _macro_aupr(cal["y_true"], fused_cal[alpha])
            alpha_rows.append({
                "seed": seed, "alpha": alpha,
                "calibration_macro_aupr": cal_aupr[alpha],
                "evaluation_macro_f1_diagnostic": _macro_f1(
                    y_eval, fused_eval[alpha],
                    _threshold_vector(calibrate_thresholds(cal["y_true"], fused_cal[alpha])),
                ),
                "evaluation_macro_aupr_diagnostic": _macro_aupr(y_eval, fused_eval[alpha]),
            })

        # Select only on calibration AUPR; ties choose the smallest weight to favor TF-IDF.
        chosen_alpha = max(ALPHA_GRID, key=lambda alpha: (cal_aupr[alpha], -alpha))
        chosen_cal = fused_cal[chosen_alpha]
        chosen_eval = fused_eval[chosen_alpha]
        thresholds = calibrate_thresholds(cal["y_true"], chosen_cal)
        baseline_thresholds = calibrate_thresholds(cal["y_true"], sparse_cal)
        reranker_thresholds = calibrate_thresholds(cal["y_true"], cross_cal)
        baseline_vector = _threshold_vector(baseline_thresholds)
        hybrid_vector = _threshold_vector(thresholds)
        reranker_vector = _threshold_vector(reranker_thresholds)
        base_f1 = _macro_f1(y_eval, sparse_eval, baseline_vector)
        hybrid_f1 = _macro_f1(y_eval, chosen_eval, hybrid_vector)
        reranker_f1 = _macro_f1(y_eval, cross_eval, reranker_vector)
        base_aupr = _macro_aupr(y_eval, sparse_eval)
        hybrid_aupr = _macro_aupr(y_eval, chosen_eval)
        reranker_aupr = _macro_aupr(y_eval, cross_eval)
        candidate_fraction = float(np.mean(cross_eval > 0))
        per_seed.append({
            "seed": seed, "selected_alpha": chosen_alpha,
            "selected_calibration_macro_aupr": cal_aupr[chosen_alpha],
            "reranked_score_fraction": candidate_fraction,
            "tfidf_macro_f1": base_f1, "hybrid_macro_f1": hybrid_f1,
            "standalone_reranker_macro_f1": reranker_f1,
            "hybrid_delta_f1": hybrid_f1 - base_f1,
            "tfidf_macro_aupr": base_aupr, "hybrid_macro_aupr": hybrid_aupr,
            "standalone_reranker_macro_aupr": reranker_aupr,
            "hybrid_delta_aupr": hybrid_aupr - base_aupr,
        })

        base_frame = _metrics_at_thresholds(y_eval, sparse_eval, baseline_thresholds).set_index("category")
        hybrid_frame = _metrics_at_thresholds(y_eval, chosen_eval, thresholds).set_index("category")
        for category in CATEGORIES:
            category_rows.append({
                "seed": seed, "category": category,
                "support": int(base_frame.loc[category, "support"]),
                "tfidf_f1": float(base_frame.loc[category, "f1"]),
                "hybrid_f1": float(hybrid_frame.loc[category, "f1"]),
                "delta_f1": float(hybrid_frame.loc[category, "f1"] - base_frame.loc[category, "f1"]),
                "tfidf_aupr": float(base_frame.loc[category, "aupr"]),
                "hybrid_aupr": float(hybrid_frame.loc[category, "aupr"]),
                "delta_aupr": float(hybrid_frame.loc[category, "aupr"] - base_frame.loc[category, "aupr"]),
            })
        intervals[str(seed)] = _bootstrap_delta(
            y_eval, sparse_eval, chosen_eval, baseline_vector, hybrid_vector,
            contract_rows, seed, bootstrap_replicates,
        )

    frame = pd.DataFrame(per_seed)
    aggregate = {
        metric: {"mean": float(frame[metric].mean()),
                 "std_between_seeds": float(frame[metric].std(ddof=1))}
        for metric in ("tfidf_macro_f1", "hybrid_macro_f1", "hybrid_delta_f1",
                       "tfidf_macro_aupr", "hybrid_macro_aupr", "hybrid_delta_aupr")
    }
    aggregate["f1_wins"] = int((frame.hybrid_delta_f1 > 0).sum())
    aggregate["aupr_wins"] = int((frame.hybrid_delta_aupr > 0).sum())
    result = {
        "seeds": list(seeds), "alpha_grid": list(ALPHA_GRID),
        "bootstrap_replicates_per_seed": bootstrap_replicates,
        "selection": "one global alpha selected by calibration macro-AUPR; ties favor smaller alpha",
        "thresholds": "per-category thresholds fit on the same calibration partition",
        "fusion": "logit interpolation on reranker candidates; TF-IDF probability retained off-candidate",
        "macro_f1": "fixed 41-category macro; zero-support categories contribute F1=0",
        "macro_aupr": "mean over categories with at least one evaluation positive",
        "paired_contract_bootstrap_95pct": intervals,
        "aggregate": aggregate,
        "per_seed": per_seed,
        "evaluation_status": "exploratory only; split results were inspected in prior experiments",
        "test_data_used": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "per_seed.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(OUTPUT / "alpha_grid.csv", index=False)
    pd.DataFrame(category_rows).to_csv(OUTPUT / "per_category_by_seed.csv", index=False)
    (OUTPUT / "paired_bootstrap.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "summary.json").write_text(json.dumps({
        "aggregate": aggregate, "selection": result["selection"],
        "evaluation_status": result["evaluation_status"], "seeds": list(seeds),
    }, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(evaluate(tuple(args.seeds), args.bootstrap_replicates), indent=2))


if __name__ == "__main__":
    main()
