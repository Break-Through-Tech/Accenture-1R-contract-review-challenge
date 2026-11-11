"""Aggregate predeclared reranker runs and paired contract-bootstrap uncertainty.

This is a reporting-only script. It reads saved predictions from CUAD training-file
splits, validates their provenance and calibration thresholds, and never reads test.json
or changes model/threshold choices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer

from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.labels.splits import split_by_contract
from src.models.encoder.experiment_utils import ROOT, TRAIN_PATH, calibrate_thresholds, digest_contract_ids
from src.pipeline import prepare_contracts, to_model_inputs
from scripts.train_retrieval_reranker import _metrics_at_thresholds

SEEDS = (42, 7, 17, 23, 101)
OUTPUT = ROOT / "results/retrieval_reranker_multiseed"


def _run_dir(seed: int) -> Path:
    return ROOT / ("results/retrieval_reranker_seed42" if seed == 42
                   else f"results/retrieval_reranker_seed{seed}")


def _baseline_dir(seed: int) -> Path:
    return ROOT / ("results/word_char_tfidf_nested" if seed == 42
                   else f"results/word_char_tfidf_nested_seed{seed}")


def _threshold_vector(values: dict[str, float]) -> np.ndarray:
    return np.asarray([values[category] for category in CATEGORIES], dtype=np.float32)


def _macro_f1(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> float:
    predicted = scores >= thresholds[None, :]
    true_positive = np.sum((y_true == 1) & predicted, axis=0)
    false_positive = np.sum((y_true == 0) & predicted, axis=0)
    false_negative = np.sum((y_true == 1) & ~predicted, axis=0)
    denominator = 2 * true_positive + false_positive + false_negative
    per_category = np.divide(
        2 * true_positive, denominator,
        out=np.zeros(len(CATEGORIES), dtype=np.float64), where=denominator != 0,
    )
    # Fixed 41-category macro convention: unsupported classes contribute zero.
    return float(np.mean(per_category))


def _macro_aupr(y_true: np.ndarray, scores: np.ndarray) -> float:
    active = np.any(y_true == 1, axis=0)
    if not np.any(active):
        return float("nan")
    return float(np.mean(average_precision_score(y_true[:, active], scores[:, active], average=None)))


def _prepare_evaluation(seed: int, tokenizer, contracts):
    development, evaluation = split_by_contract(contracts, val_fraction=0.2, seed=seed)
    training, calibration = split_by_contract(development, val_fraction=0.2, seed=seed)
    eval_chunks = prepare_contracts(evaluation, tokenizer, max_tokens=512, overlap_tokens=64)
    (_, y_true) = to_model_inputs(eval_chunks)
    ids = {
        "training": [contract.contract_id for contract in training],
        "calibration": [contract.contract_id for contract in calibration],
        "evaluation": [contract.contract_id for contract in evaluation],
    }
    groups = [
        np.flatnonzero(np.asarray([chunk.chunk.contract_id for chunk in eval_chunks]) == contract_id)
        for contract_id in ids["evaluation"]
    ]
    if any(len(rows) == 0 for rows in groups):
        raise ValueError(f"Seed {seed}: an evaluation contract has no chunks")
    return y_true.astype(np.uint8), ids, groups


def _bootstrap_delta(
    y_true: np.ndarray, baseline_scores: np.ndarray, reranker_scores: np.ndarray,
    baseline_thresholds: np.ndarray, reranker_thresholds: np.ndarray,
    contract_rows: list[np.ndarray], seed: int, replicates: int,
) -> dict[str, dict[str, float]]:
    """Paired cluster bootstrap, resampling evaluation contracts with replacement."""
    rng = np.random.default_rng(20261105 + seed)
    f1_deltas = np.empty(replicates, dtype=np.float64)
    aupr_deltas = np.empty(replicates, dtype=np.float64)
    n_contracts = len(contract_rows)
    for replicate in range(replicates):
        selected = rng.integers(0, n_contracts, size=n_contracts)
        rows = np.concatenate([contract_rows[index] for index in selected])
        labels = y_true[rows]
        f1_deltas[replicate] = (
            _macro_f1(labels, reranker_scores[rows], reranker_thresholds)
            - _macro_f1(labels, baseline_scores[rows], baseline_thresholds)
        )
        aupr_deltas[replicate] = (
            _macro_aupr(labels, reranker_scores[rows])
            - _macro_aupr(labels, baseline_scores[rows])
        )
    return {
        "macro_f1_delta": {
            "lower_95": float(np.quantile(f1_deltas, 0.025)),
            "median": float(np.quantile(f1_deltas, 0.5)),
            "upper_95": float(np.quantile(f1_deltas, 0.975)),
        },
        "macro_aupr_delta": {
            "lower_95": float(np.quantile(aupr_deltas, 0.025)),
            "median": float(np.quantile(aupr_deltas, 0.5)),
            "upper_95": float(np.quantile(aupr_deltas, 0.975)),
        },
    }


def analyze(seeds: tuple[int, ...] = SEEDS, replicates: int = 1000) -> dict:
    if replicates < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("replicates must be positive and seeds must be unique")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )
    contracts = load_cuad(TRAIN_PATH)
    per_seed, per_category, intervals = [], [], {}
    data_sha256 = None

    for seed in seeds:
        run_dir = _run_dir(seed)
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        if report.get("test_data_used") is not False:
            raise ValueError(f"Seed {seed}: test-set provenance is not explicitly false")
        if report["split"]["seed"] != seed:
            raise ValueError(f"Seed {seed}: report seed does not match directory")
        if report["chunking"] != {
            "tokenizer": TOKENIZER_NAME, "tokenizer_revision": TOKENIZER_REVISION,
            "max_tokens": 512, "overlap_tokens": 64,
        }:
            raise ValueError(f"Seed {seed}: chunking settings differ")
        data_sha256 = data_sha256 or report["data_sha256"]
        if report["data_sha256"] != data_sha256:
            raise ValueError("Runs were produced from different training data")

        y_true, ids, contract_rows = _prepare_evaluation(seed, tokenizer, contracts)
        split_hashes = {name: digest_contract_ids(values) for name, values in ids.items()}
        if split_hashes != report["split"]["contract_id_sha256"]:
            raise ValueError(f"Seed {seed}: contract split hashes do not match saved report")

        predictions = np.load(run_dir / "evaluation_predictions.npz")
        calibration = np.load(run_dir / "calibration_predictions.npz")
        if not np.array_equal(y_true, predictions["y_true"]):
            raise ValueError(f"Seed {seed}: reproduced evaluation labels differ from saved predictions")
        thresholds = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
        expected_baseline = calibrate_thresholds(calibration["y_true"], calibration["retriever_score"])
        expected_reranker = calibrate_thresholds(calibration["y_true"], calibration["reranker_score"])
        for model, expected in (("retriever", expected_baseline), ("reranker", expected_reranker)):
            if any(not np.isclose(thresholds[model][category], expected[category])
                   for category in CATEGORIES):
                raise ValueError(f"Seed {seed}: {model} thresholds do not match calibration predictions")

        # Confirm the same raw sparse model scores appear in the independent baseline run.
        baseline_path = _baseline_dir(seed) / "word_char_tfidf_evaluation.npz"
        independent = np.load(baseline_path)
        if (not np.array_equal(independent["y_true"], y_true) or
                not np.allclose(independent["y_prob"], predictions["retriever_score"], atol=0, rtol=0)):
            raise ValueError(f"Seed {seed}: TF-IDF control scores are not exactly aligned")

        baseline_scores = predictions["retriever_score"]
        reranker_scores = predictions["reranker_score"]
        baseline_thresholds = _threshold_vector(thresholds["retriever"])
        reranker_thresholds = _threshold_vector(thresholds["reranker"])
        baseline_f1 = _macro_f1(y_true, baseline_scores, baseline_thresholds)
        reranker_f1 = _macro_f1(y_true, reranker_scores, reranker_thresholds)
        baseline_aupr = _macro_aupr(y_true, baseline_scores)
        reranker_aupr = _macro_aupr(y_true, reranker_scores)
        per_seed.append({
            "seed": seed, "evaluation_contracts": len(contract_rows),
            "word_char_f1": baseline_f1, "reranker_f1": reranker_f1,
            "delta_f1": reranker_f1 - baseline_f1,
            "word_char_aupr": baseline_aupr, "reranker_aupr": reranker_aupr,
            "delta_aupr": reranker_aupr - baseline_aupr,
            "candidate_pair_recall_at_16": report["candidate_generation"]["candidate_recall_at_k"]["16"]["pair_recall"],
            "contract_category_recall_at_16": report["candidate_generation"]["candidate_recall_at_k"]["16"]["contract_category_recall"],
            "candidate_pairs_scored": report["candidate_generation"]["evaluation_candidate_pairs_scored"],
        })

        baseline_frame = _metrics_at_thresholds(y_true, baseline_scores, thresholds["retriever"])
        reranker_frame = _metrics_at_thresholds(y_true, reranker_scores, thresholds["reranker"])
        baseline_frame = baseline_frame.set_index("category")
        reranker_frame = reranker_frame.set_index("category")
        for category in CATEGORIES:
            per_category.append({
                "seed": seed, "category": category,
                "support": int(baseline_frame.loc[category, "support"]),
                "word_char_f1": float(baseline_frame.loc[category, "f1"]),
                "reranker_f1": float(reranker_frame.loc[category, "f1"]),
                "delta_f1": float(reranker_frame.loc[category, "f1"] - baseline_frame.loc[category, "f1"]),
                "word_char_aupr": float(baseline_frame.loc[category, "aupr"]),
                "reranker_aupr": float(reranker_frame.loc[category, "aupr"]),
                "delta_aupr": float(reranker_frame.loc[category, "aupr"] - baseline_frame.loc[category, "aupr"]),
            })

        intervals[str(seed)] = _bootstrap_delta(
            y_true, baseline_scores, reranker_scores, baseline_thresholds,
            reranker_thresholds, contract_rows, seed, replicates,
        )

    seed_frame = pd.DataFrame(per_seed)
    aggregate = {}
    for metric in ("word_char_f1", "reranker_f1", "delta_f1",
                   "word_char_aupr", "reranker_aupr", "delta_aupr"):
        aggregate[metric] = {
            "mean": float(seed_frame[metric].mean()),
            "std_between_seeds": float(seed_frame[metric].std(ddof=1)),
        }
    aggregate["f1_wins"] = int((seed_frame["delta_f1"] > 0).sum())
    aggregate["aupr_wins"] = int((seed_frame["delta_aupr"] > 0).sum())
    result = {
        "seeds": list(seeds), "bootstrap_replicates_per_seed": replicates,
        "bootstrap_seed_rule": "numpy default_rng(20261105 + split_seed)",
        "data_sha256": data_sha256,
        "protocol": {
            "splits": "same saved contract-level train/calibration/evaluation IDs as each paired TF-IDF run",
            "thresholds": "both systems fit per-category thresholds on calibration predictions only",
            "macro_f1": "fixed 41-category macro; zero-support categories contribute F1=0",
            "macro_aupr": "mean over categories with at least one evaluation positive",
            "bootstrap": "paired resampling of evaluation contracts with replacement; thresholds stay fixed",
            "test_data_used": False,
            "interpretation": "five-seed summary is descriptive; seeded evaluation sets overlap and are not independent",
        },
        "aggregate": aggregate,
        "per_seed": per_seed,
        "paired_contract_bootstrap_95pct": intervals,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_frame.to_csv(OUTPUT / "per_seed.csv", index=False)
    pd.DataFrame(per_category).to_csv(OUTPUT / "per_category_by_seed.csv", index=False)
    (OUTPUT / "paired_bootstrap.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "summary.json").write_text(json.dumps({
        "aggregate": aggregate, "protocol": result["protocol"], "seeds": list(seeds),
    }, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(analyze(tuple(args.seeds), args.bootstrap_replicates), indent=2))


if __name__ == "__main__":
    main()
