"""Generate training-only out-of-fold reranker scores and fit a TF-IDF fusion weight.

The OOF folds are drawn only from the seed-42 training contracts. The official calibration
contracts select fold checkpoint epochs and final category thresholds; evaluation contracts
are scored only after the fusion weight is fixed. This command never reads test.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from transformers import AutoTokenizer

from scripts.analyze_reranker_multiseed import (
    _bootstrap_delta, _macro_aupr, _macro_f1, _prepare_evaluation,
    _run_dir, _threshold_vector,
)
from scripts.evaluate_tfidf_reranker_fusion import ALPHA_GRID, _fuse
from scripts.train_retrieval_reranker import _metrics_at_thresholds
from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.labels.splits import split_by_contract
from src.models.encoder.experiment_utils import (
    ROOT, TRAIN_PATH, calibrate_thresholds, digest_contract_ids,
)

OOF_ROOT = ROOT / "results/reranker_oof_seed42"
FUSION_ROOT = ROOT / "results/reranker_oof_fusion_seed42"
SEED = 42
FOLDS = 5
FOLD_SEED = 20261105


def _base_split(contracts):
    development, evaluation = split_by_contract(contracts, val_fraction=0.2, seed=SEED)
    training, calibration = split_by_contract(development, val_fraction=0.2, seed=SEED)
    return training, calibration, evaluation


def prepare_manifests() -> dict:
    """Create five disjoint OOF manifests over the original seed-42 training contracts."""
    contracts = load_cuad(TRAIN_PATH)
    training, calibration, evaluation = _base_split(contracts)
    source_report = json.loads((_run_dir(SEED) / "report.json").read_text(encoding="utf-8"))
    expected = source_report["split"]["contract_id_sha256"]
    groups = {"training": training, "calibration": calibration, "evaluation": evaluation}
    ids = {name: [contract.contract_id for contract in group] for name, group in groups.items()}
    if {name: digest_contract_ids(values) for name, values in ids.items()} != expected:
        raise ValueError("Seed-42 baseline split does not match saved reranker report")

    # KFold indices are applied to the ordered training IDs, then checked as a partition.
    seen = []
    fold_details = []
    for fold_index, (train_indices, heldout_indices) in enumerate(
        KFold(n_splits=FOLDS, shuffle=True, random_state=FOLD_SEED).split(ids["training"])
    ):
        outer_train = [ids["training"][int(index)] for index in train_indices]
        heldout = [ids["training"][int(index)] for index in heldout_indices]
        manifest = {
            "training": outer_train,
            "calibration": ids["calibration"],
            "evaluation": heldout,
        }
        folder = OOF_ROOT / f"fold_{fold_index}"
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / "split.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        fold_details.append({
            "fold": fold_index, "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": manifest_hash, "heldout_ids": heldout,
            "training_id_sha256": digest_contract_ids(outer_train),
            "heldout_id_sha256": digest_contract_ids(heldout),
        })
        seen.extend(heldout)
    if len(seen) != len(set(seen)) or set(seen) != set(ids["training"]):
        raise RuntimeError("OOF folds do not partition the original training contracts")

    manifest_index = {
        "seed": SEED, "folds": FOLDS, "fold_seed": FOLD_SEED,
        "source_data_sha256": source_report["data_sha256"],
        "source_training_id_sha256": digest_contract_ids(ids["training"]),
        "source_calibration_id_sha256": digest_contract_ids(ids["calibration"]),
        "source_evaluation_id_sha256": digest_contract_ids(ids["evaluation"]),
        "oof_folds": fold_details,
        "protocol": "each training contract is held out once; all outer-fold models share the original 66 calibration contracts; original 82 evaluation contracts are excluded from OOF fitting",
        "test_data_used": False,
    }
    (OOF_ROOT / "manifest_index.json").write_text(
        json.dumps(manifest_index, indent=2) + "\n", encoding="utf-8",
    )
    return manifest_index


def train_folds(manifest_index: dict | None = None) -> None:
    """Train one reranker per OOF fold using the existing reproducible training entry point."""
    if manifest_index is None:
        manifest_index = json.loads((OOF_ROOT / "manifest_index.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for fold in manifest_index["oof_folds"]:
        index = fold["fold"]
        output = OOF_ROOT / f"fold_{index}" / "result"
        checkpoint = ROOT / "outputs/reranker_oof_seed42" / f"fold_{index}" / "best"
        report_path = output / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("split", {}).get("manifest_sha256") == fold["manifest_sha256"]:
                print(f"Fold {index} already complete; skipping.", flush=True)
                continue
        command = [
            sys.executable, "-m", "scripts.train_retrieval_reranker",
            "--split-manifest", str(ROOT / fold["manifest"]),
            "--seed", str(4200 + index),
            "--output", str(output),
            "--checkpoint-dir", str(checkpoint),
        ]
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def _validate_oof_runs(manifest_index: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_labels, all_sparse, all_cross = [], [], []
    seen_ids = []
    for fold in manifest_index["oof_folds"]:
        index = fold["fold"]
        fold_dir = OOF_ROOT / f"fold_{index}"
        manifest_path = ROOT / fold["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result_dir = fold_dir / "result"
        report = json.loads((result_dir / "report.json").read_text(encoding="utf-8"))
        pred = np.load(result_dir / "evaluation_predictions.npz")
        expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if report["split"].get("manifest_sha256") != expected_hash:
            raise ValueError(f"Fold {index}: manifest hash does not match training report")
        if report["test_data_used"] is not False:
            raise ValueError(f"Fold {index}: test-data provenance is not explicitly false")
        if report["split"]["contract_id_sha256"]["evaluation"] != fold["heldout_id_sha256"]:
            raise ValueError(f"Fold {index}: held-out contract IDs differ")
        if int(np.count_nonzero(pred["reranker_score"] > 0)) != report[
            "candidate_generation"]["evaluation_candidate_pairs_scored"]:
            raise ValueError(f"Fold {index}: candidate mask does not match scored pair count")
        if pred["y_true"].shape[0] != report["split"]["chunk_counts"]["evaluation"]:
            raise ValueError(f"Fold {index}: OOF prediction rows do not match chunks")
        all_labels.append(pred["y_true"])
        all_sparse.append(pred["retriever_score"])
        all_cross.append(pred["reranker_score"])
        seen_ids.extend(manifest["evaluation"])
    expected_ids = [contract_id for fold in manifest_index["oof_folds"]
                    for contract_id in fold["heldout_ids"]]
    if len(seen_ids) != len(set(seen_ids)) or set(seen_ids) != set(expected_ids):
        raise ValueError("OOF contract IDs are duplicated or missing")
    return tuple(np.concatenate(values, axis=0) for values in (all_labels, all_sparse, all_cross))


def analyze(bootstrap_replicates: int = 1000) -> dict:
    """Fit one global blend weight on training-only OOF predictions and score seed 42 once."""
    manifest_index = json.loads((OOF_ROOT / "manifest_index.json").read_text(encoding="utf-8"))
    if manifest_index.get("test_data_used") is not False:
        raise ValueError("OOF manifest does not explicitly exclude test data")
    y_oof, sparse_oof, cross_oof = _validate_oof_runs(manifest_index)
    alpha_scores = {
        alpha: _macro_aupr(y_oof, _fuse(sparse_oof, cross_oof, alpha))
        for alpha in ALPHA_GRID
    }
    selected_alpha = max(ALPHA_GRID, key=lambda alpha: (alpha_scores[alpha], -alpha))

    run_dir = _run_dir(SEED)
    calibration = np.load(run_dir / "calibration_predictions.npz")
    evaluation = np.load(run_dir / "evaluation_predictions.npz")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )
    contracts = load_cuad(TRAIN_PATH)
    y_eval, ids, contract_rows = _prepare_evaluation(SEED, tokenizer, contracts)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if {name: digest_contract_ids(values) for name, values in ids.items()} != report[
        "split"]["contract_id_sha256"]:
        raise ValueError("Final seed-42 evaluation split differs from saved report")
    if not np.array_equal(y_eval, evaluation["y_true"]):
        raise ValueError("Final evaluation labels do not match saved scores")

    sparse_cal, cross_cal = calibration["retriever_score"], calibration["reranker_score"]
    sparse_eval, cross_eval = evaluation["retriever_score"], evaluation["reranker_score"]
    fused_cal = _fuse(sparse_cal, cross_cal, selected_alpha)
    fused_eval = _fuse(sparse_eval, cross_eval, selected_alpha)
    baseline_thresholds = calibrate_thresholds(calibration["y_true"], sparse_cal)
    fusion_thresholds = calibrate_thresholds(calibration["y_true"], fused_cal)
    base_vector, fusion_vector = _threshold_vector(baseline_thresholds), _threshold_vector(fusion_thresholds)
    baseline_f1 = _macro_f1(y_eval, sparse_eval, base_vector)
    fusion_f1 = _macro_f1(y_eval, fused_eval, fusion_vector)
    baseline_aupr = _macro_aupr(y_eval, sparse_eval)
    fusion_aupr = _macro_aupr(y_eval, fused_eval)
    bootstrap = _bootstrap_delta(
        y_eval, sparse_eval, fused_eval, base_vector, fusion_vector,
        contract_rows, SEED, bootstrap_replicates,
    )
    base_frame = _metrics_at_thresholds(y_eval, sparse_eval, baseline_thresholds).set_index("category")
    fusion_frame = _metrics_at_thresholds(y_eval, fused_eval, fusion_thresholds).set_index("category")
    per_category = []
    for category in CATEGORIES:
        per_category.append({
            "category": category, "support": int(base_frame.loc[category, "support"]),
            "tfidf_f1": float(base_frame.loc[category, "f1"]),
            "fusion_f1": float(fusion_frame.loc[category, "f1"]),
            "delta_f1": float(fusion_frame.loc[category, "f1"] - base_frame.loc[category, "f1"]),
            "tfidf_aupr": float(base_frame.loc[category, "aupr"]),
            "fusion_aupr": float(fusion_frame.loc[category, "aupr"]),
            "delta_aupr": float(fusion_frame.loc[category, "aupr"] - base_frame.loc[category, "aupr"]),
        })

    result = {
        "data_sha256": report["data_sha256"],
        "seed": SEED,
        "oof_training_contracts": sum(len(fold["heldout_ids"]) for fold in manifest_index["oof_folds"]),
        "oof_training_id_sha256": manifest_index["source_training_id_sha256"],
        "oof_fold_manifest_hashes": [fold["manifest_sha256"] for fold in manifest_index["oof_folds"]],
        "alpha_grid": list(ALPHA_GRID),
        "training_only_oof_macro_aupr_by_alpha": {str(key): value for key, value in alpha_scores.items()},
        "selected_alpha": selected_alpha,
        "alpha_selection": "global alpha selected by macro-AUPR on out-of-fold predictions for the 260 seed-42 training contracts only; calibration/evaluation scores are not compared across alpha values",
        "base_model_checkpoint_selection": "each fold's best epoch selected by macro-AUPR on the separate 66-contract calibration partition",
        "thresholds": "per-category thresholds fit on the same separate 66-contract calibration partition",
        "evaluation": {
            "contracts": len(contract_rows),
            "tfidf_macro_f1": baseline_f1, "fusion_macro_f1": fusion_f1,
            "delta_macro_f1": fusion_f1 - baseline_f1,
            "tfidf_macro_aupr": baseline_aupr, "fusion_macro_aupr": fusion_aupr,
            "delta_macro_aupr": fusion_aupr - baseline_aupr,
            "paired_contract_bootstrap_95pct": bootstrap,
        },
        "per_category": per_category,
        "interpretation": "exploratory only: seed-42 evaluation contracts were inspected in earlier experiments; this does not provide a fresh confirmation",
        "test_data_used": False,
    }
    FUSION_ROOT.mkdir(parents=True, exist_ok=True)
    (FUSION_ROOT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (FUSION_ROOT / "thresholds.json").write_text(json.dumps({
        "tfidf": baseline_thresholds, "fusion": fusion_thresholds,
    }, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(per_category).to_csv(FUSION_ROOT / "per_category_metrics.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "analyze"))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "prepare":
        index = prepare_manifests()
        result = {
            "seed": index["seed"], "folds": index["folds"],
            "source_training_id_sha256": index["source_training_id_sha256"],
            "manifest_hashes": [fold["manifest_sha256"] for fold in index["oof_folds"]],
            "manifest_index": str((OOF_ROOT / "manifest_index.json").relative_to(ROOT)),
        }
    elif args.command == "train":
        train_folds()
        result = {"trained_folds": FOLDS, "output": str(OOF_ROOT.relative_to(ROOT))}
    else:
        result = analyze(args.bootstrap_replicates)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
