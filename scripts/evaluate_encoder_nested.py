"""Train and evaluate DistilRoBERTa on the same nested CUAD split as hybrid TF-IDF.

Training uses 260 contracts, threshold calibration uses 66 contracts, and final
validation uses the disjoint 82-contract evaluation partition for seed 42. It reads
only train_separate_questions.json; test.json remains reserved and untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.labels.splits import split_by_contract
from src.metrics import compute_base_metrics, compute_aupr
from src.models.encoder.config import load_config
from src.models.encoder.evaluate import tune_thresholds
from src.models.encoder.predict import load_checkpoint, predict_proba
from src.models.encoder.train import fit
from src.pipeline import prepare_contracts, to_model_inputs
from src.registry.bundle import ClauseModelBundle
from src.registry.mlflow_registry import register_bundle
from src.registry.predictors import EncoderPredictor
from src.thresholds import apply_thresholds

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/cuad/train_separate_questions.json"


def _split_hash(contract_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(contract_ids).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/encoder.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/encoder_nested")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--register", action="store_true", help="Register the trained checkpoint in local MLflow")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.chunking_method != "sentence_token":
        parser.error("nested comparison requires chunking_method=sentence_token")
    if config.model_name != TOKENIZER_NAME or config.model_revision != TOKENIZER_REVISION:
        parser.error("encoder and shared chunker must use the pinned common DistilRoBERTa revision")

    contracts = load_cuad(TRAIN_PATH)
    development, evaluation = split_by_contract(
        contracts, val_fraction=args.evaluation_fraction, seed=args.seed,
    )
    training, calibration = split_by_contract(
        development, val_fraction=args.calibration_fraction, seed=args.seed,
    )
    groups = {"training": training, "calibration": calibration, "evaluation": evaluation}
    sets = [set(contract.contract_id for contract in group) for group in groups.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("contract split leakage detected")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )
    prepared = {
        name: to_model_inputs(prepare_contracts(group, tokenizer, config.max_length, config.overlap_tokens))
        for name, group in groups.items()
    }
    (train_texts, train_labels), train_chunks = prepared["training"]
    (cal_texts, cal_labels), calibration_chunks = prepared["calibration"]
    (eval_texts, eval_labels), evaluation_chunks = prepared["evaluation"]

    args.output.mkdir(parents=True, exist_ok=True)
    run_config = config.model_copy(update={"seed": args.seed, "output_dir": args.output / "train"})
    fit(run_config, train_chunks, calibration_chunks)
    checkpoint = args.output / "train/best"
    model, checkpoint_tokenizer = load_checkpoint(checkpoint, device="cpu")
    calibration_probs = predict_proba(
        model, checkpoint_tokenizer, [item.chunk for item in calibration_chunks],
        max_length=config.max_length, batch_size=args.batch_size,
    )
    evaluation_probs = predict_proba(
        model, checkpoint_tokenizer, [item.chunk for item in evaluation_chunks],
        max_length=config.max_length, batch_size=args.batch_size,
    )
    thresholds = tune_thresholds(cal_labels, calibration_probs, CATEGORIES)
    predictions = apply_thresholds(evaluation_probs, CATEGORIES, thresholds)
    report = compute_base_metrics(eval_labels, predictions, CATEGORIES)
    report["aupr"] = compute_aupr(eval_labels, evaluation_probs)
    report["threshold"] = report["category"].map(thresholds)
    report.loc[report["support"] == 0, ["precision", "recall", "f1"]] = np.nan
    report.to_csv(args.output / "per_category_metrics.csv", index=False)
    np.savez_compressed(args.output / "calibration_predictions.npz", y_true=cal_labels, y_prob=calibration_probs)
    np.savez_compressed(args.output / "evaluation_predictions.npz", y_true=eval_labels, y_prob=evaluation_probs)
    (args.output / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")

    metrics = report[["precision", "recall", "f1", "aupr"]].mean(skipna=True).to_dict()
    split = {
        "seed": args.seed,
        "counts": {name: len(group) for name, group in groups.items()},
        "id_sha256": {name: _split_hash([c.contract_id for c in group]) for name, group in groups.items()},
        "chunk_counts": {"training": len(train_chunks), "calibration": len(calibration_chunks),
                         "evaluation": len(evaluation_chunks)},
    }
    summary = {
        "model": "distilroberta_multilabel",
        "split": split,
        "data_sha256": hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        "chunking": {"method": "sentence_token", "max_tokens": config.max_length,
                     "overlap_tokens": config.overlap_tokens, "tokenizer": TOKENIZER_NAME,
                     "tokenizer_revision": TOKENIZER_REVISION},
        "thresholds_fit_on": "66 calibration contracts only",
        "metrics_on": "disjoint 82-contract evaluation partition",
        "macro_metrics": {key: float(value) for key, value in metrics.items()},
        "test_data_used": False,
    }
    (args.output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.register:
        bundle = ClauseModelBundle(
            predictor=EncoderPredictor(model, checkpoint_tokenizer, config.max_length, args.batch_size),
            categories=CATEGORIES, thresholds=thresholds,
            model_name="distilroberta_multilabel",
            config=run_config.model_dump(mode="json"), seed=args.seed,
        )
        registered = register_bundle(
            bundle, experiment_name="clause-review-encoder", registry_name="clause-classifier",
            metrics={f"evaluation_{key}": float(value) for key, value in metrics.items()},
            metadata=summary, tracking_uri=ROOT / "outputs/mlflow/tracking.db",
        )
        summary["registry"] = {"run_id": registered.run_id, "model_version": registered.model_version}
        (args.output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
