"""Fit the seed-42 TF-IDF control and register its reproducible local bundle.

This registers a research artifact in the local SQLite/MLflow registry. It does not
promote the model to a production/default status and never reads data/cuad/test.json.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from transformers import AutoTokenizer

from scripts.analyze_reranker_multiseed import _macro_aupr, _macro_f1, _threshold_vector
from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.labels.splits import split_by_contract
from src.models.encoder.experiment_utils import ROOT, TRAIN_PATH, calibrate_thresholds, digest_contract_ids
from src.models.naive.word_tfidf import fit_word_char_tfidf, predict_word_char_probabilities
from src.pipeline import prepare_contracts, to_model_inputs
from src.registry.bundle import ClauseModelBundle
from src.registry.mlflow_registry import register_bundle
from src.registry.predictors import VectorizedClassifier

TRACKING_URI = ROOT / "outputs/mlflow/tracking.db"
REGISTRY_NAME = "clause-classifier-word-char-tfidf"


def register() -> dict:
    contracts = load_cuad(TRAIN_PATH)
    development, evaluation = split_by_contract(contracts, val_fraction=0.2, seed=42)
    training, calibration = split_by_contract(development, val_fraction=0.2, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )

    def tensors(groups):
        chunks = prepare_contracts(groups, tokenizer, max_tokens=512, overlap_tokens=64)
        return to_model_inputs(chunks)

    train_texts, y_train = tensors(training)
    calibration_texts, y_cal = tensors(calibration)
    evaluation_texts, y_eval = tensors(evaluation)
    features, classifier = fit_word_char_tfidf(train_texts, y_train, c=1.0)
    cal_scores = predict_word_char_probabilities(features, classifier, calibration_texts)
    thresholds = calibrate_thresholds(y_cal, cal_scores)
    eval_scores = predict_word_char_probabilities(features, classifier, evaluation_texts)
    threshold_vector = _threshold_vector(thresholds)
    metrics = {
        "evaluation_macro_f1_diagnostic": _macro_f1(y_eval, eval_scores, threshold_vector),
        "evaluation_macro_aupr_diagnostic": _macro_aupr(y_eval, eval_scores),
    }
    ids = {
        "training": [contract.contract_id for contract in training],
        "calibration": [contract.contract_id for contract in calibration],
        "evaluation": [contract.contract_id for contract in evaluation],
    }
    bundle = ClauseModelBundle(
        predictor=VectorizedClassifier(features, classifier),
        categories=CATEGORIES,
        thresholds=thresholds,
        model_name="word_char_tfidf_seed42",
        config={
            "c": 1.0, "word_ngrams": [1, 2], "char_wb_ngrams": [3, 5],
            "word_max_features": 100000, "char_max_features": 100000,
            "class_weight": "balanced", "chunker": "sentence_token",
            "max_tokens": 512, "overlap_tokens": 64,
            "tokenizer": TOKENIZER_NAME, "tokenizer_revision": TOKENIZER_REVISION,
        },
        seed=42,
    )
    data_sha256 = hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()
    metadata = {
        "purpose": "research baseline registration; not production promotion",
        "data_path": str(TRAIN_PATH.relative_to(ROOT)), "data_sha256": data_sha256,
        "split_method": "seeded contract-level nested split",
        "split_seed": 42,
        "contract_counts": {name: len(group) for name, group in (
            ("training", training), ("calibration", calibration), ("evaluation", evaluation),
        )},
        "contract_id_sha256": {name: digest_contract_ids(group) for name, group in ids.items()},
        "thresholds_fit_on": "calibration contracts only",
        "evaluation_metrics": "diagnostic only; evaluation partitions have prior research exposure",
        "test_data_used": False,
        "category_schema_sha256": hashlib.sha256("\n".join(CATEGORIES).encode()).hexdigest(),
    }
    registered = register_bundle(
        bundle,
        experiment_name="clause-review-tfidf-baseline",
        registry_name=REGISTRY_NAME,
        metrics=metrics,
        metadata=metadata,
        tracking_uri=TRACKING_URI,
    )
    # Fail closed on serialization or registry-load drift.
    roundtrip = registered.predict_proba([evaluation_texts[0]])
    if roundtrip.shape != (1, len(CATEGORIES)) or not np.isfinite(roundtrip).all():
        raise RuntimeError("registered TF-IDF bundle failed the round-trip inference smoke test")
    return {
        "registry_name": registered.registry_name,
        "model_version": registered.model_version,
        "model_key": registered.model_key,
        "run_id": registered.run_id,
        "tracking_uri": str(TRACKING_URI.relative_to(ROOT)),
        "metrics": metrics,
        "metadata": metadata,
        "test_data_used": False,
    }


def main() -> None:
    print(json.dumps(register(), indent=2))


if __name__ == "__main__":
    main()
