"""Compare word + character TF-IDF against word-only with contract-level splits.

Uses only the official CUAD training file. The 20% final evaluation contracts
are never used for threshold calibration; the remaining 80% are split into model
training and threshold calibration contracts. test.json is never loaded.
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
from src.labels.loader import Contract, load_cuad
from src.labels.splits import split_by_contract
from src.metrics import compute_base_metrics
from src.models.naive.tfid_baseline import fit_tfidf_baseline
from src.models.naive.word_tfidf import fit_word_char_tfidf, predict_word_char_probabilities
from src.pipeline import prepare_contracts, to_model_inputs
from src.thresholds import apply_thresholds, find_best_thresholds

ROOT = Path(__file__).resolve().parents[1]


def _ids_sha256(ids: list[str]) -> str:
    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prepare(contracts: list[Contract], tokenizer, args):
    chunks = prepare_contracts(
        contracts, tokenizer, max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    return to_model_inputs(chunks), chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/cuad/train_separate_questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/word_char_tfidf_nested")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-fraction", type=float, default=0.2,
                        help="Fraction of the non-evaluation contracts reserved for threshold calibration")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 < args.evaluation_fraction < 1 or not 0 < args.calibration_fraction < 1:
        parser.error("evaluation and calibration fractions must be between 0 and 1")

    contracts = load_cuad(args.data)
    development_contracts, evaluation_contracts = split_by_contract(
        contracts, val_fraction=args.evaluation_fraction, seed=args.seed,
    )
    training_contracts, calibration_contracts = split_by_contract(
        development_contracts, val_fraction=args.calibration_fraction, seed=args.seed,
    )
    ids = {
        "training": [x.contract_id for x in training_contracts],
        "calibration": [x.contract_id for x in calibration_contracts],
        "evaluation": [x.contract_id for x in evaluation_contracts],
    }
    assert not (set(ids["training"]) & set(ids["calibration"]))
    assert not (set(ids["training"]) & set(ids["evaluation"]))
    assert not (set(ids["calibration"]) & set(ids["evaluation"]))

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )
    (train_texts, y_train), train_chunks = _prepare(training_contracts, tokenizer, args)
    (cal_texts, y_cal), calibration_chunks = _prepare(calibration_contracts, tokenizer, args)
    (eval_texts, y_eval), evaluation_chunks = _prepare(evaluation_contracts, tokenizer, args)

    models = {}
    baseline_vectorizer, baseline = fit_tfidf_baseline(train_texts, y_train)
    models["word_tfidf"] = (
        lambda texts: baseline.predict_proba(baseline_vectorizer.transform(texts)),
        {"vectorizer": baseline_vectorizer, "classifier": baseline},
    )
    features, hybrid = fit_word_char_tfidf(train_texts, y_train, c=args.c)
    models["word_char_tfidf"] = (
        lambda texts: predict_word_char_probabilities(features, hybrid, texts),
        {"features": features, "classifier": hybrid},
    )

    args.output.mkdir(parents=True, exist_ok=True)
    metric_frames = []
    calibration_thresholds = {}
    for name, (predict, _) in models.items():
        calibration_probs = predict(cal_texts)
        thresholds = find_best_thresholds(y_cal, calibration_probs, CATEGORIES)
        calibration_thresholds[name] = thresholds
        evaluation_probs = predict(eval_texts)
        predictions = apply_thresholds(evaluation_probs, CATEGORIES, thresholds)
        metrics = compute_base_metrics(y_eval, predictions, CATEGORIES)
        metrics.insert(0, "model", name)
        metric_frames.append(metrics)
        np.savez_compressed(
            args.output / f"{name}_evaluation.npz", y_true=y_eval, y_prob=evaluation_probs,
        )
        np.savez_compressed(
            args.output / f"{name}_calibration.npz", y_true=y_cal, y_prob=calibration_probs,
        )
        (args.output / f"{name}_thresholds.json").write_text(
            json.dumps({key: float(value) for key, value in thresholds.items()}, indent=2) + "\n",
            encoding="utf-8",
        )

    results = pd.concat(metric_frames, ignore_index=True)
    results.to_csv(args.output / "per_category_metrics.csv", index=False)
    macro_f1 = results.groupby("model")["f1"].mean().to_dict()
    report = {
        "experiment": "word_char_tfidf_nested_holdout",
        "data": str(args.data.relative_to(ROOT) if args.data.is_relative_to(ROOT) else args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "split": {
            "method": "split_by_contract",
            "seed": args.seed,
            "evaluation_fraction": args.evaluation_fraction,
            "calibration_fraction_of_non_evaluation": args.calibration_fraction,
            "contract_counts": {name: len(group) for name, group in (
                ("training", training_contracts), ("calibration", calibration_contracts),
                ("evaluation", evaluation_contracts))},
            "contract_id_sha256": {name: _ids_sha256(group) for name, group in ids.items()},
            "chunk_counts": {"training": len(train_chunks), "calibration": len(calibration_chunks),
                             "evaluation": len(evaluation_chunks)},
        },
        "chunking": {"method": "chunk_sentence_windows", "tokenizer": TOKENIZER_NAME,
                     "tokenizer_revision": TOKENIZER_REVISION, "max_tokens": args.max_tokens,
                     "overlap_tokens": args.overlap_tokens},
        "training": {"c": args.c, "word_features": "1-2 grams, sublinear TF, max 100000",
                     "character_features": "char_wb 3-5 grams, min_df=2, sublinear TF, max 100000",
                     "class_weight": "balanced"},
        "thresholds_fit_on": "calibration contracts only",
        "metrics_on": "held-out evaluation contracts, never used for model/threshold fitting",
        "macro_f1_evaluation": macro_f1,
        "word_char_beats_word_macro_f1": macro_f1["word_char_tfidf"] > macro_f1["word_tfidf"],
        "test_data_used": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
