"""Compare three additional clause-classification architectures on a fixed nested split.

Adds neighboring-chunk context, label-dependency classifier chains, and positive
training-example retrieval. Includes word-only and word+char controls. Contracts are
split before chunking; calibration and evaluation contracts are disjoint. Never reads
CUAD test.json.
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
from src.metrics import compute_aupr, compute_base_metrics
from src.models.naive.classifier_chain import fit_word_char_chain, predict_word_char_chain
from src.models.naive.context_tfidf import add_neighbor_context
from src.models.naive.prototype_retrieval import fit_positive_prototype_retriever
from src.models.naive.tfid_baseline import fit_tfidf_baseline
from src.models.naive.word_tfidf import fit_word_char_tfidf, predict_word_char_probabilities
from src.pipeline import prepare_contracts, to_model_inputs
from src.thresholds import apply_thresholds, find_best_thresholds

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ["word_tfidf", "word_char_tfidf", "context_word_char_tfidf",
               "word_char_classifier_chain", "positive_prototype_retrieval"]


def _digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _thresholds(y_true: np.ndarray, y_scores: np.ndarray) -> dict[str, float]:
    """Tune thresholds only where calibration has positives; otherwise use 0.5."""
    active = y_true.sum(axis=0) > 0
    tuned = find_best_thresholds(y_true[:, active], y_scores[:, active],
                                 [category for category, flag in zip(CATEGORIES, active) if flag])
    return {category: float(tuned.get(category, 0.5)) for category in CATEGORIES}


def _fit_models(train_texts, y_train, train_chunks, *, seed: int):
    """Fit the controls and three alternative models on one common train partition."""
    baseline_vectorizer, baseline = fit_tfidf_baseline(train_texts, y_train)
    hybrid_features, hybrid = fit_word_char_tfidf(train_texts, y_train)
    context_train = add_neighbor_context(train_chunks)
    context_features, context_model = fit_word_char_tfidf(context_train, y_train)
    chain_features, chain = fit_word_char_chain(train_texts, y_train, seed=seed)
    retriever = fit_positive_prototype_retriever(train_texts, y_train)

    return {
        "word_tfidf": (lambda texts, ctx: baseline.predict_proba(baseline_vectorizer.transform(texts)), None),
        "word_char_tfidf": (lambda texts, ctx: predict_word_char_probabilities(hybrid_features, hybrid, texts), None),
        "context_word_char_tfidf": (
            lambda texts, ctx: predict_word_char_probabilities(context_features, context_model, ctx), None),
        "word_char_classifier_chain": (
            lambda texts, ctx: predict_word_char_chain(chain_features, chain, texts), None),
        "positive_prototype_retrieval": (lambda texts, ctx: retriever.predict_proba(texts), None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/cuad/train_separate_questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/architecture_suite_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    args = parser.parse_args()

    contracts = load_cuad(args.data)
    development, evaluation = split_by_contract(contracts, args.evaluation_fraction, args.seed)
    training, calibration = split_by_contract(development, args.calibration_fraction, args.seed)
    groups = {"training": training, "calibration": calibration, "evaluation": evaluation}
    id_sets = [set(contract.contract_id for contract in group) for group in groups.values()]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("contract split leakage detected")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )
    prepared = {}
    for name, group in groups.items():
        chunks = prepare_contracts(group, tokenizer, args.max_tokens, args.overlap_tokens)
        prepared[name] = (to_model_inputs(chunks), chunks)
    (train_texts, y_train), train_chunks = prepared["training"]
    (cal_texts, y_cal), calibration_chunks = prepared["calibration"]
    (eval_texts, y_eval), evaluation_chunks = prepared["evaluation"]
    context_cal = add_neighbor_context(calibration_chunks)
    context_eval = add_neighbor_context(evaluation_chunks)

    models = _fit_models(train_texts, y_train, train_chunks, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    metric_frames = []
    for name, (predict, _) in models.items():
        calibration_scores = predict(cal_texts, context_cal)
        thresholds = _thresholds(y_cal, calibration_scores)
        evaluation_scores = predict(eval_texts, context_eval)
        predictions = apply_thresholds(evaluation_scores, CATEGORIES, thresholds)
        report = compute_base_metrics(y_eval, predictions, CATEGORIES)
        report["aupr"] = compute_aupr(y_eval, evaluation_scores)
        report.insert(0, "model", name)
        metric_frames.append(report)
        (args.output / f"{name}_thresholds.json").write_text(
            json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(args.output / f"{name}_calibration.npz", y_true=y_cal, y_score=calibration_scores)
        np.savez_compressed(args.output / f"{name}_evaluation.npz", y_true=y_eval, y_score=evaluation_scores)

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(args.output / "per_category_metrics.csv", index=False)
    macro = metrics.groupby("model")[["precision", "recall", "f1", "aupr"]].mean().to_dict("index")
    ids = {name: [contract.contract_id for contract in group] for name, group in groups.items()}
    report = {
        "experiment": "three_additional_architectures",
        "architectures": {
            "context_word_char_tfidf": "word+char TF-IDF on current chunk plus adjacent chunk texts; center-chunk labels",
            "word_char_classifier_chain": "random-order classifier chain over word+char TF-IDF features",
            "positive_prototype_retrieval": "max cosine similarity to category-positive train chunks in word+char TF-IDF space",
            "controls": ["word_tfidf", "word_char_tfidf"],
        },
        "split": {"seed": args.seed, "contract_counts": {name: len(group) for name, group in groups.items()},
                  "contract_id_sha256": {name: _digest(value) for name, value in ids.items()},
                  "chunk_counts": {name: len(prepared[name][1]) for name in groups}},
        "chunking": {"method": "sentence_token", "tokenizer": TOKENIZER_NAME,
                     "tokenizer_revision": TOKENIZER_REVISION, "max_tokens": args.max_tokens,
                     "overlap_tokens": args.overlap_tokens},
        "thresholds_fit_on": "calibration contracts only",
        "metrics_on": "held-out evaluation contracts",
        "macro_metrics": macro,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "test_data_used": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
