"""Compare word+character TF-IDF with the word-only baseline on a fixed split.

Reads only the official training file, splits contracts before chunking, tunes
per-category thresholds on validation predictions, and never loads test.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.metrics import compute_base_metrics
from src.models.naive.tfid_baseline import fit_tfidf_baseline
from src.models.naive.word_tfidf import fit_word_char_tfidf, predict_word_char_probabilities
from src.pipeline import prepare_train_val, to_model_inputs
from src.thresholds import apply_thresholds, find_best_thresholds

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/cuad/train_separate_questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/word_char_tfidf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()

    contracts = load_cuad(args.data)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )
    train_chunks, val_chunks = prepare_train_val(
        contracts, tokenizer, val_fraction=args.val_fraction, seed=args.seed,
        max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens,
    )
    train_texts, y_train = to_model_inputs(train_chunks)
    val_texts, y_val = to_model_inputs(val_chunks)

    # Both approaches see exactly the same shared train/validation chunks.
    baseline_vectorizer, baseline = fit_tfidf_baseline(train_texts, y_train)
    baseline_probs = baseline.predict_proba(baseline_vectorizer.transform(val_texts))
    features, candidate = fit_word_char_tfidf(train_texts, y_train, c=args.c)
    candidate_probs = predict_word_char_probabilities(features, candidate, val_texts)

    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, probs in (("word_tfidf", baseline_probs), ("word_char_tfidf", candidate_probs)):
        thresholds = find_best_thresholds(y_val, probs, CATEGORIES)
        predictions = apply_thresholds(probs, CATEGORIES, thresholds)
        metrics = compute_base_metrics(y_val, predictions, CATEGORIES)
        metrics.insert(0, "model", name)
        summary.append(metrics)
        thresholds_path = args.output / f"{name}_thresholds.json"
        thresholds_path.write_text(
            json.dumps({key: float(value) for key, value in thresholds.items()}, indent=2) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(args.output / f"{name}_validation.npz", y_true=y_val, y_prob=probs)

    import pandas as pd
    combined = pd.concat(summary, ignore_index=True)
    combined.to_csv(args.output / "per_category_metrics.csv", index=False)
    macro = combined.groupby("model")["f1"].mean().to_dict()
    report = {
        "experiment": "word_char_tfidf",
        "data": str(args.data.relative_to(ROOT) if args.data.is_relative_to(ROOT) else args.data),
        "split": {"method": "split_by_contract", "seed": args.seed, "val_fraction": args.val_fraction,
                  "train_contracts": len({x.chunk.contract_id for x in train_chunks}),
                  "validation_contracts": len({x.chunk.contract_id for x in val_chunks})},
        "chunking": {"method": "chunk_sentence_windows", "tokenizer": TOKENIZER_NAME,
                     "tokenizer_revision": TOKENIZER_REVISION, "max_tokens": args.max_tokens,
                     "overlap_tokens": args.overlap_tokens},
        "training": {"c": args.c, "word_features": "1-2 grams, sublinear TF, max 100000",
                     "character_features": "char_wb 3-5 grams, min_df=2, sublinear TF, max 100000",
                     "class_weight": "balanced"},
        "thresholds_tuned_on": "validation (optimistic; use a separate calibration split for final reporting)",
        "macro_f1_descriptive": macro,
        "beats_word_tfidf_on_validation_macro_f1": macro["word_char_tfidf"] > macro["word_tfidf"],
        "test_data_used": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
