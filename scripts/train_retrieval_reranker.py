"""Train/evaluate a sparse retriever plus category-conditioned cross-encoder reranker.

The sparse word/character retriever is fitted on training contracts only. It proposes
up to K chunks per contract/category; a pairwise transformer reranks only those candidates.
The command uses CUAD's training file, calibrates thresholds on calibration contracts,
and never reads test.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.labels.splits import split_by_contract
from src.metrics import compute_aupr, compute_base_metrics
from src.models.encoder.category_descriptions import load_category_descriptions
from src.models.encoder.experiment_utils import (
    ROOT, TRAIN_PATH, calibrate_thresholds, digest_contract_ids,
)
from src.models.naive.retrieval_reranker import (
    PairCrossEncoder, PairDataset, PairExample, build_pair_cross_encoder,
    build_training_pairs, candidate_recall, collate_pairs, dense_candidate_scores,
    retrieve_top_k,
)
from src.models.naive.word_tfidf import fit_word_char_tfidf, predict_word_char_probabilities
from src.pipeline import prepare_contracts, to_model_inputs

DESCRIPTIONS_PATH = ROOT / "data/cuad/category_descriptions.csv"


def _predict_candidates(model, tokenizer, chunks, candidates, prompts, device,
                        batch_size, max_length, amp_enabled):
    if not candidates:
        return np.empty(0, dtype=np.float32)
    examples = [PairExample(
        prompt=prompts[candidate.category_index],
        text=chunks[candidate.chunk_row].chunk.text,
        category_index=candidate.category_index,
        chunk_row=candidate.chunk_row,
        target=0,
    ) for candidate in candidates]
    loader = DataLoader(
        PairDataset(examples), batch_size=batch_size, shuffle=False,
        collate_fn=partial(collate_pairs, tokenizer=tokenizer, max_length=max_length),
    )
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits, _ = model(**batch)
            probabilities.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
    return np.asarray(probabilities, dtype=np.float32)


def _dense_reranker_scores(model, tokenizer, chunks, candidates, prompts, device,
                           batch_size, max_length, amp_enabled):
    pair_probabilities = _predict_candidates(
        model, tokenizer, chunks, candidates, prompts, device,
        batch_size, max_length, amp_enabled,
    )
    return dense_candidate_scores(
        chunks, candidates, pair_probabilities, num_labels=len(CATEGORIES),
    )


def _metrics_at_thresholds(y_true, scores, thresholds):
    vector = np.asarray([thresholds[category] for category in CATEGORIES])
    predictions = (scores >= vector).astype(np.uint8)
    metrics = compute_base_metrics(y_true, predictions, CATEGORIES)
    aupr_by_category = dict(zip(CATEGORIES, compute_aupr(y_true, scores)))
    metrics["aupr"] = metrics["category"].map(aupr_by_category)
    metrics["threshold"] = metrics["category"].map(thresholds)
    metrics.loc[metrics["support"] == 0, ["precision", "recall", "f1"]] = np.nan
    return metrics


def _save_checkpoint(path, model, tokenizer, features, retriever, config):
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(path / "encoder")
    tokenizer.save_pretrained(path / "encoder")
    torch.save(
        {key: value.detach().cpu() for key, value in model.classifier.state_dict().items()},
        path / "cross_encoder_head.pt",
    )
    joblib.dump({"features": features, "classifier": retriever}, path / "retriever.joblib")
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _load_checkpoint(path, device):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(path / "encoder", local_files_only=True)
    encoder = AutoModel.from_pretrained(path / "encoder", local_files_only=True)
    model = PairCrossEncoder(encoder, dropout=config["dropout"])
    head = torch.load(path / "cross_encoder_head.pt", map_location="cpu", weights_only=True)
    model.classifier.load_state_dict(head)
    retriever_bundle = joblib.load(path / "retriever.joblib")
    model.to(device).eval()
    return model, tokenizer, retriever_bundle["features"], retriever_bundle["classifier"], config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/retrieval_reranker.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/retrieval_reranker_seed42")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "outputs/retrieval_reranker_seed42/best")
    parser.add_argument("--evaluate-checkpoint", type=Path,
                        help="Score an existing best/ checkpoint without retraining")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--candidate-top-k", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--overlap-tokens", type=int)
    parser.add_argument("--c", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--negatives-per-positive", type=int)
    parser.add_argument("--min-hard-negatives", type=int)
    parser.add_argument("--evaluation-fraction", type=float)
    parser.add_argument("--calibration-fraction", type=float)
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["model_name"] != TOKENIZER_NAME or config["model_revision"] != TOKENIZER_REVISION:
        parser.error("model_name and model_revision must match the shared pinned tokenizer")
    for field in (
        "seed", "epochs", "learning_rate", "batch_size", "candidate_top_k",
        "max_length", "max_tokens", "overlap_tokens", "c", "dropout",
        "negatives_per_positive", "min_hard_negatives", "evaluation_fraction",
        "calibration_fraction",
    ):
        if getattr(args, field) is None:
            setattr(args, field, config[field])
    if args.epochs < 1 or args.batch_size < 1 or args.candidate_top_k < 1:
        parser.error("epochs, batch size, and candidate top-K must be positive")
    if args.max_tokens > 512 or not 0 <= args.overlap_tokens < args.max_tokens - 2:
        parser.error("require max_tokens <= 512 and overlap_tokens < max_tokens - 2")
    if not 0 < args.evaluation_fraction < 1 or not 0 < args.calibration_fraction < 1:
        parser.error("split fractions must be between zero and one")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    args.output.mkdir(parents=True, exist_ok=True)
    best_path = args.evaluate_checkpoint or args.checkpoint_dir
    if args.evaluate_checkpoint:
        saved_config = json.loads((best_path / "config.json").read_text(encoding="utf-8"))
        args.seed = saved_config["seed"]
        args.max_tokens = saved_config["max_tokens"]
        args.overlap_tokens = saved_config["overlap_tokens"]
        args.max_length = saved_config["max_length"]
        args.candidate_top_k = saved_config["candidate_top_k"]

    contracts = load_cuad(TRAIN_PATH)
    development, evaluation = split_by_contract(contracts, args.evaluation_fraction, args.seed)
    training, calibration = split_by_contract(development, args.calibration_fraction, args.seed)
    groups = {"training": training, "calibration": calibration, "evaluation": evaluation}
    ids = {name: [contract.contract_id for contract in group] for name, group in groups.items()}
    id_sets = [set(group) for group in ids.values()]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("contract split leakage detected")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )
    prompts = load_category_descriptions(DESCRIPTIONS_PATH)
    chunks = {
        name: prepare_contracts(group, tokenizer, args.max_tokens, args.overlap_tokens)
        for name, group in groups.items()
    }
    model_inputs = {name: to_model_inputs(value) for name, value in chunks.items()}
    texts = {name: value[0] for name, value in model_inputs.items()}
    labels = {name: value[1].astype(np.uint8) for name, value in model_inputs.items()}

    history = []
    best_calibration_aupr = -1.0
    if args.evaluate_checkpoint:
        model, tokenizer, features, retriever, saved_config = _load_checkpoint(best_path, device)
        args.max_length = saved_config["max_length"]
        args.max_tokens = saved_config["max_tokens"]
        args.overlap_tokens = saved_config["overlap_tokens"]
        args.candidate_top_k = saved_config["candidate_top_k"]
        args.seed = saved_config["seed"]
        args.c = saved_config["c"]
        args.dropout = saved_config["dropout"]
        args.learning_rate = saved_config["learning_rate"]
        args.batch_size = saved_config["batch_size"]
        args.negatives_per_positive = saved_config["negatives_per_positive"]
        args.min_hard_negatives = saved_config["min_hard_negatives"]
        history = saved_config.get("history", [])
        best_calibration_aupr = saved_config.get("best_calibration_macro_aupr", -1.0)
        training_retrieval = predict_word_char_probabilities(features, retriever, texts["training"])
        training_pairs, active_categories = build_training_pairs(
            chunks["training"], labels["training"], training_retrieval, prompts,
            top_k=args.candidate_top_k,
            negatives_per_positive=args.negatives_per_positive,
            min_hard_negatives=args.min_hard_negatives,
            seed=args.seed,
        )
        if active_categories.tolist() != saved_config["active_categories"]:
            raise RuntimeError("training category activity differs from the saved checkpoint")
    else:
        features, retriever = fit_word_char_tfidf(texts["training"], labels["training"], c=args.c)
        retrieval_train = predict_word_char_probabilities(features, retriever, texts["training"])
        active_categories = labels["training"].sum(axis=0) > 0
        training_pairs, active_categories = build_training_pairs(
            chunks["training"], labels["training"], retrieval_train, prompts,
            top_k=args.candidate_top_k,
            negatives_per_positive=args.negatives_per_positive,
            min_hard_negatives=args.min_hard_negatives,
            seed=args.seed,
        )
        if not active_categories.any():
            raise RuntimeError("training split has no positive category examples")
        loader = DataLoader(
            PairDataset(training_pairs), batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            collate_fn=partial(collate_pairs, tokenizer=tokenizer, max_length=args.max_length),
        )
        model = build_pair_cross_encoder(TOKENIZER_NAME, TOKENIZER_REVISION, args.dropout).to(device)
        if hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        cal_retrieval = predict_word_char_probabilities(features, retriever, texts["calibration"])
        cal_candidates = retrieve_top_k(
            cal_retrieval, chunks["calibration"], top_k=args.candidate_top_k,
            active_categories=active_categories,
        )
        y_cal = labels["calibration"]
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            start = time.monotonic()
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                target = batch.pop("labels")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    _, loss = model(**batch, labels=target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach())

            cal_scores = _dense_reranker_scores(
                model, tokenizer, chunks["calibration"], cal_candidates, prompts,
                device, args.batch_size, args.max_length, amp_enabled,
            )
            calibration_aupr = float(np.nanmean(compute_aupr(y_cal, cal_scores)))
            record = {
                "epoch": epoch + 1,
                "training_loss": total_loss / max(len(loader), 1),
                "calibration_macro_aupr": calibration_aupr,
                "epoch_seconds": time.monotonic() - start,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if calibration_aupr > best_calibration_aupr:
                best_calibration_aupr = calibration_aupr
                _save_checkpoint(best_path, model, tokenizer, features, retriever, {
                    "architecture": "word_char_top_k_retriever_category_conditioned_cross_encoder",
                    "model_name": TOKENIZER_NAME, "model_revision": TOKENIZER_REVISION,
                    "seed": args.seed, "max_length": args.max_length,
                    "max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens,
                    "candidate_top_k": args.candidate_top_k, "c": args.c,
                    "dropout": args.dropout, "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "negatives_per_positive": args.negatives_per_positive,
                    "min_hard_negatives": args.min_hard_negatives,
                    "training_pair_count": len(training_pairs),
                    "active_categories": active_categories.tolist(),
                    "best_calibration_macro_aupr": best_calibration_aupr,
                    "history": history,
                })
        model, tokenizer, features, retriever, saved_config = _load_checkpoint(best_path, device)
        args.max_length = saved_config["max_length"]
        args.candidate_top_k = saved_config["candidate_top_k"]

    active_categories = np.asarray(
        json.loads((best_path / "config.json").read_text(encoding="utf-8"))["active_categories"],
        dtype=bool,
    )
    retrieval_scores = {
        name: predict_word_char_probabilities(features, retriever, texts[name])
        for name in ("calibration", "evaluation")
    }
    candidate_sets = {
        name: retrieve_top_k(
            retrieval_scores[name], chunks[name], top_k=args.candidate_top_k,
            active_categories=active_categories,
        ) for name in ("calibration", "evaluation")
    }
    reranker_scores = {
        name: _dense_reranker_scores(
            model, tokenizer, chunks[name], candidate_sets[name], prompts,
            device, args.batch_size, args.max_length, amp_enabled,
        ) for name in ("calibration", "evaluation")
    }

    thresholds = calibrate_thresholds(labels["calibration"], reranker_scores["calibration"])
    baseline_thresholds = calibrate_thresholds(
        labels["calibration"], retrieval_scores["calibration"],
    )
    baseline_eval = _metrics_at_thresholds(
        labels["evaluation"], retrieval_scores["evaluation"], baseline_thresholds,
    )
    baseline_eval["model"] = "word_char_tfidf_retriever"
    reranker_eval = _metrics_at_thresholds(
        labels["evaluation"], reranker_scores["evaluation"], thresholds,
    )
    reranker_eval["model"] = "retrieval_cross_encoder_reranker"
    metrics = pd.concat([baseline_eval, reranker_eval], ignore_index=True)
    metrics.to_csv(args.output / "per_category_metrics.csv", index=False)

    for name in ("calibration", "evaluation"):
        np.savez_compressed(
            args.output / f"{name}_predictions.npz",
            y_true=labels[name],
            retriever_score=retrieval_scores[name],
            reranker_score=reranker_scores[name],
        )
    (args.output / "thresholds.json").write_text(json.dumps({
        "retriever": baseline_thresholds,
        "reranker": thresholds,
    }, indent=2) + "\n", encoding="utf-8")

    retrieval_eval = retrieval_scores["evaluation"]
    candidate_recall_by_k = {}
    for k in (1, 4, 8, 16):
        selected = retrieve_top_k(
            retrieval_eval, chunks["evaluation"], top_k=k,
            active_categories=active_categories,
        )
        candidate_recall_by_k[str(k)] = candidate_recall(
            labels["evaluation"], chunks["evaluation"], selected,
        )
    evaluation_macro = metrics.groupby("model")["f1"].mean().to_dict()
    evaluation_aupr = metrics.groupby("model")["aupr"].mean().to_dict()
    summary = {
        "architecture": "training-only word-character retrieval followed by a category-description/chunk cross-encoder",
        "candidate_generation": {
            "top_k_per_contract_category": args.candidate_top_k,
            "retrieval_fit_on": "training contracts only",
            "hard_negative_mining_on": "training contracts only",
            "candidate_recall_at_k": candidate_recall_by_k,
            "evaluation_candidate_pairs_scored": len(candidate_sets["evaluation"]),
        },
        "training": {
            "training_pair_count": len(training_pairs),
            "epochs": len(history), "learning_rate": args.learning_rate,
            "batch_size": args.batch_size, "max_length": args.max_length,
            "negatives_per_positive": args.negatives_per_positive,
            "min_hard_negatives": args.min_hard_negatives,
            "device": str(device), "amp": amp_enabled,
            "best_calibration_macro_aupr": best_calibration_aupr,
            "history": history,
        },
        "split": {
            "seed": args.seed,
            "contract_counts": {name: len(group) for name, group in groups.items()},
            "contract_id_sha256": {name: digest_contract_ids(ids[name]) for name in ids},
            "chunk_counts": {name: len(value) for name, value in chunks.items()},
        },
        "chunking": {
            "tokenizer": TOKENIZER_NAME, "tokenizer_revision": TOKENIZER_REVISION,
            "max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens,
        },
        "thresholds_fit_on": "calibration contracts only",
        "metrics_on": "held-out evaluation contracts only",
        "macro_f1_evaluation": evaluation_macro,
        "macro_aupr_evaluation": evaluation_aupr,
        "data_sha256": hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        "category_descriptions_sha256": hashlib.sha256(DESCRIPTIONS_PATH.read_bytes()).hexdigest(),
        "test_data_used": False,
    }
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
