"""Train/evaluate supervised-contrastive chunk embeddings with train-only prototypes.

Batches contain same-label positives and sparse-TF-IDF nearest hard negatives. The
sparse vectorizer is fitted on training chunks only and is used only for batch mining;
category prototypes are also computed only from training embeddings. Thresholds are
calibrated on contract-level calibration data. test.json is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.labels.splits import split_by_contract
from src.metrics import compute_aupr, compute_base_metrics
from src.models.encoder.contrastive import (
    ContrastiveChunkEncoder,
    build_category_prototypes,
    sample_category_triplet_batch,
    score_category_prototypes,
    supervised_contrastive_loss,
)
from src.models.encoder.experiment_utils import (
    ROOT, TRAIN_PATH, calibrate_thresholds, digest_contract_ids,
)
from src.models.naive.word_tfidf import fit_word_char_tfidf, predict_word_char_probabilities
from src.pipeline import prepare_contracts, to_model_inputs


def _embed_texts(model, tokenizer, texts, device, batch_size, max_length, amp_enabled):
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size], truncation=True, max_length=max_length,
                padding=True, return_tensors="pt",
            )
            inputs = {key: encoded[key].to(device) for key in ("input_ids", "attention_mask")}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                embeddings = model(**inputs)
            batches.append(embeddings.float().cpu())
    if not batches:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(batches)


def _prototype_scores(embeddings, positive, negative, device, temperature):
    with torch.no_grad():
        scores = score_category_prototypes(
            embeddings.to(device), positive.to(device), negative.to(device),
            temperature=temperature,
        )
    return scores.cpu().numpy()


def _metric_frame(y_true, scores, thresholds):
    vector = np.asarray([thresholds[category] for category in CATEGORIES])
    predictions = (scores >= vector).astype(np.uint8)
    metrics = compute_base_metrics(y_true, predictions, CATEGORIES)
    aupr = dict(zip(CATEGORIES, compute_aupr(y_true, scores)))
    metrics["aupr"] = metrics["category"].map(aupr)
    metrics["threshold"] = metrics["category"].map(thresholds)
    metrics.loc[metrics["support"] == 0, ["precision", "recall", "f1"]] = np.nan
    return metrics


def _candidate_recall(labels, scores, chunks, active, ks=(1, 4, 8, 16)):
    grouped = {}
    for row, item in enumerate(chunks):
        grouped.setdefault(item.chunk.contract_id, []).append(row)
    positives = {
        (item.chunk.contract_id, item.chunk.chunk_index, int(category))
        for row, item in enumerate(chunks)
        for category in np.flatnonzero(labels[row])
    }
    output = {}
    for k in ks:
        retrieved = set()
        for contract_id, rows in grouped.items():
            for category in np.flatnonzero(active):
                ranked = sorted(
                    rows,
                    key=lambda row: (-float(scores[row, category]), chunks[row].chunk.chunk_index),
                )[:k]
                for row in ranked:
                    if labels[row, category]:
                        item = chunks[row]
                        retrieved.add((contract_id, item.chunk.chunk_index, int(category)))
        output[str(k)] = {
            "positive_chunk_category_pairs": len(positives),
            "retrieved_positive_pairs": len(positives & retrieved),
            "recall": len(positives & retrieved) / len(positives) if positives else None,
        }
    return output


def _save_checkpoint(path, model, tokenizer, positive, negative, config):
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(path / "encoder")
    tokenizer.save_pretrained(path / "encoder")
    projection = {key: value.detach().cpu() for key, value in model.projection.state_dict().items()}
    torch.save({"projection": projection, "positive": positive.cpu(), "negative": negative.cpu()},
               path / "contrastive_head.pt")
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _load_checkpoint(path, device):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    encoder = AutoModel.from_pretrained(path / "encoder", local_files_only=True)
    model = ContrastiveChunkEncoder(
        encoder, embedding_size=config["embedding_size"], dropout=config["dropout"],
    )
    head = torch.load(path / "contrastive_head.pt", map_location="cpu", weights_only=True)
    model.projection.load_state_dict(head["projection"])
    model.to(device).eval()
    return model, head["positive"], head["negative"], config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/contrastive.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/contrastive_seed42")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "outputs/contrastive_seed42/best")
    parser.add_argument("--evaluate-checkpoint", type=Path)
    for name, kind in (
        ("seed", int), ("evaluation-fraction", float), ("calibration-fraction", float),
        ("max-tokens", int), ("overlap-tokens", int), ("embedding-size", int),
        ("dropout", float), ("temperature", float), ("prototype-temperature", float),
        ("hard-negative-weight", float), ("hard-negative-pool", int),
        ("categories-per-batch", int), ("steps-per-epoch", int), ("epochs", int),
        ("learning-rate", float), ("embedding-batch-size", int),
    ):
        parser.add_argument(f"--{name}", type=kind)
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["model_name"] != TOKENIZER_NAME or config["model_revision"] != TOKENIZER_REVISION:
        parser.error("model_name and model_revision must match the shared pinned tokenizer")
    fields = (
        "seed", "evaluation_fraction", "calibration_fraction", "max_tokens", "overlap_tokens",
        "embedding_size", "dropout", "temperature", "prototype_temperature",
        "hard_negative_weight", "hard_negative_pool", "categories_per_batch",
        "steps_per_epoch", "epochs", "learning_rate", "embedding_batch_size",
    )
    for field in fields:
        if getattr(args, field) is None:
            setattr(args, field, config[field])
    if args.evaluate_checkpoint:
        saved = json.loads((args.evaluate_checkpoint / "config.json").read_text(encoding="utf-8"))
        for field in fields:
            setattr(args, field, saved[field])
    if args.epochs < 1 or args.steps_per_epoch < 1:
        parser.error("epochs and steps per epoch must be positive")
    if args.temperature <= 0 or args.prototype_temperature <= 0:
        parser.error("contrastive and prototype temperatures must be positive")
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

    contracts = load_cuad(TRAIN_PATH)
    development, evaluation = split_by_contract(contracts, args.evaluation_fraction, args.seed)
    training, calibration = split_by_contract(development, args.calibration_fraction, args.seed)
    groups = {"training": training, "calibration": calibration, "evaluation": evaluation}
    contract_ids = {name: [item.contract_id for item in values] for name, values in groups.items()}
    id_sets = [set(values) for values in contract_ids.values()]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("contract split leakage detected")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )
    chunks = {
        name: prepare_contracts(group, tokenizer, args.max_tokens, args.overlap_tokens)
        for name, group in groups.items()
    }
    texts, labels = {}, {}
    for name, value in chunks.items():
        texts[name], labels[name] = to_model_inputs(value)
        labels[name] = labels[name].astype(np.uint8)
    active_categories = labels["training"].sum(axis=0) > 0
    if not active_categories.any():
        raise RuntimeError("training split contains no positive categories")

    history = []
    best_calibration_aupr = -1.0
    if args.evaluate_checkpoint:
        model, positive_prototypes, negative_prototypes, saved_config = _load_checkpoint(best_path, device)
        history = saved_config.get("history", [])
        best_calibration_aupr = saved_config["best_calibration_macro_aupr"]
    else:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=100_000,
            sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
        )
        train_sparse = vectorizer.fit_transform(texts["training"])
        active_categories = labels["training"].sum(axis=0) > 0
        encoder = AutoModel.from_pretrained(
            TOKENIZER_NAME, revision=TOKENIZER_REVISION,
            cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
        )
        model = ContrastiveChunkEncoder(
            encoder, embedding_size=args.embedding_size, dropout=args.dropout,
        ).to(device)
        if hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        rng = np.random.default_rng(args.seed)
        for epoch in range(args.epochs):
            model.train()
            loss_sum = 0.0
            for _ in range(args.steps_per_epoch):
                indices, _ = sample_category_triplet_batch(
                    labels["training"], train_sparse, active_categories, rng,
                    categories_per_batch=args.categories_per_batch,
                    hard_negative_pool=args.hard_negative_pool,
                )
                batch_texts = [texts["training"][index] for index in indices]
                encoded = tokenizer(
                    batch_texts, truncation=True, max_length=args.max_tokens,
                    padding=True, return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                batch_labels = torch.tensor(labels["training"][indices], dtype=torch.float32, device=device)
                batch_similarity = (train_sparse[indices] @ train_sparse[indices].T).toarray()
                lexical = torch.tensor(batch_similarity, dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    embeddings = model(input_ids, attention_mask)
                    loss = supervised_contrastive_loss(
                        embeddings, batch_labels, lexical,
                        temperature=args.temperature,
                        hard_negative_weight=args.hard_negative_weight,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach())

            train_embeddings = _embed_texts(
                model, tokenizer, texts["training"], device,
                args.embedding_batch_size, args.max_tokens, amp_enabled,
            )
            y_train_tensor = torch.tensor(labels["training"], dtype=torch.float32)
            positive_prototypes, negative_prototypes = build_category_prototypes(
                train_embeddings, y_train_tensor,
            )
            cal_embeddings = _embed_texts(
                model, tokenizer, texts["calibration"], device,
                args.embedding_batch_size, args.max_tokens, amp_enabled,
            )
            cal_scores = _prototype_scores(
                cal_embeddings, positive_prototypes, negative_prototypes,
                device, args.prototype_temperature,
            )
            calibration_aupr = float(np.nanmean(compute_aupr(labels["calibration"], cal_scores)))
            record = {"epoch": epoch + 1, "mean_contrastive_loss": loss_sum / args.steps_per_epoch,
                      "calibration_macro_aupr": calibration_aupr}
            history.append(record)
            print(json.dumps(record), flush=True)
            if calibration_aupr > best_calibration_aupr:
                best_calibration_aupr = calibration_aupr
                _save_checkpoint(best_path, model, tokenizer, positive_prototypes,
                                 negative_prototypes, {
                    "architecture": "multilabel_supervised_contrastive_chunk_encoder",
                    "model_name": TOKENIZER_NAME, "model_revision": TOKENIZER_REVISION,
                    "seed": args.seed, "evaluation_fraction": args.evaluation_fraction,
                    "calibration_fraction": args.calibration_fraction,
                    "max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens,
                    "embedding_size": args.embedding_size, "dropout": args.dropout,
                    "temperature": args.temperature,
                    "prototype_temperature": args.prototype_temperature,
                    "hard_negative_weight": args.hard_negative_weight,
                    "hard_negative_pool": args.hard_negative_pool,
                    "categories_per_batch": args.categories_per_batch,
                    "steps_per_epoch": args.steps_per_epoch, "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "embedding_batch_size": args.embedding_batch_size,
                    "active_categories": active_categories.tolist(),
                    "best_calibration_macro_aupr": best_calibration_aupr,
                    "history": history,
                })
        saved_config = json.loads((best_path / "config.json").read_text(encoding="utf-8"))
        saved_config["history"] = history
        saved_config["epochs_ran"] = len(history)
        saved_config["selected_epoch"] = max(
            history, key=lambda row: row["calibration_macro_aupr"],
        )["epoch"]
        (best_path / "config.json").write_text(
            json.dumps(saved_config, indent=2) + "\n", encoding="utf-8",
        )
        model, positive_prototypes, negative_prototypes, saved_config = _load_checkpoint(best_path, device)
        history = saved_config["history"]

    active_categories = np.asarray(saved_config["active_categories"], dtype=bool)
    embeddings = {
        name: _embed_texts(
            model, tokenizer, texts[name], device,
            args.embedding_batch_size, args.max_tokens, amp_enabled,
        ) for name in ("calibration", "evaluation")
    }
    scores = {
        name: _prototype_scores(
            embeddings[name], positive_prototypes, negative_prototypes,
            device, args.prototype_temperature,
        ) for name in ("calibration", "evaluation")
    }
    thresholds = calibrate_thresholds(labels["calibration"], scores["calibration"])
    contrastive_metrics = _metric_frame(labels["evaluation"], scores["evaluation"], thresholds)
    contrastive_metrics["model"] = "supervised_contrastive_prototypes"

    features, baseline = fit_word_char_tfidf(texts["training"], labels["training"], c=1.0)
    baseline_cal = predict_word_char_probabilities(features, baseline, texts["calibration"])
    baseline_eval = predict_word_char_probabilities(features, baseline, texts["evaluation"])
    baseline_thresholds = calibrate_thresholds(labels["calibration"], baseline_cal)
    baseline_metrics = _metric_frame(labels["evaluation"], baseline_eval, baseline_thresholds)
    baseline_metrics["model"] = "word_char_tfidf"
    metrics = pd.concat([baseline_metrics, contrastive_metrics], ignore_index=True)
    metrics.to_csv(args.output / "per_category_metrics.csv", index=False)

    for name in ("calibration", "evaluation"):
        np.savez_compressed(args.output / f"{name}_predictions.npz",
                            y_true=labels[name], contrastive_score=scores[name],
                            baseline_score=baseline_cal if name == "calibration" else baseline_eval)
    (args.output / "thresholds.json").write_text(json.dumps({
        "supervised_contrastive_prototypes": thresholds,
        "word_char_tfidf": baseline_thresholds,
    }, indent=2) + "\n", encoding="utf-8")

    report = {
        "architecture": "multi-label supervised contrastive chunk embeddings with train-only positive/negative prototypes",
        "contrastive_training": {
            "positive_pairs": "chunks sharing one or more labels",
            "hard_negatives": "training-only TF-IDF nearest chunks with the anchor category absent",
            "hard_negative_tf_idf_fit_on": "training chunks only",
            "embedding_size": args.embedding_size, "temperature": args.temperature,
            "hard_negative_weight": args.hard_negative_weight,
            "hard_negative_pool": args.hard_negative_pool,
            "categories_per_batch": args.categories_per_batch,
            "steps_per_epoch": args.steps_per_epoch,
            "epochs_ran": saved_config.get("epochs_ran", len(history)),
            "selected_epoch": saved_config.get("selected_epoch", len(history)),
            "learning_rate": args.learning_rate, "history": history,
            "prototype_fit_on": "training embeddings only",
            "prototype_temperature": args.prototype_temperature,
            "device": str(device), "amp": amp_enabled,
        },
        "prototype_retrieval_recall_at_k": _candidate_recall(
            labels["evaluation"], scores["evaluation"], chunks["evaluation"], active_categories,
        ),
        "split": {
            "seed": args.seed,
            "contract_counts": {name: len(value) for name, value in groups.items()},
            "contract_id_sha256": {name: digest_contract_ids(contract_ids[name]) for name in groups},
            "chunk_counts": {name: len(value) for name, value in chunks.items()},
        },
        "chunking": {"max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens,
                     "tokenizer": TOKENIZER_NAME, "tokenizer_revision": TOKENIZER_REVISION},
        "thresholds_fit_on": "calibration contracts only",
        "metrics_on": "held-out evaluation contracts only",
        "macro_f1_evaluation": metrics.groupby("model")["f1"].mean().to_dict(),
        "macro_aupr_evaluation": metrics.groupby("model")["aupr"].mean().to_dict(),
        "data_sha256": hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        "test_data_used": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n",
                                              encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
