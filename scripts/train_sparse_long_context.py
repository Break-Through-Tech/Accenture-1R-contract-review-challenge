"""Train/evaluate Longformer-style sparse attention over every contract chunk.

A frozen pinned encoder represents overlapping chunks. A chunk-level local/global sparse
transformer processes the complete variable-length sequence: each chunk attends to a
bounded neighborhood plus a contract-global token, while the global token attends to all
chunks. This keeps context cost linear in contract length for a fixed local radius.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.labels.loader import load_cuad
from src.labels.splits import split_by_contract
from src.metrics import compute_aupr, compute_base_metrics
from src.models.encoder.experiment_utils import (
    ROOT, TRAIN_PATH, calibrate_thresholds, digest_contract_ids,
)
from src.models.encoder.sparse_long_context import (
    SparseLongContextClassifier,
    overlap_consensus_contract_scores,
    sparse_attention_edge_count,
)
from src.models.naive.word_tfidf import fit_word_char_tfidf, predict_word_char_probabilities
from src.pipeline import prepare_contracts


def _encode_chunks(encoder, tokenizer, chunks, device, batch_size, max_length):
    encoder.eval()
    outputs = []
    texts = [item.chunk.text for item in chunks]
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size], truncation=True, max_length=max_length,
                padding=True, return_tensors="pt",
            )
            inputs = {key: encoded[key].to(device) for key in ("input_ids", "attention_mask")}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                hidden = encoder(**inputs).last_hidden_state[:, 0]
            outputs.append(hidden.float().cpu())
    return torch.cat(outputs, dim=0)


def _group_contract_chunks(chunks, embeddings):
    grouped_chunks, grouped_embeddings = defaultdict(list), defaultdict(list)
    for item, embedding in zip(chunks, embeddings):
        contract_id = item.chunk.contract_id
        grouped_chunks[contract_id].append(item)
        grouped_embeddings[contract_id].append(embedding)
    for contract_id in grouped_chunks:
        order = np.argsort([item.chunk.chunk_index for item in grouped_chunks[contract_id]])
        grouped_chunks[contract_id] = [grouped_chunks[contract_id][i] for i in order]
        grouped_embeddings[contract_id] = torch.stack(
            [grouped_embeddings[contract_id][i] for i in order], dim=0,
        )
    return grouped_chunks, grouped_embeddings


def _predict_contracts(model, embeddings_by_contract, device):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for contract_id, embeddings in embeddings_by_contract.items():
            logits = model(embeddings.to(device))
            predictions[contract_id] = torch.sigmoid(logits.float()).cpu().numpy()
    return predictions


def _align_chunk_scores(chunks, predictions):
    y_true = np.asarray([item.labels for item in chunks], dtype=np.uint8)
    y_score = np.asarray([
        predictions[item.chunk.contract_id][item.chunk.chunk_index] for item in chunks
    ], dtype=np.float32)
    return y_true, y_score


def _contract_targets(contracts):
    targets = np.zeros((len(contracts), len(CATEGORIES)), dtype=np.uint8)
    for row, contract in enumerate(contracts):
        for span in contract.spans:
            targets[row, CATEGORY_TO_INDEX[span.category]] = 1
    return targets


def _align_contract_scores(contracts, scores_by_contract):
    return np.asarray([scores_by_contract[contract.contract_id] for contract in contracts],
                      dtype=np.float32)


def _metric_frame(y_true, scores, thresholds):
    vector = np.asarray([thresholds[category] for category in CATEGORIES])
    predicted = (scores >= vector).astype(np.uint8)
    metrics = compute_base_metrics(y_true, predicted, CATEGORIES)
    aupr = dict(zip(CATEGORIES, compute_aupr(y_true, scores)))
    metrics["aupr"] = metrics["category"].map(aupr)
    metrics["threshold"] = metrics["category"].map(thresholds)
    metrics.loc[metrics["support"] == 0, ["precision", "recall", "f1"]] = np.nan
    return metrics


def _source_coverage(contracts, chunks_by_contract):
    covered_chars = 0
    total_chars = 0
    fully_covered = 0
    for contract in contracts:
        intervals = sorted(
            (item.chunk.start, item.chunk.end)
            for item in chunks_by_contract[contract.contract_id]
        )
        cursor = 0
        covered = 0
        for start, end in intervals:
            start = max(start, cursor)
            if end > start:
                covered += end - start
                cursor = end
        total = len(contract.text)
        total_chars += total
        covered_chars += covered
        fully_covered += int(covered == total)
    return {
        "contracts": len(contracts), "fully_covered_contracts": fully_covered,
        "total_source_characters": total_chars, "covered_source_characters": covered_chars,
        "source_character_coverage": covered_chars / total_chars if total_chars else 1.0,
    }


def _save_checkpoint(path, model, config):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
               path / "sparse_context_head.pt")
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _load_checkpoint(path, input_size, device):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model = SparseLongContextClassifier(
        input_size=input_size, num_labels=len(CATEGORIES),
        hidden_size=config["hidden_size"], layers=config["layers"],
        heads=config["attention_heads"], local_radius=config["local_radius"],
        max_chunks=config["max_chunks"], dropout=config["dropout"],
    )
    state = torch.load(path / "sparse_context_head.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sparse_long_context.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/sparse_long_context_seed42")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "outputs/sparse_long_context_seed42/best")
    parser.add_argument("--evaluate-checkpoint", type=Path)
    for name, kind in (
        ("seed", int), ("evaluation-fraction", float), ("calibration-fraction", float),
        ("max-tokens", int), ("overlap-tokens", int), ("encoder-batch-size", int),
        ("max-chunks", int), ("hidden-size", int), ("layers", int),
        ("attention-heads", int), ("local-radius", int), ("dropout", float),
        ("epochs", int), ("learning-rate", float), ("pos-weight-cap", float),
    ):
        parser.add_argument(f"--{name}", type=kind)
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["model_name"] != TOKENIZER_NAME or config["model_revision"] != TOKENIZER_REVISION:
        parser.error("model_name and model_revision must match the shared pinned tokenizer")
    fields = (
        "seed", "evaluation_fraction", "calibration_fraction", "max_tokens", "overlap_tokens",
        "encoder_batch_size", "max_chunks", "hidden_size", "layers", "attention_heads",
        "local_radius", "dropout", "epochs", "learning_rate", "pos_weight_cap",
    )
    for field in fields:
        if getattr(args, field) is None:
            setattr(args, field, config[field])
    if args.evaluate_checkpoint:
        saved = json.loads((args.evaluate_checkpoint / "config.json").read_text(encoding="utf-8"))
        for field in fields:
            setattr(args, field, saved[field])
    if args.epochs < 1 or args.layers < 1 or args.encoder_batch_size < 1:
        parser.error("epochs, layers, and encoder batch size must be positive")
    if args.max_tokens > 512 or not 0 <= args.overlap_tokens < args.max_tokens - 2:
        parser.error("require max_tokens <= 512 and overlap_tokens < max_tokens - 2")
    if args.local_radius < 0 or args.max_chunks < 1:
        parser.error("local radius must be nonnegative and max_chunks positive")
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
    ids = {name: [contract.contract_id for contract in values] for name, values in groups.items()}
    id_sets = [set(value) for value in ids.values()]
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
    labels = {
        name: np.asarray([item.labels for item in values], dtype=np.uint8)
        for name, values in chunks.items()
    }
    encoder = AutoModel.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    ).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    feature_start = time.monotonic()
    embeddings = {
        name: _encode_chunks(
            encoder, tokenizer, values, device, args.encoder_batch_size, args.max_tokens,
        ) for name, values in chunks.items()
    }
    feature_extraction_seconds = time.monotonic() - feature_start
    input_size = int(encoder.config.hidden_size)
    del encoder
    chunks_by_contract = {}
    embeddings_by_contract = {}
    for name in groups:
        chunks_by_contract[name], embeddings_by_contract[name] = _group_contract_chunks(
            chunks[name], embeddings[name],
        )

    history = []
    best_calibration_aupr = -1.0
    if args.evaluate_checkpoint:
        model, saved_config = _load_checkpoint(best_path, input_size, device)
        history = saved_config.get("history", [])
        best_calibration_aupr = saved_config["best_calibration_macro_aupr"]
    else:
        model = SparseLongContextClassifier(
            input_size=input_size, num_labels=len(CATEGORIES), hidden_size=args.hidden_size,
            layers=args.layers, heads=args.attention_heads, local_radius=args.local_radius,
            max_chunks=args.max_chunks, dropout=args.dropout,
        ).to(device)
        pos = labels["training"].sum(axis=0)
        pos_weight = np.divide(len(labels["training"]) - pos, pos,
                               out=np.ones_like(pos, dtype=float), where=pos > 0)
        pos_weight = torch.tensor(np.clip(pos_weight, 1, args.pos_weight_cap),
                                  dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        for epoch in range(args.epochs):
            model.train()
            order = list(embeddings_by_contract["training"])
            random.Random(args.seed + epoch).shuffle(order)
            total_loss = 0.0
            for contract_id in order:
                features = embeddings_by_contract["training"][contract_id].to(device)
                targets = torch.tensor(
                    [item.labels for item in chunks_by_contract["training"][contract_id]],
                    dtype=torch.float32, device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach())
            cal_predictions = _predict_contracts(
                model, embeddings_by_contract["calibration"], device,
            )
            y_cal, score_cal = _align_chunk_scores(chunks["calibration"], cal_predictions)
            calibration_aupr = float(np.nanmean(compute_aupr(y_cal, score_cal)))
            record = {"epoch": epoch + 1,
                      "training_loss_per_contract": total_loss / len(order),
                      "calibration_chunk_macro_aupr": calibration_aupr}
            history.append(record)
            print(json.dumps(record), flush=True)
            if calibration_aupr > best_calibration_aupr:
                best_calibration_aupr = calibration_aupr
                _save_checkpoint(best_path, model, {
                    "architecture": "frozen_chunk_encoder_local_global_sparse_contract_attention",
                    "model_name": TOKENIZER_NAME, "model_revision": TOKENIZER_REVISION,
                    "seed": args.seed, "evaluation_fraction": args.evaluation_fraction,
                    "calibration_fraction": args.calibration_fraction,
                    "max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens,
                    "encoder_batch_size": args.encoder_batch_size,
                    "encoder_frozen": True, "input_size": input_size,
                    "max_chunks": args.max_chunks, "hidden_size": args.hidden_size,
                    "layers": args.layers, "attention_heads": args.attention_heads,
                    "local_radius": args.local_radius, "dropout": args.dropout,
                    "epochs": args.epochs, "learning_rate": args.learning_rate,
                    "pos_weight_cap": args.pos_weight_cap,
                    "best_calibration_macro_aupr": best_calibration_aupr,
                    "history": history,
                })
        saved_config = json.loads((best_path / "config.json").read_text(encoding="utf-8"))
        saved_config["history"] = history
        saved_config["epochs_ran"] = len(history)
        saved_config["selected_epoch"] = max(
            history, key=lambda row: row["calibration_chunk_macro_aupr"],
        )["epoch"]
        (best_path / "config.json").write_text(
            json.dumps(saved_config, indent=2) + "\n", encoding="utf-8",
        )
        model, saved_config = _load_checkpoint(best_path, input_size, device)
        history = saved_config["history"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    context_start = time.monotonic()
    predictions = {
        name: _predict_contracts(model, embeddings_by_contract[name], device)
        for name in ("calibration", "evaluation")
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    context_inference_seconds = time.monotonic() - context_start
    peak_context_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )

    y_cal, score_cal = _align_chunk_scores(chunks["calibration"], predictions["calibration"])
    y_eval, score_eval = _align_chunk_scores(chunks["evaluation"], predictions["evaluation"])
    chunk_thresholds = calibrate_thresholds(y_cal, score_cal)
    chunk_metrics = _metric_frame(y_eval, score_eval, chunk_thresholds)
    chunk_metrics["model"] = "sparse_local_global_context"

    baseline_features, baseline = fit_word_char_tfidf(
        [item.chunk.text for item in chunks["training"]], labels["training"], c=1.0,
    )
    baseline_cal = predict_word_char_probabilities(
        baseline_features, baseline, [item.chunk.text for item in chunks["calibration"]],
    )
    baseline_eval = predict_word_char_probabilities(
        baseline_features, baseline, [item.chunk.text for item in chunks["evaluation"]],
    )
    baseline_thresholds = calibrate_thresholds(labels["calibration"], baseline_cal)
    baseline_chunk_metrics = _metric_frame(labels["evaluation"], baseline_eval, baseline_thresholds)
    baseline_chunk_metrics["model"] = "word_char_tfidf"
    pd.concat([baseline_chunk_metrics, chunk_metrics], ignore_index=True).to_csv(
        args.output / "per_category_metrics.csv", index=False,
    )

    contract_targets = {name: _contract_targets(group) for name, group in groups.items()}
    contract_scores = {
        name: overlap_consensus_contract_scores(chunks[name], score_matrix)
        for name, score_matrix in (
            ("calibration", _align_chunk_scores(chunks["calibration"], predictions["calibration"])[1]),
            ("evaluation", _align_chunk_scores(chunks["evaluation"], predictions["evaluation"])[1]),
        )
    }
    baseline_contract_scores = {
        "calibration": overlap_consensus_contract_scores(chunks["calibration"], baseline_cal),
        "evaluation": overlap_consensus_contract_scores(chunks["evaluation"], baseline_eval),
    }
    model_cal_contract = _align_contract_scores(calibration, contract_scores["calibration"])
    model_eval_contract = _align_contract_scores(evaluation, contract_scores["evaluation"])
    baseline_cal_contract = _align_contract_scores(calibration, baseline_contract_scores["calibration"])
    baseline_eval_contract = _align_contract_scores(evaluation, baseline_contract_scores["evaluation"])
    model_contract_thresholds = calibrate_thresholds(contract_targets["calibration"], model_cal_contract)
    baseline_contract_thresholds = calibrate_thresholds(
        contract_targets["calibration"], baseline_cal_contract,
    )
    model_contract_metrics = _metric_frame(
        contract_targets["evaluation"], model_eval_contract, model_contract_thresholds,
    )
    model_contract_metrics["model"] = "sparse_local_global_context"
    baseline_contract_metrics = _metric_frame(
        contract_targets["evaluation"], baseline_eval_contract, baseline_contract_thresholds,
    )
    baseline_contract_metrics["model"] = "word_char_tfidf"
    pd.concat([baseline_contract_metrics, model_contract_metrics], ignore_index=True).to_csv(
        args.output / "contract_per_category_metrics.csv", index=False,
    )

    np.savez_compressed(args.output / "calibration_predictions.npz",
                        y_true=y_cal, sparse_score=score_cal, baseline_score=baseline_cal,
                        contract_y_true=contract_targets["calibration"],
                        sparse_contract_score=model_cal_contract,
                        baseline_contract_score=baseline_cal_contract)
    np.savez_compressed(args.output / "evaluation_predictions.npz",
                        y_true=y_eval, sparse_score=score_eval, baseline_score=baseline_eval,
                        contract_y_true=contract_targets["evaluation"],
                        sparse_contract_score=model_eval_contract,
                        baseline_contract_score=baseline_eval_contract)
    (args.output / "thresholds.json").write_text(json.dumps({
        "sparse_chunk": chunk_thresholds, "word_char_tfidf_chunk": baseline_thresholds,
        "sparse_overlap_consensus_contract": model_contract_thresholds,
        "word_char_tfidf_overlap_consensus_contract": baseline_contract_thresholds,
    }, indent=2) + "\n", encoding="utf-8")

    max_eval_chunks = max(len(value) for value in chunks_by_contract["evaluation"].values())
    eval_sparse_links = sum(
        sparse_attention_edge_count(len(value), args.local_radius)
        for value in embeddings_by_contract["evaluation"].values()
    )
    eval_dense_links = sum(len(value) ** 2 for value in embeddings_by_contract["evaluation"].values())
    coverage = {name: _source_coverage(groups[name], chunks_by_contract[name])
                for name in groups}
    chunk_model_metrics = pd.concat([baseline_chunk_metrics, chunk_metrics], ignore_index=True)
    contract_model_metrics = pd.concat([baseline_contract_metrics, model_contract_metrics], ignore_index=True)
    report = {
        "architecture": "frozen chunk encoder with local-window plus contract-global sparse attention over all chunks",
        "context": {
            "local_radius_chunks": args.local_radius,
            "local_window_chunks": 2 * args.local_radius + 1,
            "global_token_count": 1,
            "attention_layers": args.layers,
            "estimated_evaluation_sparse_links_per_layer": eval_sparse_links,
            "equivalent_dense_chunk_links": eval_dense_links,
            "sparse_links_per_layer_divided_by_dense": eval_sparse_links / eval_dense_links
                if eval_dense_links else None,
            "overlap_aggregation": "maximum segment-wise mean of scores from windows covering each original character interval",
            "source_character_coverage": coverage,
            "max_evaluation_chunks_in_contract": max_eval_chunks,
        },
        "training": {
            "encoder_frozen": True, "epochs_ran": saved_config.get("epochs_ran", len(history)),
            "selected_epoch": saved_config.get("selected_epoch", len(history)),
            "learning_rate": args.learning_rate, "hidden_size": args.hidden_size,
            "layers": args.layers, "attention_heads": args.attention_heads,
            "local_radius": args.local_radius, "dropout": args.dropout,
            "pos_weight_cap": args.pos_weight_cap, "history": history,
            "feature_extraction_seconds": feature_extraction_seconds,
            "sparse_context_inference_seconds_calibration_plus_evaluation": context_inference_seconds,
            "sparse_context_peak_cuda_memory_bytes": peak_context_memory_bytes,
            "device": str(device),
        },
        "split": {
            "seed": args.seed,
            "contract_counts": {name: len(value) for name, value in groups.items()},
            "contract_id_sha256": {name: digest_contract_ids(ids[name]) for name in groups},
            "chunk_counts": {name: len(value) for name, value in chunks.items()},
        },
        "chunking": {"max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens,
                     "tokenizer": TOKENIZER_NAME, "tokenizer_revision": TOKENIZER_REVISION},
        "thresholds_fit_on": "calibration chunks for chunk metrics and calibration contract labels for overlap-consensus metrics",
        "metrics_on": "held-out evaluation contracts only",
        "macro_f1_chunk_evaluation": chunk_model_metrics.groupby("model")["f1"].mean().to_dict(),
        "macro_aupr_chunk_evaluation": chunk_model_metrics.groupby("model")["aupr"].mean().to_dict(),
        "macro_f1_contract_evaluation": contract_model_metrics.groupby("model")["f1"].mean().to_dict(),
        "macro_aupr_contract_evaluation": contract_model_metrics.groupby("model")["aupr"].mean().to_dict(),
        "data_sha256": hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        "test_data_used": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n",
                                              encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


def _contract_targets(contracts):
    targets = np.zeros((len(contracts), len(CATEGORIES)), dtype=np.uint8)
    for row, contract in enumerate(contracts):
        for span in contract.spans:
            targets[row, CATEGORY_TO_INDEX[span.category]] = 1
    return targets


def _align_contract_scores(contracts, score_by_contract):
    return np.asarray([score_by_contract[contract.contract_id] for contract in contracts],
                      dtype=np.float32)


if __name__ == "__main__":
    main()
