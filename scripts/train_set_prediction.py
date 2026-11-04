"""Train/evaluate a Hungarian-matched set predictor for CUAD answer-span instances.

A frozen pinned encoder supplies one embedding per overlapping text chunk. Learned
category/instance slots attend over each complete contract and predict presence plus
start/end chunk pointers. Hungarian matching trains the slots against CUAD answer spans.
Predicted chunk intervals map back to source character offsets; thresholds use calibration
contracts only. test.json is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoModel, AutoTokenizer

from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.labels.loader import Contract, load_cuad
from src.labels.splits import split_by_contract
from src.metrics import compute_aupr, compute_base_metrics
from src.models.encoder.experiment_utils import (
    ROOT, TRAIN_PATH, calibrate_thresholds, digest_contract_ids,
)
from src.models.encoder.set_prediction import (
    ContractSetPredictor, decode_contract_set, map_gold_spans_to_chunks,
    matched_set_loss, span_iou,
)
from src.models.naive.word_tfidf import fit_word_char_tfidf, predict_word_char_probabilities
from src.pipeline import prepare_contracts, to_model_inputs


def _encode_chunk_embeddings(encoder, tokenizer, chunks, device, batch_size, max_length):
    encoder.eval()
    texts = [item.chunk.text for item in chunks]
    outputs = []
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


def _group_chunks(chunks, embeddings):
    grouped_chunks = defaultdict(list)
    grouped_embeddings = defaultdict(list)
    for item, embedding in zip(chunks, embeddings):
        grouped_chunks[item.chunk.contract_id].append(item)
        grouped_embeddings[item.chunk.contract_id].append(embedding)
    for contract_id in grouped_chunks:
        order = np.argsort([item.chunk.chunk_index for item in grouped_chunks[contract_id]])
        grouped_chunks[contract_id] = [grouped_chunks[contract_id][i] for i in order]
        grouped_embeddings[contract_id] = torch.stack(
            [grouped_embeddings[contract_id][i] for i in order], dim=0,
        )
    return grouped_chunks, grouped_embeddings


def _gold_by_contract(contracts, chunks_by_contract, max_slots):
    contract_by_id = {contract.contract_id: contract for contract in contracts}
    mapped = {}
    uncovered = 0
    total = 0
    slot_covered = 0
    for contract_id, chunks in chunks_by_contract.items():
        contract = contract_by_id[contract_id]
        spans, missed = map_gold_spans_to_chunks(contract, chunks)
        mapped[contract_id] = spans
        uncovered += missed
        total += len(contract.spans)
        counts = Counter(span.category_index for span in spans)
        slot_covered += sum(min(count, max_slots) for count in counts.values())
    return mapped, {"gold_span_count": total, "mapped_span_count": total - uncovered,
                    "uncovered_span_count": uncovered,
                    "slot_covered_span_count": slot_covered,
                    "slot_coverage_of_mapped_spans": slot_covered / (total - uncovered)
                    if total != uncovered else None}


def _predict_contracts(model, embeddings_by_contract, chunks_by_contract, device):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for contract_id, embeddings in embeddings_by_contract.items():
            output = model(embeddings.to(device))
            predictions[contract_id] = decode_contract_set(output, chunks_by_contract[contract_id])
    return predictions


def _contract_targets(contracts):
    targets = np.zeros((len(contracts), len(CATEGORIES)), dtype=np.uint8)
    for row, contract in enumerate(contracts):
        for span in contract.spans:
            targets[row, CATEGORY_TO_INDEX[span.category]] = 1
    return targets


def _align_contract_scores(contracts, predictions):
    return np.asarray([predictions[contract.contract_id].contract_presence
                       for contract in contracts], dtype=np.float32)


def _align_chunk_scores(chunks, predictions):
    labels = np.asarray([item.labels for item in chunks], dtype=np.uint8)
    scores = np.asarray([
        predictions[item.chunk.contract_id].chunk_scores[item.chunk.chunk_index]
        for item in chunks
    ], dtype=np.float32)
    return labels, scores


def _span_metrics(contracts, predictions, thresholds, iou_threshold=0.5):
    counts = {category: {"tp": 0, "fp": 0, "fn": 0, "matched_iou": []}
              for category in CATEGORIES}
    for contract in contracts:
        candidate_spans = predictions[contract.contract_id].spans
        for category_index, category in enumerate(CATEGORIES):
            gold = [span for span in contract.spans
                    if CATEGORY_TO_INDEX[span.category] == category_index]
            predicted = [span for span in candidate_spans
                         if span.category_index == category_index
                         and span.score >= thresholds[category]]
            if not gold or not predicted:
                counts[category]["fp"] += len(predicted)
                counts[category]["fn"] += len(gold)
                continue
            overlaps = np.asarray([
                [span_iou(item.char_start, item.char_end, truth.start, truth.end)
                 for truth in gold] for item in predicted
            ])
            rows, columns = linear_sum_assignment(-overlaps)
            accepted = [(row, column) for row, column in zip(rows, columns)
                        if overlaps[row, column] >= iou_threshold]
            counts[category]["tp"] += len(accepted)
            counts[category]["fp"] += len(predicted) - len(accepted)
            counts[category]["fn"] += len(gold) - len(accepted)
            counts[category]["matched_iou"].extend(
                float(overlaps[row, column]) for row, column in accepted
            )
    per_category = []
    all_tp = all_fp = all_fn = 0
    all_iou = []
    for category in CATEGORIES:
        values = counts[category]
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_category.append({"category": category, "true_positive": tp,
                             "false_positive": fp, "false_negative": fn,
                             "precision_iou50": precision, "recall_iou50": recall,
                             "f1_iou50": f1, "mean_matched_iou": (
                                 float(np.mean(values["matched_iou"]))
                                 if values["matched_iou"] else None)})
        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_iou.extend(values["matched_iou"])
    micro_precision = all_tp / (all_tp + all_fp) if all_tp + all_fp else 0.0
    micro_recall = all_tp / (all_tp + all_fn) if all_tp + all_fn else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if micro_precision + micro_recall else 0.0)
    return {
        "iou_threshold": iou_threshold,
        "true_positive": all_tp, "false_positive": all_fp, "false_negative": all_fn,
        "precision_iou50_micro": micro_precision, "recall_iou50_micro": micro_recall,
        "f1_iou50_micro": micro_f1,
        "f1_iou50_macro": float(np.mean([row["f1_iou50"] for row in per_category
                                          if row["true_positive"] + row["false_negative"] > 0])),
        "mean_iou_of_matched_spans": float(np.mean(all_iou)) if all_iou else None,
        "per_category": per_category,
    }


def _metric_frame(y_true, scores, thresholds):
    vector = np.asarray([thresholds[category] for category in CATEGORIES])
    predictions = (scores >= vector).astype(np.uint8)
    metrics = compute_base_metrics(y_true, predictions, CATEGORIES)
    aupr = dict(zip(CATEGORIES, compute_aupr(y_true, scores)))
    metrics["aupr"] = metrics["category"].map(aupr)
    metrics["threshold"] = metrics["category"].map(thresholds)
    metrics.loc[metrics["support"] == 0, ["precision", "recall", "f1"]] = np.nan
    return metrics


def _save_checkpoint(path, model, config):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()},
               path / "set_predictor.pt")
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _load_checkpoint(path, hidden_size, device):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model = ContractSetPredictor(
        hidden_size, num_labels=len(CATEGORIES), max_slots=config["max_slots"],
        max_chunks=config["max_chunks"], attention_heads=config["attention_heads"],
        dropout=config["dropout"],
    )
    state = torch.load(path / "set_predictor.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/set_prediction.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/set_prediction_seed42")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "outputs/set_prediction_seed42/best")
    parser.add_argument("--evaluate-checkpoint", type=Path)
    for name, kind in (
        ("seed", int), ("evaluation-fraction", float), ("calibration-fraction", float),
        ("max-tokens", int), ("overlap-tokens", int), ("encoder-batch-size", int),
        ("max-slots", int), ("max-chunks", int), ("attention-heads", int),
        ("dropout", float), ("no-object-weight", float), ("pointer-weight", float),
        ("epochs", int), ("learning-rate", float),
    ):
        parser.add_argument(f"--{name}", type=kind)
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["model_name"] != TOKENIZER_NAME or config["model_revision"] != TOKENIZER_REVISION:
        parser.error("model_name and model_revision must match the shared pinned tokenizer")
    fields = (
        "seed", "evaluation_fraction", "calibration_fraction", "max_tokens", "overlap_tokens",
        "encoder_batch_size", "max_slots", "max_chunks", "attention_heads", "dropout",
        "no_object_weight", "pointer_weight", "epochs", "learning_rate",
    )
    for field in fields:
        if getattr(args, field) is None:
            setattr(args, field, config[field])
    if args.evaluate_checkpoint:
        saved = json.loads((args.evaluate_checkpoint / "config.json").read_text(encoding="utf-8"))
        for field in fields:
            setattr(args, field, saved[field])
    if args.epochs < 1 or args.max_slots < 1 or args.encoder_batch_size < 1:
        parser.error("epochs, max slots, and encoder batch size must be positive")
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
    chunk_labels = {name: np.asarray([item.labels for item in value], dtype=np.uint8)
                    for name, value in chunks.items()}
    encoder = AutoModel.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    ).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    embeddings = {
        name: _encode_chunk_embeddings(
            encoder, tokenizer, value, device, args.encoder_batch_size, args.max_tokens,
        ) for name, value in chunks.items()
    }
    hidden_size = int(encoder.config.hidden_size)
    del encoder
    chunks_by_contract = {}
    embeddings_by_contract = {}
    gold_by_contract = {}
    coverage = {}
    for name in groups:
        chunks_by_contract[name], embeddings_by_contract[name] = _group_chunks(chunks[name], embeddings[name])
        gold_by_contract[name], coverage[name] = _gold_by_contract(
            groups[name], chunks_by_contract[name], args.max_slots,
        )

    history = []
    best_calibration_aupr = -1.0
    if args.evaluate_checkpoint:
        model, saved_config = _load_checkpoint(best_path, hidden_size, device)
        history = saved_config.get("history", [])
        best_calibration_aupr = saved_config["best_calibration_macro_aupr"]
    else:
        model = ContractSetPredictor(
            hidden_size, num_labels=len(CATEGORIES), max_slots=args.max_slots,
            max_chunks=args.max_chunks, attention_heads=args.attention_heads,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        for epoch in range(args.epochs):
            model.train()
            order = list(chunks_by_contract["training"])
            random.Random(args.seed + epoch).shuffle(order)
            total_loss = matched = overflow = 0
            for contract_id in order:
                output = model(embeddings_by_contract["training"][contract_id].to(device))
                loss, stats = matched_set_loss(
                    output, gold_by_contract["training"][contract_id],
                    max_slots=args.max_slots, no_object_weight=args.no_object_weight,
                    pointer_weight=args.pointer_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach())
                matched += stats["matched_spans"]
                overflow += stats["overflow_spans"]
            calibration_predictions = _predict_contracts(
                model, embeddings_by_contract["calibration"],
                chunks_by_contract["calibration"], device,
            )
            cal_targets = _contract_targets(calibration)
            cal_scores = _align_contract_scores(calibration, calibration_predictions)
            calibration_aupr = float(np.nanmean(compute_aupr(cal_targets, cal_scores)))
            record = {
                "epoch": epoch + 1,
                "training_loss_per_contract": total_loss / max(len(order), 1),
                "matched_training_spans": matched,
                "training_spans_over_slot_limit": overflow,
                "calibration_contract_macro_aupr": calibration_aupr,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if calibration_aupr > best_calibration_aupr:
                best_calibration_aupr = calibration_aupr
                _save_checkpoint(best_path, model, {
                    "architecture": "hungarian_matched_category_instance_chunk_span_set_predictor",
                    "model_name": TOKENIZER_NAME, "model_revision": TOKENIZER_REVISION,
                    "seed": args.seed, "evaluation_fraction": args.evaluation_fraction,
                    "calibration_fraction": args.calibration_fraction,
                    "max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens,
                    "encoder_batch_size": args.encoder_batch_size,
                    "encoder_frozen": True, "max_slots": args.max_slots,
                    "max_chunks": args.max_chunks, "attention_heads": args.attention_heads,
                    "dropout": args.dropout, "no_object_weight": args.no_object_weight,
                    "pointer_weight": args.pointer_weight, "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "best_calibration_macro_aupr": best_calibration_aupr,
                    "history": history,
                })
        saved_config = json.loads((best_path / "config.json").read_text(encoding="utf-8"))
        saved_config["history"] = history
        saved_config["epochs_ran"] = len(history)
        saved_config["selected_epoch"] = max(
            history, key=lambda row: row["calibration_contract_macro_aupr"],
        )["epoch"]
        (best_path / "config.json").write_text(
            json.dumps(saved_config, indent=2) + "\n", encoding="utf-8",
        )
        model, saved_config = _load_checkpoint(best_path, hidden_size, device)
        history = saved_config["history"]

    predictions = {
        name: _predict_contracts(
            model, embeddings_by_contract[name], chunks_by_contract[name], device,
        ) for name in ("calibration", "evaluation")
    }
    cal_contract_targets = _contract_targets(calibration)
    eval_contract_targets = _contract_targets(evaluation)
    cal_contract_scores = _align_contract_scores(calibration, predictions["calibration"])
    eval_contract_scores = _align_contract_scores(evaluation, predictions["evaluation"])
    presence_thresholds = calibrate_thresholds(cal_contract_targets, cal_contract_scores)
    y_cal, score_cal = _align_chunk_scores(chunks["calibration"], predictions["calibration"])
    y_eval, score_eval = _align_chunk_scores(chunks["evaluation"], predictions["evaluation"])
    chunk_thresholds = calibrate_thresholds(y_cal, score_cal)
    set_metrics = _metric_frame(y_eval, score_eval, chunk_thresholds)
    set_metrics["model"] = "hungarian_set_prediction"

    texts = {name: [item.chunk.text for item in value] for name, value in chunks.items()}
    features, baseline = fit_word_char_tfidf(texts["training"], chunk_labels["training"], c=1.0)
    baseline_cal = predict_word_char_probabilities(features, baseline, texts["calibration"])
    baseline_eval = predict_word_char_probabilities(features, baseline, texts["evaluation"])
    baseline_thresholds = calibrate_thresholds(chunk_labels["calibration"], baseline_cal)
    baseline_metrics = _metric_frame(chunk_labels["evaluation"], baseline_eval, baseline_thresholds)
    baseline_metrics["model"] = "word_char_tfidf"
    pd.concat([baseline_metrics, set_metrics], ignore_index=True).to_csv(
        args.output / "per_category_metrics.csv", index=False,
    )

    for name, y_true, contract_y, contract_scores, chunk_scores in (
        ("calibration", y_cal, cal_contract_targets, cal_contract_scores, score_cal),
        ("evaluation", y_eval, eval_contract_targets, eval_contract_scores, score_eval),
    ):
        np.savez_compressed(
            args.output / f"{name}_predictions.npz",
            y_true=y_true, set_chunk_score=chunk_scores,
            contract_y_true=contract_y, contract_presence_score=contract_scores,
            baseline_score=baseline_cal if name == "calibration" else baseline_eval,
        )
    (args.output / "thresholds.json").write_text(json.dumps({
        "set_contract_presence": presence_thresholds,
        "set_chunk_projection": chunk_thresholds,
        "word_char_tfidf_chunk": baseline_thresholds,
    }, indent=2) + "\n", encoding="utf-8")

    span_metrics = _span_metrics(
        evaluation, predictions["evaluation"], presence_thresholds, iou_threshold=0.5,
    )
    report = {
        "architecture": "category-conditioned learned instance slots with Hungarian-matched start/end chunk pointers",
        "span_prediction": {
            "max_slots_per_category": args.max_slots,
            "pointer_resolution": "chunk indices mapped to end-exclusive source character boundaries",
            "encoder_frozen": True,
            "training_loss": "matched objectness/no-object BCE plus start/end pointer cross-entropy",
            "matching": "Hungarian assignment per category within each contract",
            "gold_span_coverage": coverage,
            "evaluation_span_metrics": span_metrics,
        },
        "training": {
            "epochs_ran": saved_config.get("epochs_ran", len(history)),
            "selected_epoch": saved_config.get("selected_epoch", len(history)),
            "learning_rate": args.learning_rate,
            "attention_heads": args.attention_heads, "dropout": args.dropout,
            "no_object_weight": args.no_object_weight, "pointer_weight": args.pointer_weight,
            "best_calibration_contract_macro_aupr": best_calibration_aupr,
            "history": history, "device": str(device),
        },
        "split": {
            "seed": args.seed,
            "contract_counts": {name: len(value) for name, value in groups.items()},
            "contract_id_sha256": {name: digest_contract_ids(contract_ids[name]) for name in groups},
            "chunk_counts": {name: len(value) for name, value in chunks.items()},
        },
        "chunking": {"max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens,
                     "tokenizer": TOKENIZER_NAME, "tokenizer_revision": TOKENIZER_REVISION},
        "thresholds_fit_on": "set span/presence and chunk-projection thresholds each fit on calibration contracts/chunks respectively; baseline thresholds fit on calibration chunks",
        "metrics_on": "held-out evaluation contracts only",
        "macro_f1_chunk_evaluation": pd.concat([baseline_metrics, set_metrics], ignore_index=True)
            .groupby("model")["f1"].mean().to_dict(),
        "macro_aupr_chunk_evaluation": pd.concat([baseline_metrics, set_metrics], ignore_index=True)
            .groupby("model")["aupr"].mean().to_dict(),
        "contract_presence_macro_aupr_evaluation": float(np.nanmean(compute_aupr(
            eval_contract_targets, eval_contract_scores,
        ))),
        "data_sha256": hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        "test_data_used": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n",
                                              encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
