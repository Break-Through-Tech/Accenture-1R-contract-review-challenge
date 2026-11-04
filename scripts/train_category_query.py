"""Train a category-query contract model with chunk evidence attention.

Each CUAD category is a description-initialized query that attends over every
chunk in the contract. The model is supervised for chunk labels, contract-level
presence, and attention on gold evidence chunks. Uses the fixed contract split and
never reads test.json.
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
from transformers import AutoModel, AutoTokenizer

from src.models.encoder.experiment_utils import (
    ROOT, TRAIN_PATH, align_chunk_predictions, calibrate_thresholds,
    digest_contract_ids, make_contract_loader,
)
from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.labels.splits import split_by_contract
from src.metrics import compute_aupr, compute_base_metrics
from src.models.encoder.category_descriptions import load_category_descriptions
from src.models.encoder.category_query import CategoryQueryClassifier, build_category_query_model
from src.models.encoder.contract_dataset import ContractChunkDataset
from src.pipeline import prepare_contracts
from src.registry.predictors import CategoryQueryPredictor


DESCRIPTIONS_PATH = ROOT / "data/cuad/category_descriptions.csv"


def _predict(model, loader, device, amp_dtype):
    model.eval()
    scores, attentions = {}, {}
    with torch.no_grad():
        for batch in loader:
            model_batch = {
                key: batch[key].to(device)
                for key in ("input_ids", "attention_mask", "chunk_mask")
            }
            with torch.autocast(device_type="cuda", dtype=amp_dtype or torch.float16,
                                enabled=amp_dtype is not None):
                output = model(**model_batch)
            logits = torch.sigmoid(output.logits).float().cpu().numpy()
            weights = output.attention_weights.float().cpu().numpy()
            valid = batch["chunk_mask"].numpy()
            for row, contract_id in enumerate(batch["contract_ids"]):
                count = int(valid[row].sum())
                for index in range(count):
                    scores[(contract_id, index)] = logits[row, index]
                attentions[contract_id] = weights[row, :, :count]
    return scores, attentions


def _evidence_metrics(chunks, attentions):
    hits, attention_mass = [], []
    by_contract: dict[str, dict[int, np.ndarray]] = {}
    for item in chunks:
        by_contract.setdefault(item.chunk.contract_id, {})[item.chunk.chunk_index] = np.asarray(item.labels)
    for contract_id, rows in by_contract.items():
        indices = sorted(rows)
        labels = np.asarray([rows[index] for index in indices], dtype=bool)
        weights = attentions[contract_id]
        for category_index in range(len(CATEGORIES)):
            positives = np.flatnonzero(labels[:, category_index])
            if not positives.size:
                continue
            category_attention = weights[category_index]
            hits.append(int(int(category_attention.argmax()) in positives))
            attention_mass.append(float(category_attention[positives].sum()))
    return {
        "positive_contract_category_count": len(hits),
        "attention_evidence_hit_at_1": float(np.mean(hits)) if hits else None,
        "mean_attention_mass_on_positive_chunks": float(np.mean(attention_mass)) if attention_mass else None,
    }


def _save_checkpoint(model, tokenizer, path: Path, config: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(path / "encoder")
    tokenizer.save_pretrained(path / "encoder")
    head = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("encoder.")
    }
    torch.save(head, path / "query_head.pt")
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def _load_checkpoint(path: Path, device):
    config = json.loads((path / "config.json").read_text())
    encoder = AutoModel.from_pretrained(path / "encoder")
    query_init = torch.zeros((len(CATEGORIES), encoder.config.hidden_size))
    model = CategoryQueryClassifier(
        encoder, query_init,
        encoder_chunk_batch_size=config["encoder_chunk_batch_size"],
        dropout=config["dropout"], context_layers=config["context_layers"],
        attention_heads=config["attention_heads"],
        contract_loss_weight=config["contract_loss_weight"],
        evidence_loss_weight=config["evidence_loss_weight"],
    )
    head = torch.load(path / "query_head.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(head, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"unexpected checkpoint keys: {result.unexpected_keys}")
    if any(not key.startswith("encoder.") for key in result.missing_keys):
        raise ValueError(f"missing query-head keys: {result.missing_keys}")
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/category_query.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/category_query_seed42")
    parser.add_argument("--evaluate-checkpoint", type=Path,
                        help="Skip training and score an existing best checkpoint")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--pos-weight-cap", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--context-layers", type=int)
    parser.add_argument("--attention-heads", type=int)
    parser.add_argument("--contracts-per-batch", type=int)
    parser.add_argument("--encoder-chunk-batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--overlap-tokens", type=int)
    parser.add_argument("--evaluation-fraction", type=float)
    parser.add_argument("--calibration-fraction", type=float)
    parser.add_argument("--contract-loss-weight", type=float)
    parser.add_argument("--evidence-loss-weight", type=float)
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fields = (
        "seed", "epochs", "learning_rate", "pos_weight_cap", "dropout",
        "context_layers", "attention_heads", "contracts_per_batch",
        "encoder_chunk_batch_size", "gradient_accumulation", "max_tokens",
        "overlap_tokens", "evaluation_fraction", "calibration_fraction",
        "contract_loss_weight", "evidence_loss_weight",
    )
    for key in fields:
        if getattr(args, key) is None:
            setattr(args, key, config[key])
    if config["model_name"] != TOKENIZER_NAME or config["model_revision"] != TOKENIZER_REVISION:
        parser.error("model_name and model_revision must match the pinned shared tokenizer")
    if args.epochs < 1 or args.contracts_per_batch != 1 or args.gradient_accumulation < 1:
        parser.error("epochs and gradient accumulation must be positive; use one contract per batch")
    if not 0 < args.evaluation_fraction < 1 or not 0 < args.calibration_fraction < 1:
        parser.error("split fractions must be between zero and one")
    if args.max_tokens > 512 or not 0 <= args.overlap_tokens < args.max_tokens - 2:
        parser.error("require max_tokens <= 512 and overlap_tokens < max_tokens - 2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" else None

    contracts = load_cuad(TRAIN_PATH)
    development, evaluation = split_by_contract(contracts, args.evaluation_fraction, args.seed)
    training, calibration = split_by_contract(development, args.calibration_fraction, args.seed)
    groups = {"training": training, "calibration": calibration, "evaluation": evaluation}
    ids = {name: [contract.contract_id for contract in group] for name, group in groups.items()}
    sets = [set(values) for values in ids.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("contract split leakage detected")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )
    descriptions = load_category_descriptions(DESCRIPTIONS_PATH)
    chunks = {
        name: prepare_contracts(group, tokenizer, args.max_tokens, args.overlap_tokens)
        for name, group in groups.items()
    }
    datasets = {name: ContractChunkDataset(value, tokenizer, args.max_tokens)
                for name, value in chunks.items()}
    loaders = {
        name: make_contract_loader(
            dataset, tokenizer, args.contracts_per_batch,
            shuffle=(name == "training"), seed=args.seed,
        ) for name, dataset in datasets.items()
    }
    train_chunk_labels = np.asarray([item.labels for item in chunks["training"]], dtype=np.float32)
    chunk_pos = train_chunk_labels.sum(axis=0)
    chunk_weights = np.divide(
        len(train_chunk_labels) - chunk_pos, chunk_pos,
        out=np.ones_like(chunk_pos), where=chunk_pos > 0,
    )
    train_contract_labels = np.asarray([
        np.asarray(item["labels"], dtype=np.float32).max(axis=0)
        for item in datasets["training"].contracts
    ])
    contract_pos = train_contract_labels.sum(axis=0)
    contract_weights = np.divide(
        len(train_contract_labels) - contract_pos, contract_pos,
        out=np.ones_like(contract_pos), where=contract_pos > 0,
    )
    chunk_pos_weight = torch.tensor(
        np.clip(chunk_weights, 1, args.pos_weight_cap), dtype=torch.float32, device=device,
    )
    contract_pos_weight = torch.tensor(
        np.clip(contract_weights, 1, args.pos_weight_cap), dtype=torch.float32, device=device,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    best_path = args.evaluate_checkpoint or args.output / "best"
    history = []
    best_aupr = -1.0
    if args.evaluate_checkpoint:
        model = _load_checkpoint(best_path, device)
        saved_config = json.loads((best_path / "config.json").read_text())
        args.epochs = saved_config.get("epochs", args.epochs)
        history = saved_config.get("history", [])
        best_aupr = saved_config.get("best_calibration_macro_aupr", best_aupr)
    else:
        model = build_category_query_model(
            TOKENIZER_NAME, TOKENIZER_REVISION, tokenizer, descriptions,
            encoder_chunk_batch_size=args.encoder_chunk_batch_size,
            dropout=args.dropout, context_layers=args.context_layers,
            attention_heads=args.attention_heads,
            contract_loss_weight=args.contract_loss_weight,
            evidence_loss_weight=args.evidence_loss_weight,
        ).to(device)
        if hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is not None)

    for epoch in range(0 if args.evaluate_checkpoint else args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        steps = 0
        for step, batch in enumerate(loaders["training"], start=1):
            model_batch = {
                key: batch[key].to(device)
                for key in ("input_ids", "attention_mask", "chunk_mask")
            }
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype or torch.float16,
                                enabled=amp_dtype is not None):
                output = model(
                    **model_batch, labels=labels,
                    chunk_pos_weight=chunk_pos_weight,
                    contract_pos_weight=contract_pos_weight,
                )
                loss = output.loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            loss_total += float(loss.detach()) * args.gradient_accumulation
            steps += 1
            if step % args.gradient_accumulation == 0 or step == len(loaders["training"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        calibration_scores, _ = _predict(model, loaders["calibration"], device, amp_dtype)
        y_cal, score_cal = align_chunk_predictions(chunks["calibration"], calibration_scores)
        aupr = float(np.nanmean(compute_aupr(y_cal, score_cal)))
        record = {"epoch": epoch + 1, "train_loss": loss_total / max(steps, 1),
                  "calibration_macro_aupr": aupr}
        history.append(record)
        print(json.dumps(record), flush=True)
        if aupr > best_aupr:
            best_aupr = aupr
            _save_checkpoint(model, tokenizer, best_path, {
                "architecture": "description_initialized_category_query_contract_model",
                "model_name": TOKENIZER_NAME, "model_revision": TOKENIZER_REVISION,
                "max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens,
                "encoder_chunk_batch_size": args.encoder_chunk_batch_size,
                "dropout": args.dropout, "context_layers": args.context_layers,
                "attention_heads": args.attention_heads,
                "contract_loss_weight": args.contract_loss_weight,
                "evidence_loss_weight": args.evidence_loss_weight,
                "num_labels": len(CATEGORIES), "seed": args.seed,
            })

    if not args.evaluate_checkpoint:
        saved_config = json.loads((best_path / "config.json").read_text())
        saved_config["epochs"] = args.epochs
        saved_config["best_calibration_macro_aupr"] = best_aupr
        saved_config["history"] = history
        (best_path / "config.json").write_text(json.dumps(saved_config, indent=2) + "\n")
        model = _load_checkpoint(best_path, device)

    cal_scores, _ = _predict(model, loaders["calibration"], device, amp_dtype)
    eval_scores, eval_attention = _predict(model, loaders["evaluation"], device, amp_dtype)
    y_cal, score_cal = align_chunk_predictions(chunks["calibration"], cal_scores)
    y_eval, score_eval = align_chunk_predictions(chunks["evaluation"], eval_scores)
    if args.evaluate_checkpoint:
        best_aupr = float(np.nanmean(compute_aupr(y_cal, score_cal)))
    thresholds = calibrate_thresholds(y_cal, score_cal)
    predictions = (score_eval >= np.asarray([thresholds[c] for c in CATEGORIES])).astype(np.uint8)
    metrics = compute_base_metrics(y_eval, predictions, CATEGORIES)
    metrics["aupr"] = compute_aupr(y_eval, score_eval)
    metrics["threshold"] = metrics["category"].map(thresholds)
    metrics.loc[metrics["support"] == 0, ["precision", "recall", "f1"]] = np.nan
    metrics.to_csv(args.output / "per_category_metrics.csv", index=False)
    np.savez_compressed(args.output / "calibration_predictions.npz", y_true=y_cal, y_score=score_cal)
    np.savez_compressed(args.output / "evaluation_predictions.npz", y_true=y_eval, y_score=score_eval)
    (best_path / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")

    summary = {
        "architecture": "description-initialized category queries with contract-wide cross-attention",
        "contract_counts": {name: len(value) for name, value in groups.items()},
        "contract_id_sha256": {name: digest_contract_ids(value) for name, value in ids.items()},
        "chunk_counts": {name: len(value) for name, value in chunks.items()},
        "chunking": {"method": "sentence_token", "max_tokens": args.max_tokens,
                     "overlap_tokens": args.overlap_tokens,
                     "tokenizer_revision": TOKENIZER_REVISION},
        "training": {"epochs": args.epochs, "learning_rate": args.learning_rate,
                     "pos_weight_cap": args.pos_weight_cap, "dropout": args.dropout,
                     "context_layers": args.context_layers,
                     "attention_heads": args.attention_heads,
                     "contract_loss_weight": args.contract_loss_weight,
                     "evidence_loss_weight": args.evidence_loss_weight,
                     "contracts_per_batch": args.contracts_per_batch,
                     "encoder_chunk_batch_size": args.encoder_chunk_batch_size,
                     "gradient_accumulation": args.gradient_accumulation,
                     "device": str(device), "amp": amp_dtype is not None,
                     "amp_dtype": str(amp_dtype) if amp_dtype else None,
                     "best_calibration_macro_aupr": best_aupr, "history": history},
        "evidence_localization": _evidence_metrics(chunks["evaluation"], eval_attention),
        "thresholds_fit_on": "calibration contracts only",
        "metrics_on": "held-out evaluation contracts only",
        "macro_metrics": metrics[["precision", "recall", "f1", "aupr"]].mean(skipna=True).to_dict(),
        "data_sha256": hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        "category_descriptions_sha256": hashlib.sha256(DESCRIPTIONS_PATH.read_bytes()).hexdigest(),
        "test_data_used": False,
    }
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    if args.register:
        from src.registry.bundle import ClauseModelBundle
        from src.registry.mlflow_registry import register_bundle

        model = model.to("cpu").eval()
        bundle = ClauseModelBundle(
            predictor=CategoryQueryPredictor(model, tokenizer, args.max_tokens),
            categories=CATEGORIES, thresholds=thresholds,
            model_name="distilroberta_category_query_contract",
            config=summary["training"] | summary["chunking"], seed=args.seed,
        )
        registered = register_bundle(
            bundle, experiment_name="clause-review-category-query",
            registry_name="clause-classifier-category-query",
            metrics={f"evaluation_{name}": float(value)
                     for name, value in summary["macro_metrics"].items()},
            metadata=summary, tracking_uri=ROOT / "outputs/mlflow/tracking.db",
        )
        summary["registry"] = {"run_id": registered.run_id,
                               "registry_name": registered.registry_name,
                               "model_version": registered.model_version,
                               "model_key": registered.model_key}
        (args.output / "report.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
