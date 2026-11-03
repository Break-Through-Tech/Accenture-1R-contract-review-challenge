"""Train and evaluate a chunk-encoder + contract-level BiGRU clause model.

Each sentence-aware chunk is encoded by DistilRoBERTa, then a bidirectional GRU
contextualizes the ordered chunk sequence for its entire contract and predicts all
41 labels per chunk. Uses the nested 260/66/82 contract split; thresholds are
calibrated on 66 contracts and final metrics use the disjoint 82. Never reads test.json.
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
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.labels.splits import split_by_contract
from src.metrics import compute_aupr, compute_base_metrics
from src.models.encoder.contract_dataset import ContractChunkDataset, collate_contract_sequences
from src.models.encoder.long_context import LongContextClassifier, build_long_context_model
from src.pipeline import prepare_contracts
from src.thresholds import apply_thresholds, find_best_thresholds

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/cuad/train_separate_questions.json"


def _digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _thresholds(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    active = y_true.sum(axis=0) > 0
    tuned = find_best_thresholds(
        y_true[:, active], scores[:, active],
        [category for category, flag in zip(CATEGORIES, active) if flag],
    )
    return {category: float(tuned.get(category, 0.5)) for category in CATEGORIES}


def _make_loader(dataset, tokenizer, batch_size: int, shuffle: bool = False, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda items: collate_contract_sequences(items, tokenizer.pad_token_id),
        generator=generator if shuffle else None,
    )


def _predict_dataset(model, loader, device, amp_dtype) -> dict[tuple[str, int], np.ndarray]:
    model.eval()
    results = {}
    with torch.no_grad():
        for batch in loader:
            model_batch = {key: batch[key].to(device) for key in ("input_ids", "attention_mask", "chunk_mask")}
            with torch.autocast(device_type="cuda", dtype=amp_dtype or torch.float16,
                                 enabled=amp_dtype is not None):
                output = model(**model_batch)
            probabilities = torch.sigmoid(output.logits).float().cpu().numpy()
            mask = batch["chunk_mask"].numpy()
            for row, contract_id in enumerate(batch["contract_ids"]):
                for index in range(int(mask[row].sum())):
                    results[(contract_id, index)] = probabilities[row, index]
    return results


def _aligned_arrays(chunks, predictions):
    y_true = np.asarray([item.labels for item in chunks], dtype=np.uint8)
    y_score = np.asarray([
        predictions[(item.chunk.contract_id, item.chunk.chunk_index)] for item in chunks
    ], dtype=np.float32)
    return y_true, y_score


def _save_best(model, tokenizer, destination: Path, config: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(destination / "encoder")
    tokenizer.save_pretrained(destination / "encoder")
    torch.save({
        "context": model.context.state_dict(),
        "position_embedding": model.position_embedding.state_dict()
        if model.position_embedding is not None else None,
        "classifier": model.classifier.state_dict(),
        "dropout": config["dropout"],
        "num_labels": model.num_labels,
    }, destination / "context_head.pt")
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def _load_checkpoint(checkpoint: Path, device):
    metadata = json.loads((checkpoint / "config.json").read_text())
    encoder = AutoModel.from_pretrained(checkpoint / "encoder")
    model = LongContextClassifier(
        encoder, encoder.config.hidden_size, len(CATEGORIES), metadata["dropout"],
        metadata["encoder_chunk_batch_size"], metadata["context_type"],
        metadata["context_layers"], metadata["attention_heads"],
    )
    head = torch.load(checkpoint / "context_head.pt", map_location="cpu", weights_only=True)
    model.context.load_state_dict(head["context"])
    if model.position_embedding is not None:
        model.position_embedding.load_state_dict(head["position_embedding"])
    model.classifier.load_state_dict(head["classifier"])
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/long_context.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/long_context_seed42")
    parser.add_argument("--evaluate-checkpoint", type=Path,
                        help="Skip training and evaluate an existing best checkpoint")
    parser.add_argument("--register", action="store_true", help="Register best checkpoint with local MLflow")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--pos-weight-cap", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--context-type", choices=["bigru", "transformer"])
    parser.add_argument("--context-layers", type=int)
    parser.add_argument("--attention-heads", type=int)
    parser.add_argument("--contracts-per-batch", type=int)
    parser.add_argument("--encoder-chunk-batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--overlap-tokens", type=int)
    parser.add_argument("--evaluation-fraction", type=float)
    parser.add_argument("--calibration-fraction", type=float)
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config_keys = {
        "seed": "seed", "epochs": "epochs", "learning_rate": "learning_rate",
        "contracts_per_batch": "contracts_per_batch",
        "encoder_chunk_batch_size": "encoder_chunk_batch_size",
        "gradient_accumulation": "gradient_accumulation", "max_tokens": "max_tokens",
        "overlap_tokens": "overlap_tokens", "evaluation_fraction": "evaluation_fraction",
        "calibration_fraction": "calibration_fraction", "pos_weight_cap": "pos_weight_cap",
        "dropout": "dropout", "context_type": "context_type",
        "context_layers": "context_layers", "attention_heads": "attention_heads",
    }
    for key, attr in config_keys.items():
        if getattr(args, attr) is None:
            setattr(args, attr, config[key])
    if config.get("model_name") != TOKENIZER_NAME or config.get("model_revision") != TOKENIZER_REVISION:
        parser.error("model_name and model_revision must match the pinned shared chunker tokenizer")
    if args.epochs < 1 or args.contracts_per_batch < 1 or args.gradient_accumulation < 1:
        parser.error("epochs, contracts-per-batch, and gradient-accumulation must be positive")
    if not 0 < args.calibration_fraction < 1 or not 0 < args.evaluation_fraction < 1:
        parser.error("split fractions must be between zero and one")
    if args.max_tokens > 512 or not 0 <= args.overlap_tokens < args.max_tokens - 2:
        parser.error("require max_tokens <=512 and overlap_tokens < max_tokens - 2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    # fp16 matches the tested CUDA path; BF16 triggered cuBLAS failures on this host.
    amp_dtype = torch.float16 if use_amp else None

    contracts = load_cuad(TRAIN_PATH)
    development, evaluation = split_by_contract(contracts, args.evaluation_fraction, args.seed)
    training, calibration = split_by_contract(development, args.calibration_fraction, args.seed)
    groups = {"training": training, "calibration": calibration, "evaluation": evaluation}
    ids = {name: [contract.contract_id for contract in group] for name, group in groups.items()}
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
    datasets = {name: ContractChunkDataset(value, tokenizer, args.max_tokens)
                for name, value in chunks.items()}
    loaders = {
        name: _make_loader(dataset, tokenizer, args.contracts_per_batch,
                           shuffle=(name == "training"), seed=args.seed)
        for name, dataset in datasets.items()
    }
    train_labels = np.asarray([item.labels for item in chunks["training"]], dtype=np.float32)
    positives = train_labels.sum(axis=0)
    weights = np.divide(len(train_labels) - positives, positives,
                        out=np.ones_like(positives), where=positives > 0)
    pos_weight = torch.tensor(
        np.clip(weights, 1.0, args.pos_weight_cap), dtype=torch.float32, device=device,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    best_aupr = -1.0
    best_path = args.evaluate_checkpoint or args.output / "best"
    history = []
    if args.evaluate_checkpoint:
        model = _load_checkpoint(best_path, device)
        checkpoint_config = json.loads((best_path / "config.json").read_text())
        args.context_type = checkpoint_config["context_type"]
        args.context_layers = checkpoint_config["context_layers"]
        args.attention_heads = checkpoint_config["attention_heads"]
        args.dropout = checkpoint_config["dropout"]
        args.encoder_chunk_batch_size = checkpoint_config["encoder_chunk_batch_size"]
        args.epochs = checkpoint_config.get("epochs", args.epochs)
        best_aupr = checkpoint_config.get("best_calibration_macro_aupr", best_aupr)
    else:
        model = build_long_context_model(
            TOKENIZER_NAME, TOKENIZER_REVISION, len(CATEGORIES), args.dropout,
            args.encoder_chunk_batch_size, args.context_type,
            args.context_layers, args.attention_heads,
        ).to(device)
        if hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=use_amp and amp_dtype == torch.float16,
        )

    for epoch in range(0 if args.evaluate_checkpoint else args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        step_count = 0
        for step, batch in enumerate(loaders["training"], start=1):
            model_batch = {key: batch[key].to(device) for key in ("input_ids", "attention_mask", "chunk_mask")}
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype or torch.float16,
                                 enabled=use_amp):
                output = model(**model_batch, labels=labels, pos_weight=pos_weight)
                loss = output.loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            loss_total += float(loss.detach()) * args.gradient_accumulation
            step_count += 1
            if step % args.gradient_accumulation == 0 or step == len(loaders["training"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        calibration_predictions = _predict_dataset(model, loaders["calibration"], device, amp_dtype)
        y_cal, score_cal = _aligned_arrays(chunks["calibration"], calibration_predictions)
        aupr = float(np.nanmean(compute_aupr(y_cal, score_cal)))
        epoch_record = {"epoch": epoch + 1, "train_loss": loss_total / max(step_count, 1),
                        "calibration_macro_aupr": aupr}
        history.append(epoch_record)
        print(json.dumps(epoch_record))
        if aupr > best_aupr:
            best_aupr = aupr
            _save_best(model, tokenizer, best_path, {
                "architecture": f"chunk_encoder_{args.context_type}_contract_context",
                "model_name": TOKENIZER_NAME, "model_revision": TOKENIZER_REVISION,
                "max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens,
                "dropout": args.dropout,
                "encoder_chunk_batch_size": args.encoder_chunk_batch_size,
                "context_type": args.context_type,
                "context_layers": args.context_layers,
                "attention_heads": args.attention_heads,
                "hidden_size": model.encoder.config.hidden_size,
                "num_labels": len(CATEGORIES), "seed": args.seed,
            })

    if not args.evaluate_checkpoint:
        checkpoint_config = json.loads((best_path / "config.json").read_text())
        checkpoint_config["epochs"] = args.epochs
        checkpoint_config["best_calibration_macro_aupr"] = best_aupr
        (best_path / "config.json").write_text(json.dumps(checkpoint_config, indent=2) + "\n")
        model = _load_checkpoint(best_path, device)
    cal_predictions = _predict_dataset(model, loaders["calibration"], device, amp_dtype)
    eval_predictions = _predict_dataset(model, loaders["evaluation"], device, amp_dtype)
    y_cal, score_cal = _aligned_arrays(chunks["calibration"], cal_predictions)
    y_eval, score_eval = _aligned_arrays(chunks["evaluation"], eval_predictions)
    if args.evaluate_checkpoint:
        best_aupr = float(np.nanmean(compute_aupr(y_cal, score_cal)))
    thresholds = _thresholds(y_cal, score_cal)
    predictions = apply_thresholds(score_eval, CATEGORIES, thresholds)
    report = compute_base_metrics(y_eval, predictions, CATEGORIES)
    report["aupr"] = compute_aupr(y_eval, score_eval)
    report["threshold"] = report["category"].map(thresholds)
    report.loc[report["support"] == 0, ["precision", "recall", "f1"]] = np.nan
    report.to_csv(args.output / "per_category_metrics.csv", index=False)
    np.savez_compressed(args.output / "calibration_predictions.npz", y_true=y_cal, y_score=score_cal)
    np.savez_compressed(args.output / "evaluation_predictions.npz", y_true=y_eval, y_score=score_eval)
    (best_path / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    summary = {
        "architecture": f"DistilRoBERTa chunk encoder + {args.context_type} over full contract chunk sequence",
        "contract_counts": {name: len(group) for name, group in groups.items()},
        "contract_id_sha256": {name: _digest(value) for name, value in ids.items()},
        "chunk_counts": {name: len(value) for name, value in chunks.items()},
        "chunking": {"method": "sentence_token", "max_tokens": args.max_tokens,
                     "overlap_tokens": args.overlap_tokens, "tokenizer_revision": TOKENIZER_REVISION},
        "training": {"epochs": args.epochs, "learning_rate": args.learning_rate,
                     "pos_weight_cap": args.pos_weight_cap, "dropout": args.dropout,
                     "context_type": args.context_type, "context_layers": args.context_layers,
                     "attention_heads": args.attention_heads,
                     "contracts_per_batch": args.contracts_per_batch,
                     "encoder_chunk_batch_size": args.encoder_chunk_batch_size,
                     "gradient_accumulation": args.gradient_accumulation,
                     "device": str(device), "amp": use_amp,
                     "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
                     "best_calibration_macro_aupr": best_aupr,
                     "history": history},
        "thresholds_fit_on": "calibration contracts only",
        "metrics_on": "held-out evaluation contracts only",
        "macro_metrics": report[["precision", "recall", "f1", "aupr"]].mean(skipna=True).to_dict(),
        "data_sha256": hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        "test_data_used": False,
    }
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    if args.register:
        from src.registry.bundle import ClauseModelBundle
        from src.registry.mlflow_registry import register_bundle
        from src.registry.predictors import LongContextPredictor

        # Registry artifacts must load on CPU even when training ran on CUDA.
        model = model.to("cpu").eval()
        bundle = ClauseModelBundle(
            predictor=LongContextPredictor(model, tokenizer, args.max_tokens),
            categories=CATEGORIES, thresholds=thresholds,
            model_name="distilroberta_bilstm_contract_context",
            config=summary["training"] | summary["chunking"], seed=args.seed,
        )
        registered = register_bundle(
            bundle, experiment_name="clause-review-long-context",
            registry_name="clause-classifier", metrics={
                f"evaluation_{name}": float(value)
                for name, value in summary["macro_metrics"].items()
            }, metadata=summary, tracking_uri=ROOT / "outputs/mlflow/tracking.db",
        )
        summary["registry"] = {"run_id": registered.run_id,
                               "model_version": registered.model_version}
        (args.output / "report.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
