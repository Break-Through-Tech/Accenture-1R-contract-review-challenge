"""Train/evaluate a graph neural network over chunks in each contract.

Nodes are encoded chunks. Edges connect adjacent chunks and nonlocal lexical neighbors
using a vectorizer fitted only on training chunks. The model uses weighted GraphSAGE
message passing and a multilabel chunk head. Split/calibration policy matches prior
experiments; test.json is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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
from src.models.encoder.contract_dataset import ContractChunkDataset, collate_contract_sequences
from src.models.encoder.contract_graph import ContractGraph, ContractGraphClassifier, build_contract_graphs
from src.models.encoder.experiment_utils import (
    ROOT, TRAIN_PATH, calibrate_thresholds, digest_contract_ids, align_chunk_predictions,
)
from src.models.naive.word_tfidf import fit_word_char_tfidf, predict_word_char_probabilities
from src.pipeline import prepare_contracts, to_model_inputs


class GraphContractDataset(ContractChunkDataset):
    """Tokenized contract dataset paired with its annotation-blind graph edges."""

    def __init__(self, chunks, tokenizer, max_length, graphs):
        super().__init__(chunks, tokenizer, max_length)
        for contract in self.contracts:
            contract["graph"] = graphs[contract["contract_id"]]
            contract["chunk_indices"] = [chunk.chunk_index for chunk in contract["chunks"]]


def _collate_graph_contracts(items, pad_token_id):
    if len(items) != 1:
        raise ValueError("graph batches must contain exactly one contract")
    graph = items[0]["graph"]
    tokenized = {key: value for key, value in items[0].items()
                 if key not in {"graph", "chunk_indices"}}
    batch = collate_contract_sequences([tokenized], pad_token_id)
    batch["edge_index"] = graph.edge_index
    batch["edge_weight"] = graph.edge_weight
    batch["chunk_indices"] = [items[0]["chunk_indices"]]
    return batch


def _make_loader(dataset, tokenizer, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset, batch_size=1, shuffle=shuffle, generator=generator,
        collate_fn=partial(_collate_graph_contracts, pad_token_id=tokenizer.pad_token_id),
    )


def _predict(model, loader, device, amp_enabled):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for batch in loader:
            model_batch = {
                key: batch[key].to(device)
                for key in ("input_ids", "attention_mask", "chunk_mask", "edge_index", "edge_weight")
            }
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits, _ = model(**model_batch)
            scores = torch.sigmoid(logits.float()).cpu().numpy()
            contract_id = batch["contract_ids"][0]
            for index, chunk_index in enumerate(batch["chunk_indices"][0]):
                predictions[(contract_id, chunk_index)] = scores[index]
    return predictions


def _metric_frame(y_true, scores, thresholds):
    threshold_vector = np.asarray([thresholds[category] for category in CATEGORIES])
    predicted = (scores >= threshold_vector).astype(np.uint8)
    metrics = compute_base_metrics(y_true, predicted, CATEGORIES)
    aupr = dict(zip(CATEGORIES, compute_aupr(y_true, scores)))
    metrics["aupr"] = metrics["category"].map(aupr)
    metrics["threshold"] = metrics["category"].map(thresholds)
    metrics.loc[metrics["support"] == 0, ["precision", "recall", "f1"]] = np.nan
    return metrics


def _save_checkpoint(path, model, tokenizer, vectorizer, config):
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(path / "encoder")
    tokenizer.save_pretrained(path / "encoder")
    head = {key: value.detach().cpu() for key, value in model.state_dict().items()
            if not key.startswith("encoder.")}
    torch.save(head, path / "graph_head.pt")
    joblib.dump(vectorizer, path / "graph_vectorizer.joblib")
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _load_checkpoint(path, device):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    encoder = AutoModel.from_pretrained(path / "encoder", local_files_only=True)
    model = ContractGraphClassifier(
        encoder, len(CATEGORIES), graph_hidden_size=config["graph_hidden_size"],
        graph_layers=config["graph_layers"],
        encoder_chunk_batch_size=config["encoder_chunk_batch_size"], dropout=config["dropout"],
    )
    head = torch.load(path / "graph_head.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(head, strict=False)
    if result.unexpected_keys or any(not key.startswith("encoder.") for key in result.missing_keys):
        raise ValueError(f"checkpoint head mismatch: {result}")
    return model.to(device).eval(), joblib.load(path / "graph_vectorizer.joblib"), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/contract_graph.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/contract_graph_seed42")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "outputs/contract_graph_seed42/best")
    parser.add_argument("--evaluate-checkpoint", type=Path)
    for name, kind in (
        ("seed", int), ("evaluation-fraction", float), ("calibration-fraction", float),
        ("max-tokens", int), ("overlap-tokens", int), ("encoder-chunk-batch-size", int),
        ("graph-hidden-size", int), ("graph-layers", int), ("semantic-neighbors", int),
        ("dropout", float), ("learning-rate", float), ("epochs", int),
        ("pos-weight-cap", float), ("c", float),
    ):
        parser.add_argument(f"--{name}", type=kind)
    args = parser.parse_args()

    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["model_name"] != TOKENIZER_NAME or config["model_revision"] != TOKENIZER_REVISION:
        parser.error("model_name and model_revision must match the shared pinned tokenizer")
    fields = (
        "seed", "evaluation_fraction", "calibration_fraction", "max_tokens", "overlap_tokens",
        "encoder_chunk_batch_size", "graph_hidden_size", "graph_layers", "semantic_neighbors",
        "dropout", "learning_rate", "epochs", "pos_weight_cap", "c",
    )
    for field in fields:
        if getattr(args, field) is None:
            setattr(args, field, config[field])
    if args.evaluate_checkpoint:
        saved = json.loads((args.evaluate_checkpoint / "config.json").read_text(encoding="utf-8"))
        for field in fields:
            setattr(args, field, saved[field])
    if args.epochs < 1 or args.graph_layers < 1 or args.encoder_chunk_batch_size < 1:
        parser.error("epochs, graph layers, and encoder chunk batch size must be positive")
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
    ids = {name: [item.contract_id for item in group] for name, group in groups.items()}
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
    train_texts, y_train = to_model_inputs(chunks["training"])
    labels = {name: np.asarray([item.labels for item in values], dtype=np.uint8)
              for name, values in chunks.items()}
    all_chunks = [item for group in chunks.values() for item in group]
    if args.evaluate_checkpoint:
        model, vectorizer, saved_config = _load_checkpoint(best_path, device)
        graph_by_split = _graphs_with_vectorizer(
            vectorizer, all_chunks, args.semantic_neighbors,
        )
    else:
        vectorizer, graph_by_split = build_contract_graphs(
            train_texts, all_chunks, semantic_neighbors=args.semantic_neighbors,
        )
        model = None
    datasets = {
        name: GraphContractDataset(values, tokenizer, args.max_tokens, graph_by_split)
        for name, values in chunks.items()
    }
    loaders = {
        name: _make_loader(dataset, tokenizer, shuffle=(name == "training"), seed=args.seed)
        for name, dataset in datasets.items()
    }

    history = []
    best_calibration_aupr = -1.0
    if args.evaluate_checkpoint:
        history = saved_config.get("history", [])
        best_calibration_aupr = saved_config.get("best_calibration_macro_aupr", -1.0)
    else:
        encoder = AutoModel.from_pretrained(
            TOKENIZER_NAME, revision=TOKENIZER_REVISION,
            cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
        )
        model = ContractGraphClassifier(
            encoder, len(CATEGORIES), graph_hidden_size=args.graph_hidden_size,
            graph_layers=args.graph_layers,
            encoder_chunk_batch_size=args.encoder_chunk_batch_size, dropout=args.dropout,
        ).to(device)
        if hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()
        pos = labels["training"].sum(axis=0)
        weights = np.divide(len(labels["training"]) - pos, pos,
                            out=np.ones_like(pos, dtype=float), where=pos > 0)
        pos_weight = torch.tensor(np.clip(weights, 1, args.pos_weight_cap),
                                  dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        for epoch in range(args.epochs):
            model.train()
            loss_total = 0.0
            for batch in loaders["training"]:
                model_batch = {
                    key: batch[key].to(device)
                    for key in ("input_ids", "attention_mask", "chunk_mask", "edge_index", "edge_weight")
                }
                targets = batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    _, loss = model(**model_batch, labels=targets, pos_weight=pos_weight)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                loss_total += float(loss.detach())

            calibration_predictions = _predict(model, loaders["calibration"], device, amp_enabled)
            y_cal, score_cal = align_chunk_predictions(chunks["calibration"], calibration_predictions)
            calibration_aupr = float(np.nanmean(compute_aupr(y_cal, score_cal)))
            record = {"epoch": epoch + 1,
                      "training_loss": loss_total / max(len(loaders["training"]), 1),
                      "calibration_macro_aupr": calibration_aupr}
            history.append(record)
            print(json.dumps(record), flush=True)
            if calibration_aupr > best_calibration_aupr:
                best_calibration_aupr = calibration_aupr
                _save_checkpoint(best_path, model, tokenizer, vectorizer, {
                    "architecture": "contract_chunk_weighted_graphsage",
                    "model_name": TOKENIZER_NAME, "model_revision": TOKENIZER_REVISION,
                    "seed": args.seed, "evaluation_fraction": args.evaluation_fraction,
                    "calibration_fraction": args.calibration_fraction,
                    "max_tokens": args.max_tokens, "overlap_tokens": args.overlap_tokens,
                    "encoder_chunk_batch_size": args.encoder_chunk_batch_size,
                    "graph_hidden_size": args.graph_hidden_size, "graph_layers": args.graph_layers,
                    "semantic_neighbors": args.semantic_neighbors, "dropout": args.dropout,
                    "learning_rate": args.learning_rate, "epochs": args.epochs,
                    "pos_weight_cap": args.pos_weight_cap, "c": args.c,
                    "best_calibration_macro_aupr": best_calibration_aupr,
                    "history": history,
                })
        model, vectorizer, saved_config = _load_checkpoint(best_path, device)

    prediction_maps = {
        name: _predict(model, loaders[name], device, amp_enabled)
        for name in ("calibration", "evaluation")
    }
    y_cal, score_cal = align_chunk_predictions(chunks["calibration"], prediction_maps["calibration"])
    y_eval, score_eval = align_chunk_predictions(chunks["evaluation"], prediction_maps["evaluation"])
    thresholds = calibrate_thresholds(y_cal, score_cal)
    graph_metrics = _metric_frame(y_eval, score_eval, thresholds)
    graph_metrics["model"] = "contract_graph_sage"

    features, baseline = fit_word_char_tfidf(train_texts, y_train, c=args.c)
    cal_texts = [item.chunk.text for item in chunks["calibration"]]
    eval_texts = [item.chunk.text for item in chunks["evaluation"]]
    baseline_cal = predict_word_char_probabilities(features, baseline, cal_texts)
    baseline_eval_scores = predict_word_char_probabilities(features, baseline, eval_texts)
    baseline_thresholds = calibrate_thresholds(y_cal, baseline_cal)
    baseline_metrics = _metric_frame(y_eval, baseline_eval_scores, baseline_thresholds)
    baseline_metrics["model"] = "word_char_tfidf"
    metrics = pd.concat([baseline_metrics, graph_metrics], ignore_index=True)
    metrics.to_csv(args.output / "per_category_metrics.csv", index=False)

    np.savez_compressed(args.output / "calibration_predictions.npz",
                        y_true=y_cal, graph_score=score_cal, baseline_score=baseline_cal)
    np.savez_compressed(args.output / "evaluation_predictions.npz",
                        y_true=y_eval, graph_score=score_eval, baseline_score=baseline_eval_scores)
    (args.output / "thresholds.json").write_text(json.dumps({
        "contract_graph_sage": thresholds, "word_char_tfidf": baseline_thresholds,
    }, indent=2) + "\n", encoding="utf-8")

    graph_edge_counts = {
        contract_id: int(graph.edge_index.shape[1])
        for contract_id, graph in graph_by_split.items()
    }
    report = {
        "architecture": "chunk-level pretrained encoder with weighted GraphSAGE message passing",
        "graph": {
            "adjacency_edges": "bidirectional immediate chunk neighbors",
            "semantic_edges": "top non-adjacent TF-IDF cosine neighbors per node",
            "semantic_neighbors_per_node": args.semantic_neighbors,
            "tfidf_fit_on": "training chunks only; labels unused in graph construction",
            "edge_count_by_contract": graph_edge_counts,
            "total_directed_edges": int(sum(graph_edge_counts.values())),
        },
        "training": {
            "epochs": len(history), "learning_rate": args.learning_rate,
            "graph_hidden_size": args.graph_hidden_size, "graph_layers": args.graph_layers,
            "encoder_chunk_batch_size": args.encoder_chunk_batch_size,
            "dropout": args.dropout, "pos_weight_cap": args.pos_weight_cap,
            "best_calibration_macro_aupr": best_calibration_aupr, "history": history,
            "device": str(device), "amp": amp_enabled,
        },
        "split": {
            "seed": args.seed,
            "contract_counts": {name: len(value) for name, value in groups.items()},
            "contract_id_sha256": {name: digest_contract_ids(ids[name]) for name in ids},
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


def _graphs_with_vectorizer(vectorizer, chunks, semantic_neighbors):
    """Build contract graphs with a checkpoint's already-fitted train-only vocabulary."""
    from collections import defaultdict

    from src.models.encoder.contract_graph import ContractGraph

    grouped = defaultdict(list)
    for item in chunks:
        grouped[item.chunk.contract_id].append(item)
    graphs = {}
    for contract_id, items in grouped.items():
        items.sort(key=lambda item: item.chunk.chunk_index)
        count = len(items)
        edges = {}
        for index in range(count - 1):
            edges[(index, index + 1)] = 1.0
            edges[(index + 1, index)] = 1.0
        if semantic_neighbors and count > 2:
            matrix = vectorizer.transform([item.chunk.text for item in items])
            similarities = (matrix @ matrix.T).toarray()
            for source in range(count):
                ranked = [
                    (float(similarities[source, target]), target)
                    for target in range(count)
                    if abs(source - target) > 1 and similarities[source, target] > 0
                ]
                ranked.sort(key=lambda pair: (-pair[0], pair[1]))
                for similarity, target in ranked[:semantic_neighbors]:
                    edges[(source, target)] = max(edges.get((source, target), 0.0), similarity)
                    edges[(target, source)] = max(edges.get((target, source), 0.0), similarity)
        ordered = sorted(edges.items())
        if ordered:
            edge_index = torch.tensor([[edge[0][0] for edge in ordered],
                                       [edge[0][1] for edge in ordered]], dtype=torch.long)
            edge_weight = torch.tensor([edge[1] for edge in ordered], dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)
        graphs[contract_id] = ContractGraph(edge_index, edge_weight)
    return graphs


if __name__ == "__main__":
    main()
