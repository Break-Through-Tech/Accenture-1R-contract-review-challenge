"""Shared helpers for contract-level encoder training experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.labels.categories import CATEGORIES
from src.labels.projection import LabeledChunk
from src.models.encoder.contract_dataset import collate_contract_sequences
from src.thresholds import find_best_thresholds

ROOT = Path(__file__).resolve().parents[3]
TRAIN_PATH = ROOT / "data/cuad/train_separate_questions.json"


def digest_contract_ids(contract_ids: list[str]) -> str:
    """Hash ordered contract IDs for split provenance."""
    return hashlib.sha256("\n".join(contract_ids).encode("utf-8")).hexdigest()


def calibrate_thresholds(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Fit per-category F1 thresholds on calibration rows; default unscorable labels to 0.5."""
    active = y_true.sum(axis=0) > 0
    tuned = find_best_thresholds(
        y_true[:, active], scores[:, active],
        [category for category, flag in zip(CATEGORIES, active) if flag],
    )
    return {category: float(tuned.get(category, 0.5)) for category in CATEGORIES}


def make_contract_loader(dataset, tokenizer, batch_size: int,
                         shuffle: bool = False, seed: int = 42) -> DataLoader:
    """Create a deterministic DataLoader that pads complete contract sequences."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda items: collate_contract_sequences(items, tokenizer.pad_token_id),
        generator=generator,
    )


def align_chunk_predictions(
    chunks: list[LabeledChunk], predictions: dict[tuple[str, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Align raw probabilities to the supplied labeled-chunk order."""
    y_true = np.asarray([item.labels for item in chunks], dtype=np.uint8)
    y_score = np.asarray([
        predictions[(item.chunk.contract_id, item.chunk.chunk_index)] for item in chunks
    ], dtype=np.float32)
    return y_true, y_score
