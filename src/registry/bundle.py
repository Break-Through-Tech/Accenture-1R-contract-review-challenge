"""Stable inference wrapper and versioned artifact bundle for clause models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.labels.categories import CATEGORIES


@dataclass
class ClauseModelBundle:
    """Model plus ordered label schema and calibrated thresholds.

    Predictor objects must expose ``predict_proba(texts)`` and return one
    probability column per category in ``categories`` order.
    """

    predictor: Any
    categories: list[str]
    thresholds: dict[str, float]
    model_name: str
    config: dict[str, Any]
    seed: int

    def __post_init__(self) -> None:
        if self.categories != CATEGORIES:
            raise ValueError("categories must exactly match the canonical CUAD ordering")
        if set(self.thresholds) != set(self.categories):
            raise ValueError("thresholds must contain exactly one value per category")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.thresholds.values()):
            raise ValueError("threshold values must be in [0, 1]")

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return raw probabilities, preserving category order for downstream use."""
        probs = np.asarray(self.predictor.predict_proba(texts))
        expected = (len(texts), len(self.categories))
        if probs.shape != expected:
            raise ValueError(f"predictor returned shape {probs.shape}; expected {expected}")
        if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
            raise ValueError("predictor probabilities must be finite and within [0, 1]")
        return probs

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return thresholded multi-label predictions."""
        probs = self.predict_proba(texts)
        cutoff = np.asarray([self.thresholds[c] for c in self.categories])
        return (probs >= cutoff).astype(np.uint8)
