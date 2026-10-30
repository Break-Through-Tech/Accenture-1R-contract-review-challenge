"""Adapters that expose project classifiers through the registry inference API."""

from __future__ import annotations

from typing import Any

import numpy as np


class VectorizedClassifier:
    """Adapt a fitted sklearn vectorizer and classifier to ``predict_proba(texts)``."""

    def __init__(self, vectorizer: Any, classifier: Any):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, len(self.classifier.estimators_)), dtype=float)
        return np.asarray(self.classifier.predict_proba(self.vectorizer.transform(texts)))
