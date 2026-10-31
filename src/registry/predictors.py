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


class EncoderPredictor:
    """Adapt the transformer chunk scorer to the common text-in/probabilities-out API."""

    def __init__(self, model: Any, tokenizer: Any, max_length: int, batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        from src.chunking.chunk import Chunk
        from src.models.encoder.predict import predict_proba

        chunks = [
            Chunk(contract_id="registry-inference", start=0, end=len(text),
                  text=text, chunk_index=index)
            for index, text in enumerate(texts)
        ]
        return predict_proba(
            self.model, self.tokenizer, chunks,
            max_length=self.max_length, batch_size=self.batch_size,
        )
