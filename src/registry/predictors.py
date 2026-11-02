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


class LongContextPredictor:
    """Score one contract's ordered chunk texts with the cross-chunk context model.

    A call must contain chunks from exactly one contract, in chunk-index order;
    cross-contract batching would create invalid recurrent context.
    """

    def __init__(self, model: Any, tokenizer: Any, max_length: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        import torch

        if not texts:
            return np.empty((0, self.model.num_labels), dtype=np.float32)
        encoded = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=False)
        token_count = max(len(row) for row in encoded["input_ids"])
        input_ids = torch.full((1, len(texts), token_count), self.tokenizer.pad_token_id,
                               dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for index, ids in enumerate(encoded["input_ids"]):
            input_ids[0, index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[0, index, :len(ids)] = 1
        chunk_mask = torch.ones((1, len(texts)), dtype=torch.bool)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids.to(device), attention.to(device),
                                chunk_mask.to(device)).logits[0]
            return torch.sigmoid(logits).cpu().numpy()


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
