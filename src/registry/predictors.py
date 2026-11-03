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
    cross-contract batching would mix unrelated long-range context.
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


class CategoryQueryPredictor(LongContextPredictor):
    """Category-query adapter that also exposes per-chunk evidence attention."""

    def predict_with_evidence(self, texts: list[str]) -> dict[str, np.ndarray]:
        import torch

        if not texts:
            empty_scores = np.empty((0, self.model.num_labels), dtype=np.float32)
            return {
                "probabilities": empty_scores,
                "attention_weights": empty_scores.copy(),
                "top_evidence_chunk_indices": np.full(self.model.num_labels, -1, dtype=np.int64),
            }
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
            output = self.model(input_ids.to(device), attention.to(device), chunk_mask.to(device))
            probabilities = torch.sigmoid(output.logits[0]).cpu().numpy()
            weights = output.attention_weights[0].transpose(0, 1).cpu().numpy()
        return {
            "probabilities": probabilities,
            "attention_weights": weights,
            "top_evidence_chunk_indices": weights.argmax(axis=0),
        }

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.predict_with_evidence(texts)["probabilities"]


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
