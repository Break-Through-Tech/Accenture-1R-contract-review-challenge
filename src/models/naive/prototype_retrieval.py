"""Sparse positive-exemplar retrieval over word/character TF-IDF chunks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.models.naive.word_tfidf import build_word_char_pipeline


class PositivePrototypeRetriever:
    """Rank categories by cosine similarity to their nearest positive train chunk.

    This is a nonparametric retrieval architecture. It stores only positive
    training vectors per category and exposes max cosine similarities in [0, 1];
    thresholds must be calibrated on a separate contract split.
    """

    def __init__(self, features, train_matrix, labels: np.ndarray):
        self.features = features
        self.positive_vectors = [
            train_matrix[labels[:, index] == 1] for index in range(labels.shape[1])
        ]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return each chunk's maximum positive-exemplar similarity per category."""
        if not texts:
            return np.empty((0, len(self.positive_vectors)), dtype=float)
        matrix = self._transform(texts)
        scores = np.zeros((len(texts), len(self.positive_vectors)), dtype=np.float32)
        for index, examples in enumerate(self.positive_vectors):
            if examples.shape[0]:
                similarities = cosine_similarity(matrix, examples, dense_output=True)
                scores[:, index] = similarities.max(axis=1)
        return scores

    def _transform(self, texts: list[str]):
        return self.features.transform(texts)


def fit_positive_prototype_retriever(texts: list[str], labels: np.ndarray):
    """Fit shared word/character TF-IDF and retain positive train examples only."""
    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must have the same non-zero length")
    features, _ = build_word_char_pipeline()
    matrix = features.fit_transform(texts)
    return PositivePrototypeRetriever(features, matrix, labels)
