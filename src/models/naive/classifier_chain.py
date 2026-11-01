"""Classifier-chain variant modeling dependencies between clause categories."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain

from src.models.naive.word_tfidf import build_word_char_pipeline


class WordCharChain:
    """Chain for learnable labels plus fixed outputs for constant train labels."""

    def __init__(self, chain, active_indices: np.ndarray, constant_values: np.ndarray, n_categories: int):
        self.chain = chain
        self.active_indices = active_indices
        self.constant_values = constant_values
        self.n_categories = n_categories

    @property
    def estimators_(self):
        return [] if self.chain is None else self.chain.estimators_

    def predict_proba(self, matrix) -> np.ndarray:
        result = np.tile(self.constant_values, (matrix.shape[0], 1)).astype(float)
        if self.chain is not None:
            result[:, self.active_indices] = self.chain.predict_proba(matrix)
        return result


def fit_word_char_chain(
    texts: list[str], labels: np.ndarray, *, seed: int = 42, c: float = 1.0,
):
    """Fit a randomized classifier chain over the shared sparse word/char features."""
    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must have the same non-zero length")
    features, _ = build_word_char_pipeline(c=c)
    matrix = features.fit_transform(texts)
    active = np.flatnonzero((labels.sum(axis=0) > 0) & (labels.sum(axis=0) < len(labels)))
    constant_values = (labels.sum(axis=0) == len(labels)).astype(float)
    chain = None
    if len(active):
        estimator = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
        chain = ClassifierChain(estimator=estimator, order="random", random_state=seed)
        chain.fit(matrix, labels[:, active])
    return features, WordCharChain(chain, active, constant_values, labels.shape[1])


def predict_word_char_chain(features, chain, texts: list[str]) -> np.ndarray:
    """Predict conditional per-category probabilities for a sequence of chunks."""
    if not texts:
        return np.empty((0, chain.n_categories), dtype=float)
    return np.asarray(chain.predict_proba(features.transform(texts)))
