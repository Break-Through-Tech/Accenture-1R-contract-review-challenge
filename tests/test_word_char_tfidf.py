import numpy as np
import pytest

from src.models.naive.word_tfidf import (
    fit_word_char_tfidf,
    predict_word_char_probabilities,
)


def test_word_char_model_returns_multilabel_probabilities():
    texts = [
        "The supplier shall maintain insurance coverage under this agreement.",
        "Supplier shall maintain insurance coverage throughout the term.",
        "Either party may terminate this agreement upon written notice.",
        "The customer may terminate the contract by providing notice.",
        "Insurance coverage shall be maintained by the vendor.",
        "Termination requires thirty days prior notice.",
    ]
    labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]])
    features, classifier = fit_word_char_tfidf(texts, labels, char_max_features=None)
    probabilities = predict_word_char_probabilities(features, classifier, texts)
    assert probabilities.shape == labels.shape
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_rejects_mismatched_training_rows():
    with pytest.raises(ValueError, match="same non-zero length"):
        fit_word_char_tfidf(["one"], np.empty((0, 2), dtype=int))
