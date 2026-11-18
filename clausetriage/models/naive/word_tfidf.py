"""Word-and-character TF-IDF one-vs-rest linear clause classifier.

The word channel captures legal phrases; the character channel is designed to
recover lexical variants, punctuation-heavy citations, and morphology. Both
channels are fitted on training chunks only and combined by FeatureUnion.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def build_word_char_pipeline(
    word_max_features: int | None = 100_000,
    char_max_features: int | None = 100_000,
    c: float = 1.0,
) -> tuple[FeatureUnion, OneVsRestClassifier]:
    """Create the two-channel sparse feature model with balanced per-class loss."""
    if c <= 0:
        raise ValueError("c must be positive")
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=word_max_features,
            sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=char_max_features,
            min_df=2, sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
        )),
    ])
    classifier = OneVsRestClassifier(
        LogisticRegression(C=c, max_iter=2000, class_weight="balanced"),
        n_jobs=1,
    )
    return features, classifier


def fit_word_char_tfidf(
    texts: list[str], labels: np.ndarray, *,
    word_max_features: int | None = 100_000,
    char_max_features: int | None = 100_000,
    c: float = 1.0,
) -> tuple[FeatureUnion, OneVsRestClassifier]:
    """Fit on train data only; callers tune thresholds on validation data."""
    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must have the same non-zero length")
    features, classifier = build_word_char_pipeline(
        word_max_features=word_max_features,
        char_max_features=char_max_features,
        c=c,
    )
    classifier.fit(features.fit_transform(texts), labels)
    return features, classifier


def predict_word_char_probabilities(
    features: FeatureUnion,
    classifier: OneVsRestClassifier,
    texts: list[str],
) -> np.ndarray:
    """Return raw per-category probabilities without applying thresholds."""
    if not texts:
        return np.empty((0, len(classifier.estimators_)), dtype=float)
    return np.asarray(classifier.predict_proba(features.transform(texts)))
