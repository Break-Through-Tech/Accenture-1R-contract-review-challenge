from sklearn.feature_extraction import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

"""
TF-IDF + one-vs-rest logistic regression baseline for multi-label clause classification
Used one-vs-rest logistic regression to split 41 category problem into 41 binary classification tasks

Notable Features:
max_features caps vocab to the x most informative words
ngram_range allows for both single words and two-word phrases (e.g., termination clause or governing law)
stop_words strips filler words (e.g., the, and, of, or)
class_weight automatically up-weights errors on rare categories since some categories only appear in some contracts
"""

# Initializes vectorizer and classifier
def build_tfidf_pipeline(max_features = 10000, ngram_range = (1,2)):
    vectorizer = TfidfVectorizer(max_features = max_features, ngram_range = ngram_range, stop_words = "english")
    classifier = OneVsRestClassifier(LogisticRegression(max_iter = 1000, class_weight = "balanced"))
    return vectorizer, classifier

# Learns vocab and IDF weights from training chunks and converts to vectors
def fit_tfidf_baseline(texts: list[str], labels: np.ndarray, max_features: int = 10000, ngram_range: tuple = (1, 2)):
    vectorizer, classifier = build_tfidf_pipeline(max_features, ngram_range)
    X = vectorizer.fit_transform(texts)
    classifier.fit(X, labels)
    return vectorizer, classifier

# Reuses fitted vocab, then votes yes/no on each of the 41 binary classifiers
def pred_tfidf_baseline(vectorizer, classifier, texts: list[str]) -> np.ndarray:
    X = vectorizer.transform(texts)
    return classifier.predict(X)
