import numpy as np

from src.chunking.chunk import Chunk
from src.labels.projection import LabeledChunk
from src.models.naive.classifier_chain import fit_word_char_chain, predict_word_char_chain
from src.models.naive.context_tfidf import add_neighbor_context
from src.models.naive.prototype_retrieval import fit_positive_prototype_retriever


def _labeled(contract_id, index, text, label):
    chunk = Chunk(contract_id=contract_id, start=0, end=len(text), text=text, chunk_index=index)
    return LabeledChunk(chunk=chunk, labels=[label] + [0] * 40)


def test_neighbor_context_uses_only_adjacent_chunks_from_same_contract():
    chunks = [
        _labeled("a", 1, "middle clause", 0),
        _labeled("b", 0, "other contract", 0),
        _labeled("a", 0, "first clause", 0),
        _labeled("a", 2, "last clause", 0),
    ]

    contexts = add_neighbor_context(chunks)

    assert "first clause" in contexts[0] and "last clause" in contexts[0]
    assert "middle clause" not in contexts[1]
    assert "other contract" not in contexts[0]
    assert "middle clause" in contexts[2] and "last clause" not in contexts[2]


def test_classifier_chain_returns_valid_multilabel_scores():
    texts = [
        "supplier grants license to customer", "license grant applies to affiliates",
        "supplier may terminate with notice", "termination notice period applies",
        "license is granted to affiliate", "either party may terminate agreement",
    ]
    labels = np.zeros((len(texts), 41), dtype=np.uint8)
    labels[[0, 1, 4], 0] = 1
    labels[[2, 3, 5], 1] = 1
    features, chain = fit_word_char_chain(texts, labels, seed=3)

    scores = predict_word_char_chain(features, chain, texts[:2])

    assert scores.shape == (2, 41)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_positive_prototype_retrieval_scores_nearest_training_examples():
    texts = [
        "supplier grants a license to customer", "license grant to affiliate",
        "termination requires written notice", "party may terminate this agreement",
    ]
    labels = np.zeros((4, 41), dtype=np.uint8)
    labels[[0, 1], 0] = 1
    labels[[2, 3], 1] = 1
    model = fit_positive_prototype_retriever(texts, labels)

    scores = model.predict_proba([texts[0], "license is granted to customer"])

    assert scores.shape == (2, 41)
    assert scores[0, 0] > 0.99
    assert scores[0, 0] > scores[0, 1]
    assert np.all(scores[:, 2:] == 0)
