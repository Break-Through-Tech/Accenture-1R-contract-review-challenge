from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.chunking.chunk import Chunk
from src.models.naive.retrieval_reranker import (
    PairCrossEncoder,
    PairExample,
    build_training_pairs,
    candidate_recall,
    dense_candidate_scores,
    retrieve_top_k,
)


class FakeEncoder(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embedding = nn.Embedding(32, hidden)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _chunk(contract, index, positive=False):
    text = f"clause {contract}-{index}"
    labels = [0] * 41
    labels[0] = int(positive)
    return SimpleNamespace(
        chunk=Chunk(contract_id=contract, start=0, end=len(text), text=text, chunk_index=index),
        labels=labels,
    )


def _scores(rows, values):
    scores = np.zeros((rows, 41), dtype=float)
    scores[:, :len(values[0])] = values
    return scores


def test_retrieve_top_k_is_contract_scoped_deterministic_and_annotation_blind():
    chunks = [_chunk("a", 1, True), _chunk("a", 0), _chunk("a", 2), _chunk("b", 0, True)]
    probabilities = _scores(4, [[0.7, 0.4], [0.7, 0.9], [0.1, 0.2], [0.3, 0.8]])

    candidates = retrieve_top_k(
        probabilities, chunks, top_k=1,
        active_categories=np.array([True, True] + [False] * 39),
    )

    by_key = {(c.contract_id, c.category_index): c.chunk_index for c in candidates}
    assert by_key["a", 0] == 0  # tie resolved by chunk index
    assert by_key["a", 1] == 0
    assert by_key["b", 0] == 0
    assert by_key["b", 1] == 0
    assert len(candidates) == 4


def test_retrieve_top_k_rejects_invalid_k_and_misaligned_scores():
    chunks = [_chunk("a", 0)]
    with np.testing.assert_raises(ValueError):
        retrieve_top_k(np.zeros((1, 41)), chunks, top_k=0)
    with np.testing.assert_raises(ValueError):
        retrieve_top_k(np.zeros((2, 41)), chunks, top_k=1)


def test_training_pairs_include_every_positive_and_only_retrieved_hard_negatives():
    chunks = [_chunk("a", 0, True), _chunk("a", 1), _chunk("b", 0, True), _chunk("b", 1)]
    labels = np.asarray([item.labels for item in chunks])
    retrieval = _scores(4, [[0.9, 0.1], [0.8, 0.9], [0.7, 0.2], [0.6, 0.8]])
    prompts = ["category one", "category two"] + ["unused"] * 39

    examples, active = build_training_pairs(
        chunks, labels, retrieval, prompts,
        top_k=2, negatives_per_positive=2, min_hard_negatives=1, seed=3,
    )

    positives = [example for example in examples if example.target == 1]
    negatives = [example for example in examples if example.target == 0]
    assert active.tolist() == [True] + [False] * 40
    assert len(positives) == 2  # all gold positives, including missed candidates
    assert negatives and all(example.category_index == 0 for example in negatives)
    assert all(labels[example.chunk_row, example.category_index] == 0 for example in negatives)


def test_candidate_recall_reports_pair_and_contract_category_coverage():
    chunks = [_chunk("a", 0, True), _chunk("a", 1), _chunk("b", 0, True), _chunk("b", 1)]
    labels = np.asarray([item.labels for item in chunks])
    probabilities = _scores(4, [[0.9, 0.0], [0.1, 0.0], [0.1, 0.0], [0.9, 0.0]])
    candidates = retrieve_top_k(
        probabilities, chunks, top_k=1,
        active_categories=np.array([True] + [False] * 40),
    )

    report = candidate_recall(labels, chunks, candidates)

    assert report["positive_chunk_category_pairs"] == 2
    assert report["retrieved_positive_pairs"] == 1
    assert report["pair_recall"] == 0.5
    assert report["positive_contract_categories"] == 2
    assert report["retrieved_positive_contract_categories"] == 1
    assert report["contract_category_recall"] == 0.5


def test_dense_candidate_scores_scatter_probabilities_and_zero_non_candidates():
    chunks = [_chunk("a", 0), _chunk("a", 1)]
    candidates = retrieve_top_k(
        _scores(2, [[0.9, 0.2], [0.1, 0.8]]), chunks, top_k=1,
        active_categories=np.array([True, True] + [False] * 39),
    )

    scores = dense_candidate_scores(chunks, candidates, np.array([0.7, 0.4]), num_labels=41)

    np.testing.assert_allclose(scores[0, :2], [0.7, 0.0])
    np.testing.assert_allclose(scores[1, :2], [0.0, 0.4])


def test_pair_cross_encoder_returns_binary_logits_and_trainable_loss():
    model = PairCrossEncoder(FakeEncoder(), dropout=0)
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = input_ids.ne(0).long()

    logits, loss = model(input_ids, attention_mask, labels=torch.tensor([1.0, 0.0]))
    loss.backward()

    assert logits.shape == (2,)
    assert torch.isfinite(loss)
    assert model.classifier.weight.grad is not None
    assert model.encoder.embedding.weight.grad is not None
