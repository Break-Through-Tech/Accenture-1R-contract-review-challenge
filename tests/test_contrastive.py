from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix
from torch import nn

from src.models.encoder.contrastive import (
    ContrastiveChunkEncoder,
    build_category_prototypes,
    sample_category_triplet_batch,
    score_category_prototypes,
    supervised_contrastive_loss,
)


class FakeEncoder(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embedding = nn.Embedding(32, hidden)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_contrastive_encoder_outputs_normalized_embeddings_and_gradients():
    model = ContrastiveChunkEncoder(FakeEncoder(), embedding_size=5, dropout=0)
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])

    embeddings = model(input_ids, input_ids.ne(0).long())
    embeddings.square().sum().backward()

    torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(2))
    assert model.encoder.embedding.weight.grad is not None
    assert model.projection[-1].weight.grad is not None


def test_multilabel_contrastive_loss_uses_positive_pairs_and_lexical_hardness():
    embeddings = nn.functional.normalize(torch.randn(6, 8), dim=-1).requires_grad_()
    labels = torch.tensor([
        [1, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0],
    ], dtype=torch.float32)
    lexical = torch.eye(6) * 0 + 0.2

    loss = supervised_contrastive_loss(
        embeddings, labels, lexical, temperature=0.2, hard_negative_weight=2.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert embeddings.grad is not None
    with pytest.raises(ValueError, match="no positive label pairs"):
        supervised_contrastive_loss(embeddings.detach()[:2], torch.eye(2))


def test_category_balanced_sampler_returns_positive_and_category_hard_negative_rows():
    labels = np.array([
        [1, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0],
    ])
    tfidf = csr_matrix(np.array([
        [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8],
    ], dtype=np.float32))

    indices, categories = sample_category_triplet_batch(
        labels, tfidf, np.array([True, True]), np.random.default_rng(12),
        categories_per_batch=2, hard_negative_pool=2,
    )

    assert len(indices) == 3 * len(categories) == 6
    for offset, category in enumerate(categories):
        anchor, positive, hard_negative = indices[offset * 3:offset * 3 + 3]
        assert labels[anchor, category] == labels[positive, category] == 1
        assert labels[hard_negative, category] == 0
        assert tfidf[anchor].multiply(tfidf[hard_negative]).sum() > 0


def test_prototypes_are_train_derived_and_score_positive_examples_higher():
    embeddings = nn.functional.normalize(torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1],
    ]), dim=-1)
    labels = torch.tensor([[1], [1], [0], [0]])

    positive, negative = build_category_prototypes(embeddings, labels)
    scores = score_category_prototypes(embeddings, positive, negative, temperature=0.2)

    assert positive.shape == negative.shape == (1, 2)
    assert scores[0, 0] > scores[2, 0]
    assert scores[1, 0] > 0.5
    assert scores[2, 0] < 0.5
