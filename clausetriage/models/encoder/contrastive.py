"""Multi-label supervised contrastive chunk embeddings with train-only prototypes."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ContrastiveChunkEncoder(nn.Module):
    """Project pretrained chunk representations into a normalized embedding space."""

    def __init__(self, encoder: nn.Module, embedding_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden_size = int(encoder.config.hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = encoded.last_hidden_state[:, 0]
        return F.normalize(self.projection(pooled), p=2, dim=-1)


def supervised_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
    lexical_similarity: Tensor | None = None,
    *,
    temperature: float = 0.07,
    hard_negative_weight: float = 2.0,
) -> Tensor:
    """Multi-positive supervised contrastive objective with lexical hard negatives."""
    if embeddings.ndim != 2 or labels.ndim != 2 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must be 2D and share the batch dimension")
    if temperature <= 0 or hard_negative_weight < 0:
        raise ValueError("temperature must be positive and hard-negative weight non-negative")
    if lexical_similarity is not None and lexical_similarity.shape != (len(embeddings), len(embeddings)):
        raise ValueError("lexical_similarity must be square and aligned to the batch")

    pair_labels = labels.float() @ labels.float().T
    positive = pair_labels > 0
    diagonal = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positive = positive & ~diagonal
    similarities = embeddings @ embeddings.T / temperature
    if lexical_similarity is not None and hard_negative_weight:
        negative = ~(positive | diagonal)
        lexical_bias = torch.log1p(hard_negative_weight * lexical_similarity.float().clamp_min(0))
        similarities = similarities + lexical_bias * negative
    similarities = similarities.masked_fill(diagonal, -torch.inf)
    has_positive = positive.any(dim=1)
    if not has_positive.any():
        raise ValueError("contrastive batch contains no positive label pairs")
    log_probabilities = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    positive_log_probabilities = torch.where(positive, log_probabilities, torch.zeros_like(log_probabilities))
    per_anchor = -positive_log_probabilities.sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    return per_anchor[has_positive].mean()


def sample_category_triplet_batch(
    labels: np.ndarray,
    tfidf_features,
    active_categories: np.ndarray,
    rng: np.random.Generator,
    *,
    categories_per_batch: int = 8,
    hard_negative_pool: int = 4,
) -> tuple[list[int], list[int]]:
    """Sample anchor/positive/lexical-hard-negative triples within training data."""
    if labels.ndim != 2 or labels.shape[0] != tfidf_features.shape[0]:
        raise ValueError("labels and TF-IDF rows must align")
    if active_categories.shape != (labels.shape[1],):
        raise ValueError("active_categories must match the label dimension")
    if categories_per_batch < 1 or hard_negative_pool < 1:
        raise ValueError("batch category count and hard-negative pool must be positive")
    categories = np.flatnonzero(active_categories)
    if not categories.size:
        raise ValueError("no active categories to sample")
    selected = rng.choice(categories, size=min(categories_per_batch, len(categories)), replace=False)
    indices: list[int] = []
    selected_categories: list[int] = []
    for category in selected:
        positives = np.flatnonzero(labels[:, category] == 1)
        negatives = np.flatnonzero(labels[:, category] == 0)
        if not positives.size or not negatives.size:
            continue
        chosen_positive = rng.choice(positives, size=2, replace=len(positives) < 2)
        anchor = int(chosen_positive[0])
        similarities = (tfidf_features[anchor] @ tfidf_features.T).toarray().ravel()
        ranked_negatives = negatives[np.argsort(-similarities[negatives], kind="stable")]
        hard_pool = ranked_negatives[:min(hard_negative_pool, len(ranked_negatives))]
        hard_pool = hard_pool[similarities[hard_pool] > 0]
        hard_negative = int(rng.choice(hard_pool if len(hard_pool) else negatives))
        indices.extend((anchor, int(chosen_positive[1]), hard_negative))
        selected_categories.append(int(category))
    if not indices:
        raise ValueError("could not sample any positive/negative category triples")
    return indices, selected_categories


def build_category_prototypes(embeddings: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Build normalized positive/negative centroids from training embeddings only."""
    if embeddings.ndim != 2 or labels.ndim != 2 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must be 2D and share the row dimension")
    positives = labels.float().T @ embeddings
    positive_counts = labels.float().sum(dim=0).unsqueeze(-1).clamp_min(1)
    negatives = (1 - labels.float()).T @ embeddings
    negative_counts = (1 - labels.float()).sum(dim=0).unsqueeze(-1).clamp_min(1)
    positive_prototypes = F.normalize(positives / positive_counts, p=2, dim=-1)
    negative_prototypes = F.normalize(negatives / negative_counts, p=2, dim=-1)
    active = labels.sum(dim=0) > 0
    positive_prototypes = torch.where(active.unsqueeze(-1), positive_prototypes,
                                      torch.zeros_like(positive_prototypes))
    return positive_prototypes, negative_prototypes


def score_category_prototypes(
    embeddings: Tensor,
    positive_prototypes: Tensor,
    negative_prototypes: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    """Convert positive-vs-negative prototype cosine margins to category scores."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    margin = embeddings @ positive_prototypes.T - embeddings @ negative_prototypes.T
    return torch.sigmoid(margin / temperature)
