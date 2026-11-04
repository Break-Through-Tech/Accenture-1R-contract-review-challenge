"""Chunk-graph construction and GraphSAGE-style contract classifier."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor, nn
from torch.nn import functional as F

from src.labels.projection import LabeledChunk


@dataclass(frozen=True)
class ContractGraph:
    """Weighted, directed adjacency; undirected edges are stored in both directions."""

    edge_index: Tensor
    edge_weight: Tensor


def build_contract_graphs(
    training_texts: Sequence[str],
    chunks: Sequence[LabeledChunk],
    *,
    semantic_neighbors: int = 4,
) -> tuple[TfidfVectorizer, dict[str, ContractGraph]]:
    """Fit lexical graph features on training text and connect chunks per contract.

    Edges connect immediate neighbors (weight 1) and each node's top non-adjacent
    lexical neighbors (cosine TF-IDF weight). Labels are never used to create edges.
    """
    if not training_texts:
        raise ValueError("training_texts must be non-empty")
    if semantic_neighbors < 0:
        raise ValueError("semantic_neighbors must be non-negative")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=100_000,
        sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
    )
    vectorizer.fit(training_texts)

    grouped: dict[str, list[LabeledChunk]] = defaultdict(list)
    for item in chunks:
        grouped[item.chunk.contract_id].append(item)
    graphs: dict[str, ContractGraph] = {}
    for contract_id, items in grouped.items():
        items.sort(key=lambda item: item.chunk.chunk_index)
        count = len(items)
        edges: dict[tuple[int, int], float] = {}
        for index in range(count - 1):
            edges[(index, index + 1)] = 1.0
            edges[(index + 1, index)] = 1.0
        if semantic_neighbors and count > 2:
            features = vectorizer.transform([item.chunk.text for item in items])
            similarities = (features @ features.T).toarray()
            for source in range(count):
                ranked = [
                    (float(similarities[source, target]), target)
                    for target in range(count)
                    if abs(source - target) > 1 and similarities[source, target] > 0
                ]
                ranked.sort(key=lambda pair: (-pair[0], pair[1]))
                for similarity, target in ranked[:semantic_neighbors]:
                    edges[(source, target)] = max(edges.get((source, target), 0.0), similarity)
                    edges[(target, source)] = max(edges.get((target, source), 0.0), similarity)
        ordered = sorted(edges.items())
        if ordered:
            edge_index = torch.tensor(
                [[edge[0][0] for edge in ordered], [edge[0][1] for edge in ordered]],
                dtype=torch.long,
            )
            edge_weight = torch.tensor([edge[1] for edge in ordered], dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)
        graphs[contract_id] = ContractGraph(edge_index=edge_index, edge_weight=edge_weight)
    return vectorizer, graphs


class WeightedGraphSAGELayer(nn.Module):
    """Mean message passing over weighted directed chunk edges with a residual path."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.self_projection = nn.Linear(hidden_size, hidden_size)
        self.neighbor_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.normalization = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        source, target = edge_index
        aggregate = torch.zeros_like(nodes)
        degree = nodes.new_zeros((nodes.shape[0], 1))
        if source.numel():
            weights = edge_weight.to(nodes.dtype).unsqueeze(-1)
            aggregate.index_add_(0, target, nodes[source] * weights)
            degree.index_add_(0, target, weights)
            aggregate = aggregate / degree.clamp_min(1e-8)
        update = self.self_projection(nodes) + self.neighbor_projection(aggregate)
        update = F.gelu(self.normalization(update))
        return nodes + self.dropout(update)


class ContractGraphClassifier(nn.Module):
    """Encode chunks independently, then pass messages over contract graph edges."""

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int,
        *,
        graph_hidden_size: int = 256,
        graph_layers: int = 2,
        encoder_chunk_batch_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if graph_layers < 1 or encoder_chunk_batch_size < 1:
            raise ValueError("graph_layers and encoder_chunk_batch_size must be positive")
        self.encoder = encoder
        self.encoder_chunk_batch_size = encoder_chunk_batch_size
        self.input_projection = nn.Linear(int(encoder.config.hidden_size), graph_hidden_size)
        self.graph_layers = nn.ModuleList([
            WeightedGraphSAGELayer(graph_hidden_size, dropout) for _ in range(graph_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(graph_hidden_size, num_labels)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        chunk_mask: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        labels: Tensor | None = None,
        pos_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if input_ids.shape[0] != 1:
            raise ValueError("contract graph batches must contain exactly one contract")
        valid = chunk_mask[0].bool()
        chunk_ids = input_ids[0, valid]
        chunk_attention = attention_mask[0, valid]
        representations = []
        for start in range(0, len(chunk_ids), self.encoder_chunk_batch_size):
            end = start + self.encoder_chunk_batch_size
            encoded = self.encoder(
                input_ids=chunk_ids[start:end], attention_mask=chunk_attention[start:end],
            )
            representations.append(encoded.last_hidden_state[:, 0])
        nodes = self.input_projection(torch.cat(representations, dim=0))
        for layer in self.graph_layers:
            nodes = layer(nodes, edge_index, edge_weight)
        logits = self.classifier(self.dropout(nodes))
        loss = None
        if labels is not None:
            targets = labels[0, valid].float()
            loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        return logits, loss
