"""Longformer-style local/global sparse attention over complete chunk sequences."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LocalGlobalSparseLayer(nn.Module):
    """Multi-head attention with local chunk windows and one contract-global token."""

    def __init__(self, hidden_size: int, heads: int, radius: int, dropout: float = 0.1):
        super().__init__()
        if hidden_size % heads or radius < 0:
            raise ValueError("hidden size must divide heads and local radius must be nonnegative")
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_size = hidden_size // heads
        self.radius = radius
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.global_output = nn.Linear(hidden_size, hidden_size)
        self.node_norm1 = nn.LayerNorm(hidden_size)
        self.node_norm2 = nn.LayerNorm(hidden_size)
        self.global_norm1 = nn.LayerNorm(hidden_size)
        self.global_norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: Tensor, global_state: Tensor) -> tuple[Tensor, Tensor]:
        count = nodes.shape[0]
        q, k, v = self.qkv(nodes).chunk(3, dim=-1)
        q = q.reshape(count, self.heads, self.head_size)
        k = k.reshape(count, self.heads, self.head_size)
        v = v.reshape(count, self.heads, self.head_size)
        gq, gk, gv = self.qkv(global_state.unsqueeze(0)).chunk(3, dim=-1)
        gq = gq.reshape(self.heads, self.head_size)
        gk = gk.reshape(self.heads, self.head_size)
        gv = gv.reshape(self.heads, self.head_size)

        offsets = torch.arange(-self.radius, self.radius + 1, device=nodes.device)
        positions = torch.arange(count, device=nodes.device)
        neighbor_indices = positions[:, None] + offsets[None, :]
        valid = (neighbor_indices >= 0) & (neighbor_indices < count)
        neighbors = neighbor_indices.clamp(0, count - 1)
        neighbor_keys = k[neighbors].permute(0, 2, 1, 3)
        neighbor_values = v[neighbors].permute(0, 2, 1, 3)
        scale = self.head_size ** -0.5
        local_logits = (q.unsqueeze(2) * neighbor_keys).sum(dim=-1) * scale
        local_logits = local_logits.masked_fill(~valid[:, None, :], -torch.inf)
        global_logits = (q * gk.unsqueeze(0)).sum(dim=-1).unsqueeze(-1) * scale
        weights = torch.softmax(torch.cat((local_logits, global_logits), dim=-1), dim=-1)
        local_weights, global_weights = weights[..., :-1], weights[..., -1:]
        node_message = (local_weights.unsqueeze(-1) * neighbor_values).sum(dim=2)
        node_message = node_message + global_weights * gv.view(1, self.heads, self.head_size)
        node_message = self.output(node_message.reshape(count, self.hidden_size))
        nodes = self.node_norm1(nodes + self.dropout(node_message))
        nodes = self.node_norm2(nodes + self.dropout(self.feed_forward(nodes)))

        global_logits = torch.einsum("hd,nhd->hn", gq, k) * scale
        self_logit = (gq * gk).sum(dim=-1, keepdim=True) * scale
        global_weights = torch.softmax(torch.cat((global_logits, self_logit), dim=-1), dim=-1)
        node_weights, self_weight = global_weights[:, :-1], global_weights[:, -1:]
        global_message = torch.einsum("hn,nhd->hd", node_weights, v)
        global_message = global_message + self_weight * gv
        global_message = self.global_output(global_message.reshape(1, self.hidden_size)).squeeze(0)
        global_state = self.global_norm1(global_state + self.dropout(global_message))
        global_state = self.global_norm2(
            global_state + self.dropout(self.feed_forward(global_state.unsqueeze(0)).squeeze(0)),
        )
        return nodes, global_state


class SparseLongContextClassifier(nn.Module):
    """Classify all chunks after O(N * local_window) contract context propagation."""

    def __init__(
        self,
        input_size: int,
        num_labels: int,
        *,
        hidden_size: int = 256,
        layers: int = 2,
        heads: int = 8,
        local_radius: int = 16,
        max_chunks: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        if layers < 1 or max_chunks < 1:
            raise ValueError("layers and max_chunks must be positive")
        self.hidden_size = hidden_size
        self.max_chunks = max_chunks
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position_embedding = nn.Embedding(max_chunks, hidden_size)
        self.global_token = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.global_token, std=0.02)
        self.layers = nn.ModuleList([
            LocalGlobalSparseLayer(hidden_size, heads, local_radius, dropout)
            for _ in range(layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.local_radius = local_radius

    def forward(self, chunk_embeddings: Tensor) -> Tensor:
        if chunk_embeddings.ndim != 2 or chunk_embeddings.shape[1] != self.input_projection.in_features:
            raise ValueError("chunk_embeddings must have shape [chunks, input_size]")
        count = chunk_embeddings.shape[0]
        if count < 1 or count > self.max_chunks:
            raise ValueError(f"contract must contain 1..{self.max_chunks} chunks")
        positions = torch.arange(count, device=chunk_embeddings.device)
        nodes = self.input_projection(chunk_embeddings) + self.position_embedding(positions)
        global_state = self.global_token
        for layer in self.layers:
            nodes, global_state = layer(nodes, global_state)
        return self.classifier(self.dropout(nodes))


def sparse_attention_edge_count(num_chunks: int, local_radius: int) -> int:
    """Count local, node-to-global, and global-to-node attention links per layer."""
    if num_chunks < 1 or local_radius < 0:
        raise ValueError("num_chunks must be positive and local_radius non-negative")
    local = sum(min(num_chunks - 1, index + local_radius)
                - max(0, index - local_radius) + 1 for index in range(num_chunks))
    return local + 2 * num_chunks + 1


def overlap_consensus_contract_scores(
    chunks: Sequence,
    chunk_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    """Aggregate chunk scores over source-character segments, averaging overlap windows.

    A contract/category score is the maximum active-window mean over source intervals.
    Thus a short high score does not get duplicated or amplified solely because several
    overlapping token windows cover the same characters.
    """
    if chunk_scores.ndim != 2 or chunk_scores.shape[0] != len(chunks):
        raise ValueError("chunk_scores must have one row per supplied chunk")
    grouped: dict[str, list[tuple[int, object]]] = defaultdict(list)
    for row, item in enumerate(chunks):
        grouped[item.chunk.contract_id].append((row, item))
    contract_scores = {}
    for contract_id, rows in grouped.items():
        events: dict[int, tuple[np.ndarray, int]] = {}
        for row, item in rows:
            start, end = item.chunk.start, item.chunk.end
            start_sum, start_count = events.get(
                start, (np.zeros(chunk_scores.shape[1], dtype=np.float64), 0),
            )
            events[start] = (start_sum + chunk_scores[row], start_count + 1)
            end_sum, end_count = events.get(
                end, (np.zeros(chunk_scores.shape[1], dtype=np.float64), 0),
            )
            events[end] = (end_sum - chunk_scores[row], end_count - 1)
        total = np.zeros(chunk_scores.shape[1], dtype=np.float64)
        active = 0
        best = np.zeros(chunk_scores.shape[1], dtype=np.float32)
        boundaries = sorted(events)
        for position, next_position in zip(boundaries, boundaries[1:]):
            delta, count_delta = events[position]
            total += delta
            active += count_delta
            if next_position > position and active > 0:
                best = np.maximum(best, (total / active).astype(np.float32))
        contract_scores[contract_id] = best
    return contract_scores
