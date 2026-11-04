"""Hungarian-matched, category-conditioned set prediction over contract chunk spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F

from src.labels.categories import CATEGORY_TO_INDEX, CATEGORIES
from src.labels.loader import AnswerSpan, Contract
from src.labels.projection import LabeledChunk


@dataclass(frozen=True)
class GoldChunkSpan:
    category_index: int
    start_chunk: int
    end_chunk: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class SetPredictionOutput:
    presence_logits: Tensor
    start_logits: Tensor
    end_logits: Tensor


@dataclass(frozen=True)
class PredictedSpan:
    category_index: int
    score: float
    start_chunk: int
    end_chunk: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ContractSetScores:
    contract_presence: np.ndarray
    chunk_scores: np.ndarray
    spans: tuple[PredictedSpan, ...]


def map_gold_spans_to_chunks(
    contract: Contract,
    chunks: Sequence[LabeledChunk],
) -> tuple[list[GoldChunkSpan], int]:
    """Map answer character spans to their first/last overlapping chunk positions.

    Returns mapped instances and the count of answer spans not covered by any chunk.
    Source offsets remain attached for later character-level IoU evaluation.
    """
    ordered = sorted(chunks, key=lambda item: item.chunk.chunk_index)
    if any(item.chunk.contract_id != contract.contract_id for item in ordered):
        raise ValueError("all chunks must belong to the supplied contract")
    mapped: list[GoldChunkSpan] = []
    uncovered = 0
    for span in contract.spans:
        if span.category not in CATEGORY_TO_INDEX:
            raise ValueError(f"unknown span category {span.category!r}")
        overlapping = [
            index for index, item in enumerate(ordered)
            if max(item.chunk.start, span.start) < min(item.chunk.end, span.end)
        ]
        if not overlapping:
            uncovered += 1
            continue
        mapped.append(GoldChunkSpan(
            category_index=CATEGORY_TO_INDEX[span.category],
            start_chunk=min(overlapping), end_chunk=max(overlapping),
            char_start=span.start, char_end=span.end,
        ))
    return mapped, uncovered


class ContractSetPredictor(nn.Module):
    """Category-specific learned slots attend to chunks and point to span boundaries."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_labels: int = len(CATEGORIES),
        max_slots: int = 8,
        max_chunks: int = 4096,
        attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if max_slots < 1 or max_chunks < 1 or hidden_size % attention_heads:
            raise ValueError("invalid slot/chunk limits or attention head divisibility")
        self.num_labels = num_labels
        self.max_slots = max_slots
        self.max_chunks = max_chunks
        self.hidden_size = hidden_size
        self.context_projection = nn.Linear(hidden_size, hidden_size)
        self.position_embedding = nn.Embedding(max_chunks, hidden_size)
        self.slot_queries = nn.Parameter(torch.empty(num_labels, max_slots, hidden_size))
        nn.init.normal_(self.slot_queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, attention_heads, dropout=dropout, batch_first=True,
        )
        self.query_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.presence_head = nn.Linear(hidden_size, 1)
        self.start_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.end_projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, chunk_embeddings: Tensor) -> SetPredictionOutput:
        if chunk_embeddings.ndim != 2 or chunk_embeddings.shape[1] != self.hidden_size:
            raise ValueError("chunk_embeddings must have shape [chunks, hidden_size]")
        count = chunk_embeddings.shape[0]
        if count < 1 or count > self.max_chunks:
            raise ValueError(f"contract must contain 1..{self.max_chunks} chunks")
        positions = torch.arange(count, device=chunk_embeddings.device)
        context = self.context_projection(chunk_embeddings) + self.position_embedding(positions)
        queries = self.slot_queries.reshape(1, self.num_labels * self.max_slots, self.hidden_size)
        attended, _ = self.cross_attention(
            queries, context.unsqueeze(0), context.unsqueeze(0), need_weights=False,
        )
        hidden = self.query_norm(queries + self.dropout(attended)).squeeze(0)
        presence = self.presence_head(self.dropout(hidden)).squeeze(-1)
        start = self.start_projection(hidden) @ context.T / self.hidden_size ** 0.5
        end = self.end_projection(hidden) @ context.T / self.hidden_size ** 0.5
        shape = (self.num_labels, self.max_slots)
        return SetPredictionOutput(
            presence_logits=presence.reshape(shape),
            start_logits=start.reshape(*shape, count),
            end_logits=end.reshape(*shape, count),
        )


def matched_set_loss(
    output: SetPredictionOutput,
    gold_spans: Sequence[GoldChunkSpan],
    *,
    max_slots: int,
    no_object_weight: float = 0.2,
    pointer_weight: float = 1.0,
) -> tuple[Tensor, dict[str, int]]:
    """Hungarian-match each category's slots to gold spans and train no-object slots."""
    if not 0 <= no_object_weight <= 1 or pointer_weight < 0:
        raise ValueError("invalid objectness or pointer loss weight")
    if output.presence_logits.shape[1] != max_slots:
        raise ValueError("max_slots does not match output shape")
    presence_losses = []
    pointer_losses = []
    matched_count = 0
    overflow_count = 0
    for category_index in range(output.presence_logits.shape[0]):
        gold = sorted(
            (span for span in gold_spans if span.category_index == category_index),
            key=lambda span: (span.char_start, span.char_end),
        )
        if len(gold) > max_slots:
            overflow_count += len(gold) - max_slots
            gold = gold[:max_slots]
        logits = output.presence_logits[category_index]
        targets = torch.zeros_like(logits)
        if gold:
            log_start = F.log_softmax(output.start_logits[category_index], dim=-1)
            log_end = F.log_softmax(output.end_logits[category_index], dim=-1)
            cost = -F.logsigmoid(logits).detach().unsqueeze(1)
            cost = cost - log_start[:, [span.start_chunk for span in gold]].detach()
            cost = cost - log_end[:, [span.end_chunk for span in gold]].detach()
            rows, columns = linear_sum_assignment(cost.cpu().numpy())
            targets[torch.as_tensor(rows, device=targets.device)] = 1.0
            for slot, gold_index in zip(rows.tolist(), columns.tolist()):
                span = gold[gold_index]
                pointer_losses.append(
                    F.cross_entropy(output.start_logits[category_index, slot].unsqueeze(0),
                                    torch.tensor([span.start_chunk], device=logits.device))
                    + F.cross_entropy(output.end_logits[category_index, slot].unsqueeze(0),
                                      torch.tensor([span.end_chunk], device=logits.device))
                )
            matched_count += len(rows)
        weights = torch.where(targets > 0, torch.ones_like(targets),
                              torch.full_like(targets, no_object_weight))
        presence_losses.append(
            (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights).mean()
        )
    presence_loss = torch.stack(presence_losses).mean()
    pointer_loss = torch.stack(pointer_losses).mean() if pointer_losses else presence_loss.new_zeros(())
    return presence_loss + pointer_weight * pointer_loss, {
        "matched_spans": matched_count, "overflow_spans": overflow_count,
    }


def decode_contract_set(
    output: SetPredictionOutput,
    chunks: Sequence[LabeledChunk],
) -> ContractSetScores:
    """Decode slot intervals to chunk-boundary character ranges and dense chunk scores."""
    ordered = sorted(chunks, key=lambda item: item.chunk.chunk_index)
    if output.presence_logits.shape[0] != len(CATEGORIES):
        raise ValueError("output label count does not match CUAD categories")
    presence = torch.sigmoid(output.presence_logits).detach().cpu().numpy()
    starts = output.start_logits.argmax(dim=-1).detach().cpu().numpy()
    ends = output.end_logits.argmax(dim=-1).detach().cpu().numpy()
    contract_scores = presence.max(axis=1)
    chunk_scores = np.zeros((len(ordered), len(CATEGORIES)), dtype=np.float32)
    predictions: list[PredictedSpan] = []
    for category_index in range(len(CATEGORIES)):
        for slot, score in enumerate(presence[category_index]):
            start_chunk = min(int(starts[category_index, slot]), int(ends[category_index, slot]))
            end_chunk = max(int(starts[category_index, slot]), int(ends[category_index, slot]))
            end_chunk = min(end_chunk, len(ordered) - 1)
            for index in range(start_chunk, end_chunk + 1):
                chunk_scores[index, category_index] = max(chunk_scores[index, category_index], score)
            predictions.append(PredictedSpan(
                category_index=category_index, score=float(score),
                start_chunk=start_chunk, end_chunk=end_chunk,
                char_start=ordered[start_chunk].chunk.start,
                char_end=ordered[end_chunk].chunk.end,
            ))
    return ContractSetScores(contract_scores, chunk_scores, tuple(predictions))


def span_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    """End-exclusive character-span intersection over union."""
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return float(intersection / union) if union > 0 else 0.0
