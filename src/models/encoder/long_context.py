"""Hierarchical contract model: chunk encoder plus bidirectional recurrent context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class LongContextOutput:
    """Model outputs for chunk logits and optional training loss."""

    logits: Tensor
    loss: Tensor | None = None


class LongContextClassifier(nn.Module):
    """Encode each chunk, contextualize the ordered contract sequence, classify per chunk.

    The base transformer encodes each sentence-aware chunk independently. A
    bidirectional GRU then passes information across all chunks in the contract,
    allowing every chunk's clause predictions to use long-range contract context.
    """

    def __init__(self, encoder: nn.Module, hidden_size: int, num_labels: int,
                 dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.context = nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size // 2,
            batch_first=True, bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids: Tensor, attention_mask: Tensor,
                chunk_mask: Tensor, labels: Tensor | None = None,
                pos_weight: Tensor | None = None) -> LongContextOutput:
        """Return per-chunk logits, masking padded chunks from recurrent and loss steps."""
        batch_size, chunk_count, token_count = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * chunk_count, token_count)
        flat_mask = attention_mask.reshape(batch_size * chunk_count, token_count)
        valid = chunk_mask.reshape(-1).bool()
        valid_ids = flat_ids[valid]
        valid_mask = flat_mask[valid]

        encoder_output = self.encoder(input_ids=valid_ids, attention_mask=valid_mask)
        hidden = encoder_output.last_hidden_state[:, 0]
        chunk_vectors = hidden.new_zeros((batch_size * chunk_count, hidden.shape[-1]))
        chunk_vectors[valid] = hidden
        chunk_vectors = chunk_vectors.reshape(batch_size, chunk_count, -1)
        lengths = chunk_mask.sum(dim=1).to(device="cpu")
        packed = pack_padded_sequence(
            chunk_vectors, lengths, batch_first=True, enforce_sorted=False,
        )
        contextual_packed, _ = self.context(packed)
        contextual, _ = pad_packed_sequence(
            contextual_packed, batch_first=True, total_length=chunk_count,
        )
        logits = self.classifier(self.dropout(contextual))

        loss = None
        if labels is not None:
            element_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float(), pos_weight=pos_weight, reduction="none",
            )
            mask = chunk_mask.unsqueeze(-1).to(element_loss.dtype)
            loss = (element_loss * mask).sum() / (mask.sum() * self.num_labels).clamp_min(1)
        return LongContextOutput(logits=logits, loss=loss)


def build_long_context_model(model_name: str, revision: str | None = None,
                             num_labels: int = 41, dropout: float = 0.2) -> LongContextClassifier:
    """Load a pretrained transformer encoder and attach the BiGRU chunk-context head."""
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(model_name, revision=revision)
    hidden_size = int(encoder.config.hidden_size)
    return LongContextClassifier(encoder, hidden_size, num_labels, dropout)
