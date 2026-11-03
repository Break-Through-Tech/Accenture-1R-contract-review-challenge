"""Category-query contract model with cross-attention evidence localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class CategoryQueryOutput:
    """Per-chunk logits, contract-level logits, and category-to-chunk attention."""

    logits: Tensor
    contract_logits: Tensor
    attention_weights: Tensor
    loss: Tensor | None = None


class CategoryQueryClassifier(nn.Module):
    """Encode contracts, then let each category query retrieve its evidence chunks.

    Label queries are initialized from the pretrained encoder's representations of
    CUAD category descriptions. They cross-attend to a contextualized sequence of
    chunk embeddings. Query/chunk compatibility yields chunk-level predictions;
    a multi-instance objective additionally supervises contract-level presence.
    """

    def __init__(
        self,
        encoder: nn.Module,
        label_query_init: Tensor,
        *,
        encoder_chunk_batch_size: int = 4,
        dropout: float = 0.2,
        context_layers: int = 2,
        attention_heads: int = 8,
        contract_loss_weight: float = 0.25,
        evidence_loss_weight: float = 0.1,
    ):
        super().__init__()
        hidden_size = int(encoder.config.hidden_size)
        if label_query_init.ndim != 2 or label_query_init.shape[1] != hidden_size:
            raise ValueError("label query initialization must have shape [labels, hidden_size]")
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if encoder_chunk_batch_size < 1:
            raise ValueError("encoder_chunk_batch_size must be positive")
        self.encoder = encoder
        self.encoder_chunk_batch_size = encoder_chunk_batch_size
        self.num_labels = int(label_query_init.shape[0])
        self.hidden_size = hidden_size
        self.label_queries = nn.Parameter(label_query_init.detach().clone())
        self.position_embedding = nn.Embedding(512, hidden_size)
        context_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=attention_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            context_layer, num_layers=context_layers, enable_nested_tensor=False,
        )
        self.query_attention = nn.MultiheadAttention(
            hidden_size, attention_heads, dropout=dropout, batch_first=True,
        )
        self.query_norm = nn.LayerNorm(hidden_size)
        self.chunk_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.chunk_classifier = nn.Linear(hidden_size, self.num_labels)
        self.query_scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.label_bias = nn.Parameter(torch.zeros(self.num_labels))
        self.contract_loss_weight = contract_loss_weight
        self.evidence_loss_weight = evidence_loss_weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        chunk_mask: Tensor,
        labels: Tensor | None = None,
        chunk_pos_weight: Tensor | None = None,
        contract_pos_weight: Tensor | None = None,
    ) -> CategoryQueryOutput:
        batch_size, chunk_count, token_count = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * chunk_count, token_count)
        flat_mask = attention_mask.reshape(batch_size * chunk_count, token_count)
        valid = chunk_mask.reshape(-1).bool()
        valid_ids, valid_mask = flat_ids[valid], flat_mask[valid]
        encoded_batches = []
        for start in range(0, valid_ids.shape[0], self.encoder_chunk_batch_size):
            end = start + self.encoder_chunk_batch_size
            output = self.encoder(input_ids=valid_ids[start:end], attention_mask=valid_mask[start:end])
            encoded_batches.append(output.last_hidden_state[:, 0])
        hidden = torch.cat(encoded_batches, dim=0)
        chunk_vectors = hidden.new_zeros((batch_size * chunk_count, self.hidden_size))
        chunk_vectors[valid] = hidden
        chunk_vectors = chunk_vectors.reshape(batch_size, chunk_count, self.hidden_size)

        positions = torch.arange(chunk_count, device=chunk_vectors.device).unsqueeze(0)
        contextual = self.context(
            chunk_vectors + self.position_embedding(positions),
            src_key_padding_mask=~chunk_mask.bool(),
        )
        queries = self.label_queries.unsqueeze(0).expand(batch_size, -1, -1)
        retrieved, weights = self.query_attention(
            queries, contextual, contextual,
            key_padding_mask=~chunk_mask.bool(), need_weights=True,
            average_attn_weights=True,
        )
        retrieved = self.query_norm(queries + self.dropout(retrieved))
        chunk_features = self.chunk_projection(self.dropout(contextual))
        query_features = self.query_projection(self.dropout(retrieved))
        compatibility = torch.einsum("bch,blh->bcl", chunk_features, query_features)
        compatibility = compatibility / (self.hidden_size ** 0.5)
        logits = (self.chunk_classifier(self.dropout(contextual))
                  + self.query_scale * compatibility + self.label_bias)

        contract_logits = self._contract_logits(logits, chunk_mask)
        loss = None
        if labels is not None:
            loss = self._loss(
                logits, contract_logits, weights, labels, chunk_mask,
                chunk_pos_weight, contract_pos_weight,
            )
        return CategoryQueryOutput(logits, contract_logits, weights, loss)

    @staticmethod
    def _contract_logits(logits: Tensor, chunk_mask: Tensor) -> Tensor:
        """Log-mean-exp multiple-instance aggregation over valid contract chunks."""
        masked = logits.masked_fill(~chunk_mask.unsqueeze(-1), torch.finfo(logits.dtype).min)
        counts = chunk_mask.sum(dim=1, keepdim=True).clamp_min(1).to(logits.dtype)
        return torch.logsumexp(masked, dim=1) - counts.log()

    def _loss(
        self,
        logits: Tensor,
        contract_logits: Tensor,
        attention: Tensor,
        labels: Tensor,
        chunk_mask: Tensor,
        chunk_pos_weight: Tensor | None,
        contract_pos_weight: Tensor | None,
    ) -> Tensor:
        label_values = labels.float()
        valid = chunk_mask.unsqueeze(-1).to(logits.dtype)
        chunk_loss = F.binary_cross_entropy_with_logits(
            logits, label_values, pos_weight=chunk_pos_weight, reduction="none",
        )
        chunk_loss = (chunk_loss * valid).sum() / (valid.sum() * self.num_labels).clamp_min(1)

        contract_labels = label_values.amax(dim=1)
        contract_loss = F.binary_cross_entropy_with_logits(
            contract_logits, contract_labels,
            pos_weight=contract_pos_weight, reduction="mean",
        )

        positive_chunks = label_values.transpose(1, 2) * chunk_mask.unsqueeze(1)
        positive_counts = positive_chunks.sum(dim=-1)
        has_evidence = positive_counts > 0
        evidence_targets = positive_chunks / positive_counts.unsqueeze(-1).clamp_min(1)
        evidence_rows = -(evidence_targets * attention.clamp_min(1e-8).log()).sum(dim=-1)
        evidence_loss = (evidence_rows * has_evidence).sum() / has_evidence.sum().clamp_min(1)

        return (chunk_loss + self.contract_loss_weight * contract_loss
                + self.evidence_loss_weight * evidence_loss)

    @property
    def encoder_chunk_batch_size(self) -> int:
        return self._encoder_chunk_batch_size

    @encoder_chunk_batch_size.setter
    def encoder_chunk_batch_size(self, value: int) -> None:
        if value < 1:
            raise ValueError("encoder_chunk_batch_size must be positive")
        self._encoder_chunk_batch_size = value


def build_category_query_model(
    model_name: str,
    revision: str,
    tokenizer,
    category_descriptions: Sequence[str],
    *,
    encoder_chunk_batch_size: int = 4,
    dropout: float = 0.2,
    context_layers: int = 2,
    attention_heads: int = 8,
    contract_loss_weight: float = 0.25,
    evidence_loss_weight: float = 0.1,
) -> CategoryQueryClassifier:
    """Initialize category queries from the pinned pretrained encoder."""
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(model_name, revision=revision)
    tokenized = tokenizer(
        list(category_descriptions), truncation=True, max_length=128,
        padding=True, return_tensors="pt",
    )
    device = next(encoder.parameters()).device
    with torch.no_grad():
        encoded = encoder(
            input_ids=tokenized["input_ids"].to(device),
            attention_mask=tokenized["attention_mask"].to(device),
        )
        label_query_init = encoded.last_hidden_state[:, 0].detach()
    model = CategoryQueryClassifier(
        encoder, label_query_init,
        encoder_chunk_batch_size=encoder_chunk_batch_size, dropout=dropout,
        context_layers=context_layers, attention_heads=attention_heads,
        contract_loss_weight=contract_loss_weight,
        evidence_loss_weight=evidence_loss_weight,
    )
    return model
