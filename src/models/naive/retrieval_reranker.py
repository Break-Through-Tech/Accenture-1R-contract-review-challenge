"""High-recall sparse candidate retrieval followed by category-conditioned reranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CandidatePair:
    """One candidate chunk/category decision, with stable contract-local coordinates."""

    contract_id: str
    chunk_index: int
    chunk_row: int
    category_index: int
    retrieval_score: float
    target: int | None = None


@dataclass(frozen=True)
class PairExample:
    """Text pair plus binary target used to train the cross-encoder."""

    prompt: str
    text: str
    category_index: int
    chunk_row: int
    target: int


def retrieve_top_k(
    probabilities: np.ndarray,
    chunks: Sequence[Any],
    *,
    top_k: int,
    active_categories: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> list[CandidatePair]:
    """Select each category's top-K chunks independently within each contract.

    Ties are resolved by increasing chunk index. Scores must be aligned with the
    supplied chunk list; this function does not inspect labels when ranking.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if probabilities.ndim != 2 or probabilities.shape[0] != len(chunks):
        raise ValueError("probabilities must have one row per chunk")
    num_labels = probabilities.shape[1]
    if active_categories is None:
        active_categories = np.ones(num_labels, dtype=bool)
    if active_categories.shape != (num_labels,):
        raise ValueError("active_categories must have one value per category")
    if labels is not None and labels.shape != probabilities.shape:
        raise ValueError("labels must align with probabilities")

    grouped: dict[str, list[int]] = {}
    for row, item in enumerate(chunks):
        grouped.setdefault(item.chunk.contract_id, []).append(row)

    candidates: list[CandidatePair] = []
    for contract_id, rows in grouped.items():
        ordered_rows = sorted(rows, key=lambda row: chunks[row].chunk.chunk_index)
        for category_index in np.flatnonzero(active_categories):
            ranked = sorted(
                ordered_rows,
                key=lambda row: (-float(probabilities[row, category_index]),
                                 chunks[row].chunk.chunk_index),
            )[:top_k]
            for row in ranked:
                candidates.append(CandidatePair(
                    contract_id=contract_id,
                    chunk_index=chunks[row].chunk.chunk_index,
                    chunk_row=row,
                    category_index=int(category_index),
                    retrieval_score=float(probabilities[row, category_index]),
                    target=(int(labels[row, category_index]) if labels is not None else None),
                ))
    return candidates


def build_training_pairs(
    chunks: Sequence[Any],
    labels: np.ndarray,
    retrieval_probabilities: np.ndarray,
    prompts: Sequence[str],
    *,
    top_k: int,
    negatives_per_positive: int = 3,
    min_hard_negatives: int = 8,
    seed: int = 42,
) -> tuple[list[PairExample], np.ndarray]:
    """Use all gold positives and retriever-ranked false positives as hard negatives."""
    if labels.shape != (len(chunks), len(prompts)):
        raise ValueError("labels must have shape [chunks, prompts]")
    active = labels.sum(axis=0) > 0
    candidates = retrieve_top_k(
        retrieval_probabilities, chunks, top_k=top_k, active_categories=active,
        labels=labels,
    )
    rng = np.random.default_rng(seed)
    examples: list[PairExample] = []
    hard_negatives: dict[int, list[CandidatePair]] = {int(i): [] for i in np.flatnonzero(active)}
    for candidate in candidates:
        if candidate.target == 0:
            hard_negatives[candidate.category_index].append(candidate)

    for category_index in np.flatnonzero(active):
        positive_rows = np.flatnonzero(labels[:, category_index] == 1)
        for row in positive_rows:
            examples.append(PairExample(
                prompt=prompts[category_index], text=chunks[row].chunk.text,
                category_index=int(category_index), chunk_row=int(row), target=1,
            ))
        negatives = sorted(
            hard_negatives[int(category_index)],
            key=lambda candidate: (-candidate.retrieval_score, candidate.chunk_index),
        )
        requested = max(min_hard_negatives, negatives_per_positive * len(positive_rows))
        if len(negatives) > requested:
            chosen = rng.choice(len(negatives), size=requested, replace=False)
            negatives = [negatives[int(index)] for index in sorted(chosen)]
        for candidate in negatives:
            examples.append(PairExample(
                prompt=prompts[int(category_index)],
                text=chunks[candidate.chunk_row].chunk.text,
                category_index=int(category_index), chunk_row=candidate.chunk_row,
                target=0,
            ))
    if not examples:
        raise ValueError("training data has no positive category examples")
    return examples, active


def candidate_recall(
    labels: np.ndarray,
    chunks: Sequence[Any],
    candidates: Sequence[CandidatePair],
) -> dict[str, float | int | None]:
    """Report candidate recall over positive chunk/category pairs and contract labels."""
    positives = {
        (item.chunk.contract_id, item.chunk.chunk_index, category_index)
        for row, item in enumerate(chunks)
        for category_index in np.flatnonzero(labels[row])
    }
    retrieved_positives = {
        (candidate.contract_id, candidate.chunk_index, candidate.category_index)
        for candidate in candidates
        if labels[candidate.chunk_row, candidate.category_index] == 1
    }
    positive_contract_categories = {
        (contract_id, category_index)
        for contract_id, _, category_index in positives
    }
    retrieved_contract_categories = {
        (contract_id, category_index)
        for contract_id, chunk_index, category_index in retrieved_positives
    }
    return {
        "positive_chunk_category_pairs": len(positives),
        "retrieved_positive_pairs": len(positives & retrieved_positives),
        "pair_recall": len(positives & retrieved_positives) / len(positives) if positives else None,
        "positive_contract_categories": len(positive_contract_categories),
        "retrieved_positive_contract_categories": len(
            positive_contract_categories & retrieved_contract_categories
        ),
        "contract_category_recall": (
            len(positive_contract_categories & retrieved_contract_categories)
            / len(positive_contract_categories) if positive_contract_categories else None
        ),
    }


class PairDataset(Dataset):
    """Minimal dataset; pair tokenization and padding stay in the collator."""

    def __init__(self, examples: Sequence[PairExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PairExample:
        return self.examples[index]


def collate_pairs(examples: Sequence[PairExample], tokenizer, max_length: int) -> dict[str, Tensor]:
    """Tokenize category prompt/text pairs with dynamic padding and explicit limits."""
    encoded = tokenizer(
        [example.prompt for example in examples],
        [example.text for example in examples],
        truncation="longest_first", max_length=max_length,
        padding=True, return_tensors="pt",
    )
    encoded["labels"] = torch.tensor([example.target for example in examples], dtype=torch.float32)
    return encoded


class PairCrossEncoder(nn.Module):
    """Pretrained transformer scoring a (category description, chunk) pair."""

    def __init__(self, encoder: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(int(encoder.config.hidden_size), 1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor,
                labels: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(output.last_hidden_state[:, 0])).squeeze(-1)
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return logits, loss


def build_pair_cross_encoder(model_name: str, revision: str | None = None,
                              dropout: float = 0.1) -> PairCrossEncoder:
    """Load a pretrained encoder and attach a binary candidate-pair scoring head."""
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(model_name, revision=revision)
    return PairCrossEncoder(encoder, dropout=dropout)


def dense_candidate_scores(
    chunks: Sequence[Any],
    candidates: Sequence[CandidatePair],
    candidate_probabilities: np.ndarray,
    num_labels: int,
) -> np.ndarray:
    """Scatter reranker probabilities into the dense chunk-by-category score matrix."""
    if candidate_probabilities.shape != (len(candidates),):
        raise ValueError("candidate probabilities must align with candidate pairs")
    scores = np.zeros((len(chunks), num_labels), dtype=np.float32)
    for candidate, probability in zip(candidates, candidate_probabilities):
        scores[candidate.chunk_row, candidate.category_index] = float(probability)
    return scores
