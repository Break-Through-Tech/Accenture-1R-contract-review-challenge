"""Contract-sequence dataset and padding collator for long-context modeling."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import Dataset

from src.labels.projection import LabeledChunk


class ContractChunkDataset(Dataset):
    """Group tokenized labeled chunks into complete, ordered contract sequences."""

    def __init__(self, chunks: list[LabeledChunk], tokenizer: Any, max_length: int):
        grouped: dict[str, list[LabeledChunk]] = defaultdict(list)
        for item in chunks:
            grouped[item.chunk.contract_id].append(item)
        self.contracts = []
        for contract_id, items in grouped.items():
            items.sort(key=lambda item: item.chunk.chunk_index)
            encoded = tokenizer(
                [item.chunk.text for item in items], truncation=True,
                max_length=max_length, padding=False,
            )
            self.contracts.append({
                "contract_id": contract_id,
                "input_ids": encoded["input_ids"],
                "labels": [item.labels for item in items],
                "chunks": [item.chunk for item in items],
            })
        self.contracts.sort(key=lambda item: item["contract_id"])

    def __len__(self) -> int:
        return len(self.contracts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.contracts[index]


def collate_contract_sequences(items: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    """Pad contracts and chunk token sequences into [B, chunks, tokens] tensors."""
    if not items:
        raise ValueError("cannot collate an empty contract batch")
    batch_size = len(items)
    max_chunks = max(len(item["input_ids"]) for item in items)
    max_tokens = max(len(ids) for item in items for ids in item["input_ids"])
    num_labels = len(items[0]["labels"][0])
    input_ids = torch.full((batch_size, max_chunks, max_tokens), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    chunk_mask = torch.zeros((batch_size, max_chunks), dtype=torch.bool)
    labels = torch.zeros((batch_size, max_chunks, num_labels), dtype=torch.float32)

    for batch_index, item in enumerate(items):
        for chunk_index, (token_ids, row_labels) in enumerate(zip(item["input_ids"], item["labels"])):
            length = len(token_ids)
            input_ids[batch_index, chunk_index, :length] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[batch_index, chunk_index, :length] = 1
            chunk_mask[batch_index, chunk_index] = True
            labels[batch_index, chunk_index] = torch.tensor(row_labels, dtype=torch.float32)
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "chunk_mask": chunk_mask, "labels": labels,
            "contract_ids": [item["contract_id"] for item in items]}
