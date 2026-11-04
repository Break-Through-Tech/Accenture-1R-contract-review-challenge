"""Neighbor-context augmentation for chunk-level word/character TF-IDF."""

from __future__ import annotations

from collections import defaultdict

from src.labels.projection import LabeledChunk

CONTEXT_SEPARATOR = "\n\n[ADJACENT CHUNK CONTEXT]\n\n"


def add_neighbor_context(chunks: list[LabeledChunk]) -> list[str]:
    """Augment each chunk with its immediately preceding/following chunk text.

    Labels remain those of the center chunk. Context is constructed per contract
    and in chunk-index order, so no other contracts or annotations are consulted.
    """
    by_contract: dict[str, list[LabeledChunk]] = defaultdict(list)
    for item in chunks:
        by_contract[item.chunk.contract_id].append(item)
    contexts: dict[tuple[str, int], str] = {}
    for contract_id, group in by_contract.items():
        group.sort(key=lambda item: item.chunk.chunk_index)
        for position, item in enumerate(group):
            parts = []
            if position:
                parts.append(group[position - 1].chunk.text)
            parts.append(item.chunk.text)
            if position + 1 < len(group):
                parts.append(group[position + 1].chunk.text)
            contexts[(contract_id, item.chunk.chunk_index)] = CONTEXT_SEPARATOR.join(parts)
    return [contexts[(item.chunk.contract_id, item.chunk.chunk_index)] for item in chunks]
