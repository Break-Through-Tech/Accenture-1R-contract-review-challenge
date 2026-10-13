
import numpy as np
from transformers import PreTrainedTokenizerFast

from src.chunking.token_chunker import chunk_sentence_windows
from src.labels.categories import CATEGORIES
from src.labels.loader import Contract
from src.labels.projection import LabeledChunk, project_labels
from src.labels.splits import split_by_contract


def prepare_contract(
    contract: Contract,
    tokenizer: PreTrainedTokenizerFast,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[LabeledChunk]:
    """
    Convert one CUAD contract into sentence-aware labeled chunks.

    Uses Ruva's sentence-aware chunker to preserve sentence boundaries
    and keep chunks within the model's token limit.
    """

    chunks = chunk_sentence_windows(
        contract_id=contract.contract_id,
        text=contract.text,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    labeled_chunks = project_labels(
        chunks=chunks,
        spans=contract.spans,
    )

    return labeled_chunks


def prepare_contracts(
    contracts: list[Contract],
    tokenizer: PreTrainedTokenizerFast,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[LabeledChunk]:
    """
    Process multiple CUAD contracts into sentence-aware labeled chunks.
    """

    all_labeled_chunks = []

    for contract in contracts:
        labeled_chunks = prepare_contract(
            contract=contract,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

        all_labeled_chunks.extend(labeled_chunks)

    return all_labeled_chunks


def prepare_train_val(
    contracts: list[Contract],
    tokenizer: PreTrainedTokenizerFast,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> tuple[list[LabeledChunk], list[LabeledChunk]]:
    """
    Split contracts before chunking to prevent contract-level leakage.

    Use the same tokenizer and chunking settings for both splits.
    """

    train_contracts, val_contracts = split_by_contract(
        contracts,
        val_fraction=val_fraction,
        seed=seed,
    )

    train_chunks = prepare_contracts(
        contracts=train_contracts,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    val_chunks = prepare_contracts(
        contracts=val_contracts,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    return train_chunks, val_chunks


def to_model_inputs(
    labeled_chunks: list[LabeledChunk],
) -> tuple[list[str], np.ndarray]:
    """
    Convert labeled chunks into texts and multi-label arrays
    for the classification models.
    """

    texts = [item.chunk.text for item in labeled_chunks]

    labels = np.asarray(
        [item.labels for item in labeled_chunks],
        dtype=np.int8,
    ).reshape(-1, len(CATEGORIES))

    return texts, labels
