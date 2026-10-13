import numpy as np
from src.labels.categories import CATEGORIES
from src.labels.loader import Contract
from src.chunking.chunker import chunk_contract
from src.labels.projection import LabeledChunk, project_labels
from src.labels.splits import split_by_contract

def prepare_contract(
    contract: Contract,
    window_chars: int = 1500,
    overlap_chars: int = 200
) -> list[LabeledChunk]:
    """
    Convert one CUAD contract into labeled chunks.

    Uses the existing chunking and label-projection modules
    while preserving the original contract text and offsets.
    """

    chunks = chunk_contract(
        contract_id=contract.contract_id,
        text=contract.text,
        window_chars=window_chars,
        overlap_chars=overlap_chars
    )

    labeled_chunks = project_labels(
        chunks=chunks,
        spans=contract.spans
    )

    return labeled_chunks

def prepare_contracts(
    contracts: list[Contract],
    window_chars: int = 1500,
    overlap_chars: int = 200
) -> list[LabeledChunk]:
    """
    Process multiple CUAD contracts into a combined list
    of labeled chunks.
    """

    all_labeled_chunks = []

    for contract in contracts:
        labeled_chunks = prepare_contract(
            contract,
            window_chars=window_chars,
            overlap_chars=overlap_chars
        )

        all_labeled_chunks.extend(labeled_chunks)

    return all_labeled_chunks

def prepare_train_val(
    contracts: list[Contract],
    val_fraction: float = 0.2,
    seed: int = 42,
    window_chars: int = 1500,
    overlap_chars: int = 200
) -> tuple[list[LabeledChunk], list[LabeledChunk]]:
    """
    Split CUAD contracts into training and validation sets,
    then convert each set into labeled chunks.

    Splitting before chunking prevents contract-level data leakage.
    """

    train_contracts, val_contracts = split_by_contract(
        contracts,
        val_fraction=val_fraction,
        seed=seed
    )

    train_chunks = prepare_contracts(
        train_contracts,
        window_chars=window_chars,
        overlap_chars=overlap_chars
    )

    val_chunks = prepare_contracts(
        val_contracts,
        window_chars=window_chars,
        overlap_chars=overlap_chars
    )

    return train_chunks, val_chunks

def to_model_inputs(
    labeled_chunks: list[LabeledChunk]
) -> tuple[list[str], np.ndarray]:
    """
    Convert labeled chunks into text and label arrays
    compatible with the classification models.
    """

    texts = [item.chunk.text for item in labeled_chunks]

    labels = np.asarray(
        [item.labels for item in labeled_chunks],
        dtype=np.int8
    ).reshape(-1, len(CATEGORIES))

    return texts, labels