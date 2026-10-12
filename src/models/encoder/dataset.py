from src.labels.loader import Contract
from src.labels.projection import LabeledChunk, project_labels
from src.chunking.chunker import chunk_contract


def build_labeled_chunks(contracts: list[Contract], window_chars: int, overlap_chars: int) -> list[LabeledChunk]:
    """
    Chunk every contract and project its answer spans onto the chunks.
    Returns one flat list of LabeledChunk, ordered by contract and then by chunk
    position. Uses the same chunk_contract as the online pipeline, so training and
    inference see identical chunks.
    """
    labeled: list[LabeledChunk] = []
    for contract in contracts:
        # Chunk the unaltered contract text with the configured window and overlap
        chunks = chunk_contract(contract.contract_id, contract.text, window_chars, overlap_chars)

        # Turn the contract's answer spans into a multi-hot label vector per chunk
        labeled += project_labels(chunks, contract.spans)
    return labeled
