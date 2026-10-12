from src.labels.loader import Contract
from src.labels.projection import LabeledChunk, project_labels
from src.chunking.chunker import chunk_contract
from torch import tensor, float32
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


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


class ChunkDataset(Dataset):
    """
    Tokenized LabeledChunks for the Hugging Face Trainer.
    Each item is a dict with input_ids and attention_mask (plain lists, truncated to
    max_length but not padded, so the collator pads each batch) and labels (a float32
    vector over CATEGORIES, as BCE needs float targets). All chunks are tokenized once
    at construction. labeled_chunks stays accessible so predictions can be matched to
    chunks in the same order.
    """

    def __init__(self, labeled_chunks: list[LabeledChunk], tokenizer: PreTrainedTokenizerBase,
                 max_length: int) -> None:
        """Tokenize every chunk's text once, truncating to max_length tokens without padding."""
        # Keep the labeled chunks so labels and predictions stay in this order
        self.labeled_chunks = labeled_chunks
        texts = [lc.chunk.text for lc in labeled_chunks]

        # Handle empty input, since the tokenizer raises on an empty list
        if not texts:
            self.encodings = {"input_ids": [], "attention_mask": []}
            return

        # Tokenize all chunk texts in one batched call
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)

    def __len__(self) -> int:
        """Return the number of chunks in the dataset."""
        return len(self.labeled_chunks)

    def __getitem__(self, index: int) -> dict:
        """Return input_ids, attention_mask and the float32 label vector for one chunk."""
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": tensor(self.labeled_chunks[index].labels, dtype=float32),
        }