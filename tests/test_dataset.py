import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.chunking.chunk import Chunk
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.labels.loader import AnswerSpan, Contract
from src.labels.projection import LabeledChunk
from src.models.encoder.dataset import ChunkCollator, ChunkDataset, build_labeled_chunks

TOKENIZER_NAME = "distilbert/distilroberta-base"


def make_contract(contract_id: str, text: str, *spans: tuple[str, int, int]) -> Contract:
    """Build a grounded Contract whose spans are (category, start, end) slices of text."""
    answer_spans = [
        AnswerSpan(category=category, start=start, end=end, text=text[start:end])
        for category, start, end in spans
    ]
    return Contract(contract_id=contract_id, text=text, spans=answer_spans)


def expected_labels(*categories: str) -> list[int]:
    """Build the multi-hot vector with a 1 at the column of each given category."""
    labels = [0] * len(CATEGORIES)
    for category in categories:
        labels[CATEGORY_TO_INDEX[category]] = 1
    return labels


# ----- Wrapper: golden output on small handmade contracts -----
def test_build_labeled_chunks_two_contracts_gives_expected_chunks_and_labels():
    """Checks chunk count, order, offsets and labels on two hand-computed contracts."""
    # "a" has 15 chars -> window 10, overlap 0 -> chunks [0,10) and [10,15)
    contract_a = make_contract(
        "a", "AAAAABBBBBCCCCC",
        ("Parties", 2, 5),           # lies only in chunk [0,10)
        ("Governing Law", 12, 15),   # lies only in chunk [10,15)
    )
    # "b" has 10 chars -> a single chunk [0,10) and no spans
    contract_b = make_contract("b", "x" * 10)

    result = build_labeled_chunks([contract_a, contract_b], window_chars=10, overlap_chars=0)

    assert [(r.chunk.contract_id, r.chunk.start, r.chunk.end) for r in result] == [
        ("a", 0, 10), ("a", 10, 15), ("b", 0, 10),
    ]
    assert result[0].labels == expected_labels("Parties")
    assert result[1].labels == expected_labels("Governing Law")
    assert result[2].labels == [0] * len(CATEGORIES)


def test_build_labeled_chunks_span_straddling_boundary_labels_both_chunks():
    """Checks a span crossing a chunk boundary labels every chunk it overlaps."""
    # Span [8,12) overlaps chunk [0,10) by 2 chars and chunk [10,15) by 2 chars
    contract = make_contract("a", "AAAAABBBBBCCCCC", ("Parties", 8, 12))

    result = build_labeled_chunks([contract], window_chars=10, overlap_chars=0)

    assert result[0].labels == expected_labels("Parties")
    assert result[1].labels == expected_labels("Parties")


def test_build_labeled_chunks_overlapping_windows_labels_span_in_each_overlapping_chunk():
    """Checks a span inside the overlap region is labeled in both neighboring chunks."""
    # window 10, overlap 5 -> step 5 -> chunks [0,10), [5,15); span [6,9) sits in both
    contract = make_contract("a", "AAAAABBBBBCCCCC", ("Non-Compete", 6, 9))

    result = build_labeled_chunks([contract], window_chars=10, overlap_chars=5)

    assert [(r.chunk.start, r.chunk.end) for r in result] == [(0, 10), (5, 15)]
    assert all(r.labels == expected_labels("Non-Compete") for r in result)


# ----- Wrapper: ordering and edge cases -----
def test_build_labeled_chunks_preserves_contract_then_chunk_order():
    """Checks output is grouped by contract in input order, with ascending chunk_index."""
    contracts = [make_contract("b", "x" * 25), make_contract("a", "y" * 25)]

    result = build_labeled_chunks(contracts, window_chars=10, overlap_chars=0)

    assert [(r.chunk.contract_id, r.chunk.chunk_index) for r in result] == [
        ("b", 0), ("b", 1), ("b", 2), ("a", 0), ("a", 1), ("a", 2),
    ]


def test_build_labeled_chunks_empty_contract_list_gives_empty_list():
    """Checks no contracts yields no labeled chunks."""
    assert build_labeled_chunks([], window_chars=10, overlap_chars=0) == []


def test_build_labeled_chunks_chunk_text_matches_original_contract_text():
    """Checks every chunk is an exact slice of its contract's unaltered text."""
    contract = make_contract("a", "The quick brown fox jumps over the lazy dog.")

    result = build_labeled_chunks([contract], window_chars=12, overlap_chars=4)

    assert all(contract.text[r.chunk.start:r.chunk.end] == r.chunk.text for r in result)


def test_build_labeled_chunks_invalid_window_raises_value_error():
    """Checks bad window/overlap values from the chunker propagate as ValueError."""
    contract = make_contract("a", "x" * 10)

    with pytest.raises(ValueError):
        build_labeled_chunks([contract], window_chars=10, overlap_chars=10)


@pytest.fixture(scope="module")
def tokenizer():
    """Load the DistilRoBERTa tokenizer once for the module (tokenizer files only, no weights)."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def make_labeled_chunk(text: str, *categories: str, chunk_index: int = 0) -> LabeledChunk:
    """Build a LabeledChunk over text with a 1 at the column of each given category."""
    chunk = Chunk(contract_id="c", start=0, end=len(text), text=text, chunk_index=chunk_index)
    return LabeledChunk(chunk=chunk, labels=expected_labels(*categories))


# ----- ChunkDataset: size, keys and item structure -----
def test_chunk_dataset_len_matches_number_of_labeled_chunks(tokenizer):
    """Checks the dataset length equals the number of labeled chunks it was given."""
    chunks = [make_labeled_chunk("First chunk."), make_labeled_chunk("Second chunk.", chunk_index=1)]

    assert len(ChunkDataset(chunks, tokenizer, max_length=32)) == 2


def test_chunk_dataset_empty_list_gives_empty_dataset(tokenizer):
    """Checks no chunks yields a length-0 dataset (the tokenizer itself raises on [])."""
    assert len(ChunkDataset([], tokenizer, max_length=32)) == 0


def test_chunk_dataset_item_has_exactly_the_trainer_keys(tokenizer):
    """Checks an item is a dict with input_ids, attention_mask and labels only."""
    dataset = ChunkDataset([make_labeled_chunk("Some text.")], tokenizer, max_length=32)

    assert set(dataset[0].keys()) == {"input_ids", "attention_mask", "labels"}


# ----- ChunkDataset: tokenization -----
def test_chunk_dataset_known_sentence_gives_golden_token_ids(tokenizer):
    """Checks a fixed sentence maps to the known RoBERTa ids, wrapped in <s> (0) and </s> (2)."""
    chunk = make_labeled_chunk("This Agreement is governed by Delaware law.", "Governing Law")

    item = ChunkDataset([chunk], tokenizer, max_length=32)[0]

    assert item["input_ids"] == [0, 713, 8759, 16, 22635, 30, 8951, 488, 4, 2]
    assert item["attention_mask"] == [1] * 10


def test_chunk_dataset_short_text_is_not_padded(tokenizer):
    """Checks input_ids and attention_mask have the same natural length and the mask is all ones."""
    item = ChunkDataset([make_labeled_chunk("Short text.")], tokenizer, max_length=512)[0]

    assert len(item["input_ids"]) == len(item["attention_mask"])
    assert len(item["input_ids"]) < 512
    assert set(item["attention_mask"]) == {1}


def test_chunk_dataset_long_text_is_truncated_to_max_length(tokenizer):
    """Checks a chunk over the limit is cut to exactly max_length tokens, still ending in </s>."""
    item = ChunkDataset([make_labeled_chunk("word " * 200)], tokenizer, max_length=8)[0]

    assert len(item["input_ids"]) == 8
    assert len(item["attention_mask"]) == 8
    assert item["input_ids"][-1] == tokenizer.eos_token_id


def test_chunk_dataset_chunks_of_different_length_keep_different_lengths(tokenizer):
    """Checks each item keeps its own length, so padding is left to the collator."""
    chunks = [make_labeled_chunk("Hi."), make_labeled_chunk("A much longer sentence here.", chunk_index=1)]
    dataset = ChunkDataset(chunks, tokenizer, max_length=64)

    assert len(dataset[0]["input_ids"]) < len(dataset[1]["input_ids"])


# ----- ChunkDataset: labels -----
def test_chunk_dataset_labels_are_float32_vector_over_all_categories(tokenizer):
    """Checks labels is a float32 tensor of shape (41,), as BCE targets must be floats."""
    labels = ChunkDataset([make_labeled_chunk("Some text.", "Parties")], tokenizer, max_length=32)[0]["labels"]

    assert isinstance(labels, torch.Tensor)
    assert labels.dtype == torch.float32
    assert labels.shape == (len(CATEGORIES),) == (41,)


def test_chunk_dataset_labels_match_multi_hot_vector_of_the_chunk(tokenizer):
    """Checks the label tensor holds 1.0 exactly at the columns of the chunk's categories."""
    chunk = make_labeled_chunk("Some text.", "Parties", "Governing Law")

    labels = ChunkDataset([chunk], tokenizer, max_length=32)[0]["labels"]

    assert labels[CATEGORY_TO_INDEX["Parties"]].item() == 1.0
    assert labels[CATEGORY_TO_INDEX["Governing Law"]].item() == 1.0
    assert labels.sum().item() == 2.0


def test_chunk_dataset_chunk_without_labels_gives_all_zero_tensor(tokenizer):
    """Checks an unlabeled chunk produces an all-zero float vector."""
    labels = ChunkDataset([make_labeled_chunk("Some text.")], tokenizer, max_length=32)[0]["labels"]

    assert torch.equal(labels, torch.zeros(len(CATEGORIES), dtype=torch.float32))


# ----- ChunkDataset: ordering and integration with the wrapper -----
def test_chunk_dataset_preserves_input_order_of_labels(tokenizer):
    """Checks item i carries the labels of chunk i, so predictions can be matched back by index."""
    chunks = [
        make_labeled_chunk("First.", "Parties"),
        make_labeled_chunk("Second.", "Governing Law", chunk_index=1),
    ]
    dataset = ChunkDataset(chunks, tokenizer, max_length=32)

    assert dataset[0]["labels"][CATEGORY_TO_INDEX["Parties"]].item() == 1.0
    assert dataset[1]["labels"][CATEGORY_TO_INDEX["Governing Law"]].item() == 1.0
    assert dataset.labeled_chunks == chunks


def test_chunk_dataset_built_from_wrapper_output_has_one_item_per_chunk(tokenizer):
    """Checks build_labeled_chunks output feeds ChunkDataset end to end with labels intact."""
    contract = make_contract("a", "AAAAABBBBBCCCCC", ("Parties", 2, 5))

    dataset = ChunkDataset(build_labeled_chunks([contract], 10, 0), tokenizer, max_length=32)

    assert len(dataset) == 2
    assert dataset[0]["labels"][CATEGORY_TO_INDEX["Parties"]].item() == 1.0
    assert dataset[1]["labels"].sum().item() == 0.0


# ----- ChunkCollator: batching with padding -----
# Texts tokenize to 4, 11 and 6 tokens (with <s> and </s>), so the batch pads to 11
COLLATOR_SPECS = [
    ("Short.", "Anti-Assignment"),
    ("A somewhat longer sentence for padding purposes here.", "Governing Law"),
    ("Mid length text.", "Affiliate License-Licensee"),
]


def make_items(tokenizer, specs=COLLATOR_SPECS) -> list[dict]:
    """Build ChunkDataset items for (text, category) pairs, in order."""
    chunks = [make_labeled_chunk(text, category, chunk_index=i) for i, (text, category) in enumerate(specs)]
    dataset = ChunkDataset(chunks, tokenizer, max_length=32)
    return [dataset[i] for i in range(len(dataset))]


def test_chunk_collator_batch_has_exactly_the_trainer_keys(tokenizer):
    """Checks the batch holds input_ids, attention_mask and labels only."""
    batch = ChunkCollator(tokenizer)(make_items(tokenizer))

    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}


def test_chunk_collator_pads_token_fields_to_longest_item_and_stacks_labels(tokenizer):
    """Checks shapes: token fields (batch, longest = 11) and labels (batch, 41)."""
    batch = ChunkCollator(tokenizer)(make_items(tokenizer))

    assert tuple(batch["input_ids"].shape) == (3, 11)
    assert tuple(batch["attention_mask"].shape) == (3, 11)
    assert tuple(batch["labels"].shape) == (3, len(CATEGORIES))


def test_chunk_collator_attention_mask_marks_real_tokens_and_zeros_padding(tokenizer):
    """Checks each mask row sums to its item's token count and is 0 after the last real token."""
    batch = ChunkCollator(tokenizer)(make_items(tokenizer))

    assert batch["attention_mask"].sum(dim=1).tolist() == [4, 11, 6]
    assert batch["attention_mask"][0, 4:].sum().item() == 0
    assert batch["attention_mask"][2, 6:].sum().item() == 0


def test_chunk_collator_fills_padding_with_pad_token_and_keeps_real_ids(tokenizer):
    """Checks padded positions hold the pad id and the real positions keep the original ids."""
    items = make_items(tokenizer)

    batch = ChunkCollator(tokenizer)(items)

    assert (batch["input_ids"][0, 4:] == tokenizer.pad_token_id).all()
    assert batch["input_ids"][0, :4].tolist() == items[0]["input_ids"]
    assert batch["input_ids"][1].tolist() == items[1]["input_ids"]


def test_chunk_collator_labels_are_float32_and_match_each_item_in_order(tokenizer):
    """Checks labels are float32 and row i equals item i's label vector, so order is preserved."""
    items = make_items(tokenizer)

    labels = ChunkCollator(tokenizer)(items)["labels"]

    assert labels.dtype == torch.float32
    assert all(torch.equal(labels[i], items[i]["labels"]) for i in range(3))
    assert labels.argmax(dim=1).tolist() == [
        CATEGORY_TO_INDEX["Anti-Assignment"],
        CATEGORY_TO_INDEX["Governing Law"],
        CATEGORY_TO_INDEX["Affiliate License-Licensee"],
    ]


def test_chunk_collator_does_not_modify_the_input_items(tokenizer):
    """Checks the dataset items keep their labels key and values after batching."""
    items = make_items(tokenizer)
    labels_before = [item["labels"].clone() for item in items]

    ChunkCollator(tokenizer)(items)

    assert all(set(item.keys()) == {"input_ids", "attention_mask", "labels"} for item in items)
    assert all(torch.equal(item["labels"], before) for item, before in zip(items, labels_before))


def test_chunk_collator_single_item_batch_needs_no_padding(tokenizer):
    """Checks a one-item batch keeps its natural length with an all-ones mask."""
    batch = ChunkCollator(tokenizer)(make_items(tokenizer, COLLATOR_SPECS[:1]))

    assert tuple(batch["input_ids"].shape) == (1, 4)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert tuple(batch["labels"].shape) == (1, len(CATEGORIES))


def test_chunk_collator_works_as_dataloader_collate_fn(tokenizer):
    """Checks a DataLoader using the collator yields batches in order, with the last one smaller."""
    chunks = [make_labeled_chunk(text, category, chunk_index=i) for i, (text, category) in enumerate(COLLATOR_SPECS)]
    dataset = ChunkDataset(chunks, tokenizer, max_length=32)

    batches = list(DataLoader(dataset, batch_size=2, collate_fn=ChunkCollator(tokenizer)))

    assert [tuple(b["labels"].shape) for b in batches] == [(2, len(CATEGORIES)), (1, len(CATEGORIES))]
    assert tuple(batches[0]["input_ids"].shape) == (2, 11)
    assert tuple(batches[1]["input_ids"].shape) == (1, 6)
