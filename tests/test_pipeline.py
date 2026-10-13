
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORY_TO_INDEX, CATEGORIES
from src.labels.loader import Contract, AnswerSpan
from src.pipeline import (
    prepare_contract,
    prepare_train_val,
    to_model_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tokenizer():
    """Load the same tokenizer once for all pipeline tests."""
    return AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        revision=TOKENIZER_REVISION,
        use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache",
        local_files_only=True,
    )


def test_prepare_contract(tokenizer):
    """Verify chunk boundaries, token limits, and label projection."""

    # Use a long contract so the sentence-aware chunker must split it.
    text = "Service Agreement. " + " ".join(
        f"Section {i} states that the supplier will provide "
        "technical support and monthly progress reports to the customer."
        for i in range(70)
    )

    target = "Service Agreement"
    start = text.index(target)
    end = start + len(target)

    contract = Contract(
        contract_id="test_contract",
        text=text,
        spans=[
            AnswerSpan(
                category="Document Name",
                start=start,
                end=end,
                text=target,
            )
        ],
    )

    result = prepare_contract(
        contract,
        tokenizer=tokenizer,
        max_tokens=512,
        overlap_tokens=64,
    )

    assert len(result) > 1

    category_index = CATEGORY_TO_INDEX["Document Name"]

    for item in result:
        chunk = item.chunk

        # Verify original text offsets are preserved.
        assert 0 <= chunk.start < chunk.end <= len(contract.text)
        assert chunk.text == contract.text[chunk.start:chunk.end]

        # Verify chunks fit within DistilRoBERTa's 512-token limit.
        token_ids = tokenizer(
            chunk.text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        assert len(token_ids) <= 512

        # The existing labeling policy marks any overlapping chunk.
        overlaps_annotation = (
            chunk.start < end and chunk.end > start
        )

        assert item.labels[category_index] == int(overlaps_annotation)

    assert any(
        item.labels[category_index] == 1
        for item in result
    )


def test_train_val_split_no_leakage(tokenizer):
    """Verify train and validation chunks never share contracts."""

    contracts = [
        Contract(
            contract_id=f"contract_{i}",
            text=f"This is sample contract number {i}.",
            spans=[],
        )
        for i in range(5)
    ]

    train_chunks, val_chunks = prepare_train_val(
        contracts,
        tokenizer=tokenizer,
        val_fraction=0.2,
        seed=42,
        max_tokens=512,
        overlap_tokens=64,
    )

    train_ids = {
        item.chunk.contract_id for item in train_chunks
    }

    val_ids = {
        item.chunk.contract_id for item in val_chunks
    }

    assert len(train_ids) == 4
    assert len(val_ids) == 1

    # No contract may appear in both sets.
    assert train_ids.isdisjoint(val_ids)

    # All original contracts should be represented.
    original_ids = {c.contract_id for c in contracts}

    assert train_ids | val_ids == original_ids


def test_to_model_inputs(tokenizer):
    """Verify labeled chunks become model-ready texts and labels."""

    text = "Service Agreement. The parties accept all terms."
    target = "Service Agreement"

    contract = Contract(
        contract_id="test_contract",
        text=text,
        spans=[
            AnswerSpan(
                category="Document Name",
                start=0,
                end=len(target),
                text=target,
            )
        ],
    )

    labeled_chunks = prepare_contract(
        contract,
        tokenizer=tokenizer,
        max_tokens=512,
        overlap_tokens=64,
    )

    texts, labels = to_model_inputs(labeled_chunks)

    # Verify extracted texts match the generated chunks.
    assert texts == [
        item.chunk.text for item in labeled_chunks
    ]

    # The short contract should fit into a single chunk.
    assert len(texts) == 1

    # Verify multi-label classification dimensions.
    assert labels.shape == (1, len(CATEGORIES))

    category_index = CATEGORY_TO_INDEX["Document Name"]

    # Verify the correct category is positive.
    assert labels[0, category_index] == 1

    # No other categories should be positive.
    assert labels.sum() == 1

def test_multiple_categories_and_boundary_overlap(tokenizer):
    """Verify labels when a contract has multiple categories and a span crosses chunks."""

    text = " ".join(
        f"Clause {i} confirms that the supplier must deliver "
        "the agreed services under this contract."
        for i in range(20)
    )

    # First generate chunks without labels to find a real chunk boundary.
    contract = Contract(
        contract_id="multi_label_contract",
        text=text,
        spans=[],
    )

    initial_chunks = prepare_contract(
        contract,
        tokenizer=tokenizer,
        max_tokens=48,
        overlap_tokens=0,
    )

    assert len(initial_chunks) >= 2

    first = initial_chunks[0].chunk
    second = initial_chunks[1].chunk

    # The chunks should not overlap because overlap_tokens=0.
    assert first.end <= second.start

    # Create an annotation that crosses from the first chunk into the second.
    boundary_start = first.end - 5
    boundary_end = second.start + 5

    # Create a second annotation entirely inside the first chunk.
    other_start = first.start + 5
    other_end = first.start + 12

    primary_category = "Document Name"
    secondary_category = next(
        category for category in CATEGORIES
        if category != primary_category
    )

    contract_with_labels = Contract(
        contract_id="multi_label_contract",
        text=text,
        spans=[
            AnswerSpan(
                category=primary_category,
                start=boundary_start,
                end=boundary_end,
                text=text[boundary_start:boundary_end],
            ),
            AnswerSpan(
                category=secondary_category,
                start=other_start,
                end=other_end,
                text=text[other_start:other_end],
            ),
        ],
    )

    labeled_chunks = prepare_contract(
        contract_with_labels,
        tokenizer=tokenizer,
        max_tokens=48,
        overlap_tokens=0,
    )

    # Adding annotations must not change the chunk boundaries.
    assert [
        (item.chunk.start, item.chunk.end)
        for item in labeled_chunks
    ] == [
        (item.chunk.start, item.chunk.end)
        for item in initial_chunks
    ]

    primary_index = CATEGORY_TO_INDEX[primary_category]
    secondary_index = CATEGORY_TO_INDEX[secondary_category]

    # The boundary-crossing annotation should label both chunks.
    assert labeled_chunks[0].labels[primary_index] == 1
    assert labeled_chunks[1].labels[primary_index] == 1

    # The first chunk should have two positive label categories.
    assert labeled_chunks[0].labels[secondary_index] == 1

    # Verify every label follows the project's any-overlap policy.
    for item in labeled_chunks:
        chunk = item.chunk

        for span in contract_with_labels.spans:
            expected = int(
                chunk.start < span.end
                and chunk.end > span.start
            )

            category_index = CATEGORY_TO_INDEX[span.category]

            assert item.labels[category_index] == expected

    # Verify the final multi-label model input.
    texts, labels = to_model_inputs(labeled_chunks)

    assert len(texts) == len(labeled_chunks)
    assert labels.shape == (len(labeled_chunks), len(CATEGORIES))
    assert labels[0].sum() == 2