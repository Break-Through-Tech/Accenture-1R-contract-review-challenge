
import pytest
from transformers import AutoTokenizer

from experiments.token_chunking import token_chunk_contract
from src.labels.loader import Contract, AnswerSpan
from src.labels.projection import project_labels
from src.labels.categories import CATEGORY_TO_INDEX


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("distilroberta-base")


def test_token_limits_and_offsets(tokenizer):
    """Chunks must fit the model and preserve original text positions."""

    text = (
        "SECTION 1.\n"
        + "Both parties agree to the terms of this contract. " * 300
    )

    contract = Contract(
        contract_id="test_contract",
        text=text,
        spans=[]
    )

    chunks = token_chunk_contract(
        contract,
        tokenizer,
        window_tokens=480,
        overlap_tokens=120
    )

    assert len(chunks) > 1

    # Check that every chunk fits DistilRoBERTa's limit
    for chunk in chunks:
        token_count = len(
            tokenizer.encode(
                chunk.text,
                add_special_tokens=True,
                truncation=False
            )
        )

        assert token_count <= 512

        # Check original character offsets
        assert chunk.text == text[chunk.start:chunk.end]

    # Check full text coverage
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)

    for previous, current in zip(chunks, chunks[1:]):
        assert previous.end >= current.start


def test_label_projection_compatibility(tokenizer):
    """Token-aware chunks must work with existing CUAD label projection."""

    answer = "DISTRIBUTOR AGREEMENT"
    text = answer + "\n" + (
        "The distributor agrees to the contract terms. " * 80
    )

    contract = Contract(
        contract_id="test_contract",
        text=text,
        spans=[
            AnswerSpan(
                category="Document Name",
                start=0,
                end=len(answer),
                text=answer
            )
        ]
    )

    chunks = token_chunk_contract(
        contract,
        tokenizer,
        window_tokens=40,
        overlap_tokens=10
    )

    labeled_chunks = project_labels(chunks, contract.spans)
    category_index = CATEGORY_TO_INDEX["Document Name"]

    assert len(labeled_chunks) == len(chunks)
    assert any(item.labels[category_index] == 1 for item in labeled_chunks)

    for item in labeled_chunks:
        chunk = item.chunk

        expected = int(
            chunk.start < len(answer) and chunk.end > 0
        )

        assert item.labels[category_index] == expected


def test_empty_text_and_invalid_settings(tokenizer):
    """Empty text and invalid chunk settings are handled correctly."""

    contract = Contract(
        contract_id="empty_contract",
        text="",
        spans=[]
    )

    assert token_chunk_contract(contract, tokenizer) == []

    with pytest.raises(ValueError):
        token_chunk_contract(contract, tokenizer, window_tokens=0)

    with pytest.raises(ValueError):
        token_chunk_contract(
            contract,
            tokenizer,
            window_tokens=100,
            overlap_tokens=100
        )
