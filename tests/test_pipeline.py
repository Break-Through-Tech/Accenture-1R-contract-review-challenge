from src.labels.loader import Contract, AnswerSpan
from src.pipeline import (
    prepare_contract,
    prepare_train_val,
    to_model_inputs
)
from src.labels.categories import CATEGORY_TO_INDEX, CATEGORIES


def test_prepare_contract():
    contract = Contract(
        contract_id="test_contract",
        text="abcdefghij",
        spans=[
            AnswerSpan(
                category="Document Name",
                start=3,
                end=6,
                text="def"
            )
        ]
    )

    result = prepare_contract(
        contract,
        window_chars=4,
        overlap_chars=1
    )

    assert len(result) == 3

    assert [(item.chunk.start, item.chunk.end) for item in result] == [
        (0, 4),
        (3, 7),
        (6, 10)
    ]

    for item in result:
        chunk = item.chunk
        assert chunk.text == contract.text[chunk.start:chunk.end]

    category_index = CATEGORY_TO_INDEX["Document Name"]

    assert result[0].labels[category_index] == 1
    assert result[1].labels[category_index] == 1
    assert result[2].labels[category_index] == 0

def test_train_val_split_no_leakage():
    # Create five sample contracts
    contracts = [
        Contract(
            contract_id=f"contract_{i}",
            text=f"This is sample contract number {i}.",
            spans=[]
        )
        for i in range(5)
    ]

    train_chunks, val_chunks = prepare_train_val(
        contracts,
        val_fraction=0.2,
        seed=42,
        window_chars=10,
        overlap_chars=2
    )

    train_ids = {
        item.chunk.contract_id for item in train_chunks
    }

    val_ids = {
        item.chunk.contract_id for item in val_chunks
    }

    assert len(train_ids) == 4
    assert len(val_ids) == 1

    # Verify that no contract appears in both sets
    assert train_ids.isdisjoint(val_ids)

    # Verify that every original contract is included
    original_ids = {c.contract_id for c in contracts}
    assert train_ids | val_ids == original_ids

def test_to_model_inputs():
    # sample contract with one annotated category
    contract = Contract(
        contract_id="test_contract",
        text="abcdefghij",
        spans=[
            AnswerSpan(
                category="Document Name",
                start=3,
                end=6,
                text="def"
            )
        ]
    )

    #labeled chunks
    labeled_chunks = prepare_contract(
        contract,
        window_chars=4,
        overlap_chars=1
    )


    texts, labels = to_model_inputs(labeled_chunks)

    # Verify the extracted texts
    assert texts == ["abcd", "defg", "ghij"]

    # Verify dimensions
    assert labels.shape == (3, len(CATEGORIES))

    category_index = CATEGORY_TO_INDEX["Document Name"]

    assert labels[:, category_index].tolist() == [1, 1, 0]


    assert labels.sum(axis=1).tolist() == [1, 1, 0]