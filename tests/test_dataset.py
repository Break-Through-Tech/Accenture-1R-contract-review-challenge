import pytest
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.labels.loader import AnswerSpan, Contract
from src.models.encoder.dataset import build_labeled_chunks


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
