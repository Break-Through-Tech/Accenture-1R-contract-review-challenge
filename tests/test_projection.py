import pytest
from src.chunking.chunk import Chunk
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.labels.loader import AnswerSpan
from src.labels.projection import (
    LabeledChunk,
    project_chunk_labels,
    project_labels,
    spans_overlap,
)


def make_chunk(start: int, end: int, chunk_index: int = 0) -> Chunk:
    """Build a valid Chunk covering [start, end) with filler text of the right length."""
    return Chunk(contract_id="c", start=start, end=end,
                 text="x" * (end - start), chunk_index=chunk_index)


def make_span(category: str, start: int, end: int) -> AnswerSpan:
    """Build a valid AnswerSpan covering [start, end) with filler text of the right length."""
    return AnswerSpan(category=category, start=start, end=end, text="x" * (end - start))


def expected_labels(*categories: str) -> list[int]:
    """Build the multi-hot vector with a 1 at the column of each given category."""
    labels = [0] * len(CATEGORIES)
    for category in categories:
        labels[CATEGORY_TO_INDEX[category]] = 1
    return labels


# ----- Projection: overlap rule for end-exclusive ranges -----
@pytest.mark.parametrize("chunk, span, expected", [
    ((0, 10), (5, 15), True),     # partial overlap on the right
    ((0, 10), (2, 5), True),      # span fully inside chunk
    ((2, 5), (0, 10), True),      # chunk fully inside span
    ((0, 10), (9, 20), True),     # overlap of exactly one character
    ((0, 10), (10, 15), False),   # span starts exactly where chunk ends (touching)
    ((10, 20), (0, 10), False),   # span ends exactly where chunk starts (touching)
    ((0, 10), (20, 30), False),   # disjoint
])
def test_spans_overlap_end_exclusive_ranges_match_expected(chunk, span, expected):
    """Checks spans_overlap is True only when the ranges share at least one character."""
    assert spans_overlap(*chunk, *span) is expected


# ----- Projection: labels for a single chunk -----
def test_project_chunk_labels_span_inside_chunk_sets_only_its_category():
    """Checks a span inside the chunk switches on exactly that category's column."""
    result = project_chunk_labels(make_chunk(0, 10), [make_span("Parties", 2, 5)])

    assert result.labels == expected_labels("Parties")
    assert sum(result.labels) == 1


def test_project_chunk_labels_no_spans_gives_all_zeros():
    """Checks a chunk with no spans gets an all-zero label vector."""
    result = project_chunk_labels(make_chunk(0, 10), [])

    assert result.labels == [0] * len(CATEGORIES)


def test_project_chunk_labels_span_touching_chunk_start_gives_no_label():
    """Checks a span that ends exactly at chunk.start does not label the chunk."""
    result = project_chunk_labels(make_chunk(10, 20), [make_span("Parties", 5, 10)])

    assert result.labels == [0] * len(CATEGORIES)


def test_project_chunk_labels_two_categories_sets_both_columns():
    """Checks spans of different categories each switch on their own column."""
    spans = [make_span("Parties", 0, 3), make_span("Governing Law", 5, 8)]
    result = project_chunk_labels(make_chunk(0, 10), spans)

    assert result.labels == expected_labels("Parties", "Governing Law")


def test_project_chunk_labels_repeated_category_stays_one():
    """Checks several spans of one category still give a single 1, not a count."""
    spans = [make_span("Parties", 0, 3), make_span("Parties", 5, 8)]
    result = project_chunk_labels(make_chunk(0, 10), spans)

    assert result.labels == expected_labels("Parties")


# ----- Projection: labels across all chunks of a contract -----
def test_project_labels_straddling_span_labels_both_chunks():
    """Checks a span crossing a chunk boundary labels both chunks under any-overlap."""
    chunks = [make_chunk(0, 10, 0), make_chunk(10, 20, 1)]
    result = project_labels(chunks, [make_span("Parties", 8, 12)])

    assert [r.labels for r in result] == [expected_labels("Parties")] * 2


def test_project_labels_span_ending_at_boundary_labels_only_first_chunk():
    """Checks a span ending exactly at the second chunk's start labels only the first."""
    chunks = [make_chunk(0, 10, 0), make_chunk(10, 20, 1)]
    result = project_labels(chunks, [make_span("Parties", 5, 10)])

    assert [r.labels for r in result] == [expected_labels("Parties"), [0] * len(CATEGORIES)]


def test_project_labels_preserves_chunk_order_and_identity():
    """Checks the output has one LabeledChunk per input chunk, in the same order."""
    chunks = [make_chunk(0, 10, 0), make_chunk(10, 20, 1), make_chunk(20, 30, 2)]
    result = project_labels(chunks, [])

    assert [r.chunk for r in result] == chunks


# ----- LabeledChunk: label vector validation -----
def test_labeled_chunk_all_zero_vector_is_valid():
    """Checks an all-zero label vector (a chunk with no clauses) is accepted."""
    assert LabeledChunk(chunk=make_chunk(0, 10), labels=[0] * len(CATEGORIES)).labels == [0] * 41


def test_labeled_chunk_wrong_length_raises_value_error():
    """Checks a label vector that is not one entry per category is rejected."""
    with pytest.raises(ValueError, match="Expected 41 labels"):
        LabeledChunk(chunk=make_chunk(0, 10), labels=[0] * 3)


def test_labeled_chunk_non_binary_value_raises_value_error():
    """Checks a label value other than 0 or 1 is rejected."""
    labels = [0] * len(CATEGORIES)
    labels[0] = 2
    with pytest.raises(ValueError, match="must be 0 or 1"):
        LabeledChunk(chunk=make_chunk(0, 10), labels=labels)
