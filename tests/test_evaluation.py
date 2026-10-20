"""Hand-calculated boundary metrics on small, grounded annotations."""

import pytest

from src.chunking.evaluation import evaluate_boundaries, evaluate_chunks, validate_chunks
from src.chunking.chunk import Chunk
from src.chunking.chunker import chunk_contract
from src.labels.loader import AnswerSpan, Contract


def make_contract(spans: list[tuple[str, int, int]], contract_id: str = "T") -> Contract:
    """Build annotations into a ten-character source with known offsets."""
    text = "abcdefghij"
    return Contract(contract_id=contract_id, text=text, spans=[
        AnswerSpan(category=category, start=start, end=end, text=text[start:end])
        for category, start, end in spans
    ])


# Golden aggregate: chunks [0,4), [3,7), [6,10) for window=4, overlap=1.
def test_metrics_hand_calculated_spans_yield_exact_counts() -> None:
    """Four spans yield 75% cut, 50% contained, 50% cut/uncontained, and one oversized."""
    contract = make_contract([("A", 0, 3), ("A", 3, 5), ("A", 2, 5), ("A", 1, 8)])
    report = evaluate_boundaries([contract], window_chars=4, overlap_chars=1)
    assert report.metrics.model_dump() == {
        "window_chars": 4, "overlap_chars": 1,
        "raw_annotations": 4, "unique_annotations": 4, "duplicate_annotations": 0,
        "cut_annotations": 3, "contained_annotations": 2,
        "cut_not_contained_annotations": 2, "longer_than_window_annotations": 1,
        "cut_pct": 75.0, "contained_pct": 50.0,
        "cut_not_contained_pct": 50.0, "longer_than_window_pct": 25.0,
    }
    assert {example.kind for example in report.examples} == {
        "uncut_and_contained", "cut_and_contained", "cut_not_contained", "longer_than_window",
    }


# Endpoint equality never creates a cut; overlap can rescue a cut annotation.
@pytest.mark.parametrize("start,end", [(0, 4), (4, 8), (8, 10)])
def test_endpoint_touching_is_not_a_cut(start: int, end: int) -> None:
    """Annotations exactly on back-to-back chunk endpoints are contained and uncut."""
    metrics = evaluate_boundaries([make_contract([("A", start, end)])], 4, 0).metrics
    assert metrics.cut_annotations == 0
    assert metrics.contained_annotations == 1
    assert metrics.longer_than_window_annotations == 0


def test_overlap_cut_span_is_also_fully_contained() -> None:
    """[3,5) is cut at 4 but preserved in [3,7)."""
    report = evaluate_boundaries([make_contract([("A", 3, 5)])], 4, 1)
    assert report.metrics.cut_pct == 100
    assert report.metrics.contained_pct == 100
    assert report.metrics.cut_not_contained_pct == 0
    example = report.examples[0]
    assert example.cutting_chunks == [(0, 4)]
    assert example.containing_chunks == [(3, 7)]
    assert example.boundary_excerpts[0].boundary == 4
    assert example.boundary_excerpts[0].text == "abcdefghij"


def test_multiple_cuts_and_containments_count_annotation_once() -> None:
    """Several cutting or containing windows still contribute a single count."""
    metrics = evaluate_boundaries([make_contract([("A", 3, 5), ("A", 1, 9)])], 6, 4).metrics
    assert metrics.cut_annotations == 2
    assert metrics.contained_annotations == 1
    assert metrics.cut_not_contained_annotations == 1
    assert metrics.longer_than_window_annotations == 1


# Deduplication includes contract ID and category, across repeated contract records.
def test_duplicates_preserve_different_categories_and_contracts() -> None:
    """Seven raw entries reduce to four unique annotations without merging categories."""
    first = make_contract([("A", 3, 5), ("A", 3, 5), ("B", 3, 5), ("A", 0, 3)])
    repeated = make_contract([("A", 3, 5), ("B", 3, 5)])
    second = make_contract([("A", 3, 5)], contract_id="U")
    metrics = evaluate_boundaries([first, repeated, second], 4, 1).metrics
    assert (metrics.raw_annotations, metrics.unique_annotations, metrics.duplicate_annotations) == (7, 4, 3)
    assert metrics.cut_annotations == 3
    assert metrics.cut_pct == 75
    assert metrics.contained_pct == 100


def test_conflicting_contract_text_raises_value_error() -> None:
    """Repeated IDs with different coordinate sources are rejected."""
    with pytest.raises(ValueError, match="Conflicting text"):
        evaluate_boundaries([make_contract([]), Contract(contract_id="T", text="other", spans=[])], 4, 1)


# Empty denominators and invalid settings are explicit.
@pytest.mark.parametrize("contracts", [[], [make_contract([])]])
def test_no_annotations_returns_undefined_percentages(contracts: list[Contract]) -> None:
    """No annotations produce zero counts and None percentages rather than misleading zeros."""
    report = evaluate_boundaries(contracts, 4, 1)
    assert report.metrics.unique_annotations == 0
    assert report.metrics.cut_pct is None
    assert report.metrics.contained_pct is None
    assert report.metrics.cut_not_contained_pct is None
    assert report.metrics.longer_than_window_pct is None
    assert report.examples == []


@pytest.mark.parametrize("window,overlap", [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)])
def test_invalid_configuration_raises_even_without_annotations(window: int, overlap: int) -> None:
    """Invalid windows or overlap fail independently of the dataset contents."""
    with pytest.raises(ValueError):
        evaluate_boundaries([], window, overlap)


# Supplied chunk methods share annotation counting without assuming a character window.
def test_supplied_chunks_reuse_metrics_and_category_denominators() -> None:
    """Supplied chunks deduplicate within category while preserving identical other-category spans."""
    contract = make_contract([("Parties", 3, 5), ("Parties", 3, 5), ("Governing Law", 2, 5)])
    chunks = chunk_contract("T", contract.text, 4, 1)
    report = evaluate_chunks([contract], {"T": chunks})
    assert (report.metrics.raw_annotations, report.metrics.unique_annotations, report.metrics.duplicate_annotations) == (3, 2, 1)
    assert report.metrics.contained_pct == 50
    assert report.metrics.cut_not_contained_pct == 50
    assert report.metrics.longer_than_window_annotations is None
    parties = report.per_category["Parties"]
    assert (parties.raw_annotations, parties.unique_annotations, parties.duplicate_annotations) == (2, 1, 1)
    assert parties.contained_pct == 100
    assert report.per_category["Governing Law"].cut_not_contained_pct == 100
    assert report.per_category["Insurance"].contained_pct is None


def test_chunk_validation_gap_is_rejected() -> None:
    """A source gap invalidates coverage even when both endpoints are covered."""
    contract = make_contract([])
    chunks = [Chunk(contract_id="T", start=0, end=3, text="abc", chunk_index=0),
              Chunk(contract_id="T", start=4, end=10, text="efghij", chunk_index=1)]
    with pytest.raises(ValueError, match="no source gaps"):
        validate_chunks(contract, chunks)


def test_chunk_validation_wrong_slice_is_rejected() -> None:
    """Correct-length text must still match the original source exactly."""
    contract = make_contract([])
    chunks = [Chunk(contract_id="T", start=0, end=10, text="XXXXXXXXXX", chunk_index=0)]
    with pytest.raises(ValueError, match="source slice"):
        validate_chunks(contract, chunks)
