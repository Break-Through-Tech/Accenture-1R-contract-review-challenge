import json
from pathlib import Path

import pytest
from src.chunking.chunker import chunk_contract

CUAD_TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "cuad" / "train_separate_questions.json"


def make_chunks(text, window=4, overlap=1):
    """Run chunk_contract with a fixed test contract id and small default window/overlap."""
    return chunk_contract("test_contract4", text, window, overlap)


# ----- Golden tests: hard-coded offsets worked out by hand -----
def test_golden_offsets_length_10():
    """A 10-char text yields exactly the hand-computed (start, end) spans."""
    chunks = make_chunks("abcdefghij")      # window=4, overlap=1 by default
    offsets = [(c.start, c.end) for c in chunks]
    assert offsets == [(0, 4), (3, 7), (6, 10)]


def test_golden_offsets_length_11():
    """An 11-char text ends with a short tail chunk (9, 11)."""
    chunks = make_chunks("abcdefghijk")
    offsets = [(c.start, c.end) for c in chunks]
    assert offsets == [(0, 4), (3, 7), (6, 10), (9, 11)]


# ----- Core invariants -----
def test_round_trip_text_matches_source_slice():
    """Each chunk's text equals the source text sliced at its start and end."""
    text = "abcdefghijk"
    for c in make_chunks(text):
        assert text[c.start:c.end] == c.text


def test_chunk_index_is_sequential_from_zero():
    """chunk_index runs 0..n-1 in order with no gaps."""
    chunks = make_chunks("abcdefghijk")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_are_sorted_by_start():
    """Chunk starts are strictly increasing and unique."""
    chunks = make_chunks("abcdefghijk")
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_coverage_first_starts_at_zero_and_last_ends_at_text_length():
    """The chunks span the whole text, from offset 0 to len(text)."""
    text = "abcdefghijk"
    chunks = make_chunks(text)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)


def test_no_gaps_between_consecutive_chunks():
    """Each chunk starts at or before the previous chunk's end."""
    chunks = make_chunks("abcdefghijk")
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start <= prev.end


def test_contract_id_is_set_on_every_chunk():
    """Every chunk carries the contract id passed to the chunker."""
    chunks = make_chunks("abcdefghijk")
    assert all(c.contract_id == "test_contract4" for c in chunks)


# ----- Edge cases -----
def test_empty_text_returns_empty_list():
    """Empty text produces no chunks."""
    assert make_chunks("") == []


def test_text_shorter_than_window_returns_single_chunk():
    """Text shorter than the window becomes one chunk covering all of it."""
    chunks = make_chunks("abc")
    assert [(c.start, c.end) for c in chunks] == [(0, 3)]
    assert chunks[0].text == "abc"


def test_text_equal_to_window_returns_single_chunk():
    """Text exactly one window long gives one chunk and no redundant tail."""
    chunks = make_chunks("abcd")
    assert [(c.start, c.end) for c in chunks] == [(0, 4)]


def test_zero_overlap_chunks_are_back_to_back():
    """With zero overlap, chunks abut and rejoin into the original text."""
    text = "abcdefghij"
    chunks = make_chunks(text, window=4, overlap=0)
    assert [(c.start, c.end) for c in chunks] == [(0, 4), (4, 8), (8, 10)]
    assert "".join(c.text for c in chunks) == text


# Invalid parameters: overlap >= window, non-positive window, negative overlap
@pytest.mark.parametrize("window, overlap", [
    (4, 4),     # overlap equal to window
    (4, 5),     # overlap greater than window
    (0, 0),     # zero window
    (-1, 0),    # negative window
    (4, -1),    # negative overlap
])
def test_invalid_parameters_raise(window, overlap):
    """Invalid window/overlap combinations raise ValueError."""
    with pytest.raises(ValueError):
        make_chunks("abcdefghij", window=window, overlap=overlap)


# Real CUAD data: newlines, unicode and odd whitespace expose offset bugs
@pytest.fixture(scope="module")
def cuad_contracts():
    """Load (title, text) for the first 5 CUAD training contracts, skipping if the file is missing."""
    if not CUAD_TRAIN_PATH.exists():
        pytest.skip(f"CUAD training file not found at {CUAD_TRAIN_PATH}")
    with open(CUAD_TRAIN_PATH, encoding="utf-8") as f:
        data = json.load(f)["data"]
    return [(item["title"], item["paragraphs"][0]["context"]) for item in data[:5]]


def test_real_cuad_contracts_satisfy_invariants(cuad_contracts):
    """Real contracts chunked at 1500/200 keep exact offsets, indices and full coverage."""
    for title, text in cuad_contracts:
        chunks = chunk_contract(title, text, 1500, 200)

        # Round trip, ordering and sequential indices
        for i, c in enumerate(chunks):
            assert text[c.start:c.end] == c.text
            assert c.chunk_index == i
            assert c.contract_id == title

        # Full coverage with no gaps
        assert chunks[0].start == 0
        assert chunks[-1].end == len(text)
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt.start > prev.start
            assert nxt.start <= prev.end
