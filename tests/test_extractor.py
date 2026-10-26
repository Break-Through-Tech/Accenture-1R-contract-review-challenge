from pathlib import Path

import pytest

from src.chunking.chunk import Chunk
from src.chunking.chunker import chunk_contract
from src.evidence.cues import CATEGORY_DESCRIPTIONS_PATH, load_category_cues
from src.evidence.extractor import (
    GroundedEvidence,
    Unknown,
    _sentence_spans,
    extract_evidence,
    extract_for_predictions,
)
from src.labels.loader import load_cuad

CUAD_TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "cuad" / "train_separate_questions.json"

# Sentences sit at (0, 31), (32, 77), (78, 100), (101, 120); len 120 (worked out by hand)
TEXT = (
    "This Agreement is signed today. "
    "The State of Delaware governs this Agreement. "
    "Insurance is required; "
    "notices go by mail."
)
CUES = {"governinglaw": ["governs", "state"], "insurance": ["insurance", "required"], "notice": ["mail"]}


def make_chunk(text: str = TEXT, start: int = 0, contract_id: str = "c1", chunk_index: int = 0) -> Chunk:
    """Wrap text in a Chunk at the given absolute start offset."""
    return Chunk(contract_id=contract_id, start=start, end=start + len(text), text=text, chunk_index=chunk_index)


# ----- _sentence_spans golden offsets -----
@pytest.mark.parametrize(
    "text, expected",
    [
        (TEXT, [(0, 31), (32, 77), (78, 100), (101, 120)]),
        ("  Alpha beta. Gamma; delta.  ", [(2, 13), (14, 20), (21, 27)]),  # outer whitespace trimmed
        ("No break here", [(0, 13)]),
        ("   ", []),
        ("", []),
    ],
)
def test_sentence_spans_golden_offsets(text, expected):
    """Sentence offsets match hand-computed values and exclude leading/trailing whitespace."""
    assert list(_sentence_spans(text)) == expected


# ----- extract_evidence -----
def test_extract_evidence_picks_highest_scoring_sentence():
    """Governing Law picks sentence 2, which contains both cues."""
    ev = extract_evidence(TEXT, "Governing Law", CUES)
    assert (ev.quote, ev.start, ev.end) == ("The State of Delaware governs this Agreement.", 32, 77)


def test_extract_evidence_offsets_are_contract_absolute():
    """chunk_start shifts offsets from chunk-relative to contract-absolute."""
    ev = extract_evidence(TEXT, "Insurance", CUES, chunk_start=1000)
    assert (ev.quote, ev.start, ev.end) == ("Insurance is required;", 1078, 1100)


def test_extract_evidence_tie_goes_to_earliest_sentence():
    """Sentences 1 and 2 both contain 'agreement'; the first one wins."""
    ev = extract_evidence(TEXT, "X", {"x": ["agreement"]})
    assert (ev.start, ev.end) == (0, 31)


def test_extract_evidence_category_name_is_normalized():
    """Category lookups ignore case and punctuation."""
    assert extract_evidence(TEXT, "GOVERNING-law", CUES).start == 32


@pytest.mark.parametrize(
    "category, cues, min_score",
    [
        ("Parties", CUES, 1),                  # category has no cue entry
        ("Empty", {"empty": []}, 1),           # category has an empty cue list
        ("Governing Law", {"governinglaw": ["arbitration"]}, 1),  # no sentence matches
        ("Governing Law", CUES, 3),            # best score (2) below min_score
    ],
)
def test_extract_evidence_returns_none_when_unsupported(category, cues, min_score):
    """No cues, no matching sentence, or a score under min_score yields None."""
    assert extract_evidence(TEXT, category, cues, min_score=min_score) is None


# ----- extract_for_predictions routing -----
def test_extract_for_predictions_routes_grounded_and_unknown():
    """A cue match is grounded with absolute offsets; a category without cues goes to Unknown."""
    contract_text = "Header. " + TEXT
    chunk = make_chunk(start=8, chunk_index=3)
    grounded, unknown = extract_for_predictions(contract_text, [(chunk, "Governing Law"), (chunk, "Parties")], CUES)

    assert grounded == [
        GroundedEvidence(
            contract_id="c1", chunk_index=3, category="Governing Law",
            evidence={"quote": "The State of Delaware governs this Agreement.", "start": 40, "end": 85},
        )
    ]
    assert unknown == [Unknown(contract_id="c1", chunk_index=3, category="Parties", reason="no_cue_match")]


def test_extract_for_predictions_wrong_offsets_go_to_unknown():
    """A chunk whose start doesn't line up with the contract text yields span_mismatch, not evidence."""
    contract_text = "Header. " + TEXT
    chunk = make_chunk(start=0)  # really sits at offset 8
    grounded, unknown = extract_for_predictions(contract_text, [(chunk, "Governing Law")], CUES)
    assert grounded == []
    assert [u.reason for u in unknown] == ["span_mismatch"]


def test_extract_for_predictions_empty_input_returns_empty_lists():
    """No predictions means nothing grounded and nothing unknown."""
    assert extract_for_predictions(TEXT, [], CUES) == ([], [])


# ----- Real CUAD data: every grounded quote must slice back out of the contract -----
@pytest.fixture(scope="module")
def cuad_contracts():
    """Load the first 5 CUAD training contracts, skipping if the data files are missing."""
    if not CUAD_TRAIN_PATH.exists() or not CATEGORY_DESCRIPTIONS_PATH.exists():
        pytest.skip("CUAD training file or category CSV not found")
    return load_cuad(CUAD_TRAIN_PATH)[:5]


def test_real_cuad_grounded_evidence_matches_contract_text(cuad_contracts):
    """Gold categories on 1500/200 chunks never mismatch, and every quote equals text[start:end] inside its chunk."""
    cues = load_category_cues()
    total_grounded = 0
    for contract in cuad_contracts:
        chunks = chunk_contract(contract.contract_id, contract.text, 1500, 200)
        predictions = [
            (chunk, span.category)
            for chunk in chunks
            for span in contract.spans
            if span.start < chunk.end and span.end > chunk.start
        ]
        grounded, unknown = extract_for_predictions(contract.text, predictions, cues)

        assert len(grounded) + len(unknown) == len(predictions)
        assert all(u.reason == "no_cue_match" for u in unknown)
        by_index = {c.chunk_index: c for c in chunks}
        for g in grounded:
            ev, chunk = g.evidence, by_index[g.chunk_index]
            assert contract.text[ev.start:ev.end] == ev.quote
            assert chunk.start <= ev.start < ev.end <= chunk.end
        total_grounded += len(grounded)
    assert total_grounded > 0
