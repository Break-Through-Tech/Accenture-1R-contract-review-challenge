import numpy as np
import pytest

from src.chunking.chunk import Chunk
from src.evidence.cues import normalize_category
from src.labels.categories import CATEGORIES
from src.risk.inputs import clauses_from_probabilities, load_cached_probabilities
from src.risk.schema import WeightSet
from src.risk.scoring import score_contract_clauses

TEXT = "Licensee liability is uncapped. Either party may audit records. Nothing else here."
CUES = {normalize_category("Uncapped Liability"): ["liability"], normalize_category("Audit Rights"): ["audit"]}
THRESHOLDS = {category: 0.5 for category in CATEGORIES}

WEIGHTS = WeightSet(
    weight_version_id="test-v1",
    created_by="tests",
    signal_weights={"category_severity": 0.4, "one_sidedness": 0.2, "magnitude": 0.2, "contract_context": 0.2},
    medium_threshold=0.3,
    high_threshold=0.7,
    category_severity={"Uncapped Liability": 0.9},
    default_severity=0.1,
    cooccurrence_rules=[
        {"target_category": "Uncapped Liability", "other_category": "Insurance", "when": "absent", "adjustment": 0.5},
    ],
)


def _chunk(text: str = TEXT, start: int = 0, index: int = 0, contract_id: str = "c1") -> Chunk:
    """Build a Chunk for the text at this offset."""
    return Chunk(contract_id=contract_id, chunk_index=index, start=start, end=start + len(text), text=text)


def _matrix(rows: int, values: dict[tuple[int, str], float]) -> np.ndarray:
    """Build a probability matrix that is 0 except for the given (row, category) values."""
    matrix = np.zeros((rows, len(CATEGORIES)))
    for (row, category), value in values.items():
        matrix[row, CATEGORIES.index(category)] = value
    return matrix


def _clauses(matrix: np.ndarray, chunks: list[Chunk] | None = None, thresholds: dict = THRESHOLDS, cues: dict = CUES):
    """Run clauses_from_probabilities on contract c1 with this test file's defaults."""
    return clauses_from_probabilities({"c1": TEXT}, chunks or [_chunk()], matrix, thresholds, cues)


# ----- Thresholding -----
@pytest.mark.parametrize(
    "probability, detected",
    [(0.49, False), (0.5, True), (0.51, True)],
)
def test_clauses_from_probabilities_applies_threshold(probability, detected):
    """A probability at or above the threshold is detected; one below it is not."""
    clauses = _clauses(_matrix(1, {(0, "Uncapped Liability"): probability}))
    assert [c.category for c in clauses] == (["Uncapped Liability"] if detected else [])


def test_clauses_from_probabilities_uses_each_categorys_threshold():
    """The same probability passes one category's threshold and fails another's."""
    thresholds = THRESHOLDS | {"Audit Rights": 0.9}
    matrix = _matrix(1, {(0, "Uncapped Liability"): 0.6, (0, "Audit Rights"): 0.6})
    assert [c.category for c in _clauses(matrix, thresholds=thresholds)] == ["Uncapped Liability"]


def test_clauses_from_probabilities_missing_threshold_raises():
    """A thresholds table without every category fails loudly rather than skipping one."""
    with pytest.raises(KeyError):
        _clauses(_matrix(1, {}), thresholds={"Parties": 0.5})


# ----- Row contents -----
def test_clauses_from_probabilities_carries_probability_and_evidence():
    """Each row keeps its cached probability and a quote grounded at absolute contract offsets."""
    start = TEXT.index("Either")
    chunk = _chunk(TEXT[start:], start=start, index=3)
    [clause] = _clauses(_matrix(1, {(0, "Audit Rights"): 0.8}), [chunk])
    assert (clause.contract_id, clause.chunk_index, clause.category) == ("c1", 3, "Audit Rights")
    assert clause.probability == pytest.approx(0.8)
    assert clause.evidence.quote == "Either party may audit records."
    assert TEXT[clause.evidence.start:clause.evidence.end] == clause.evidence.quote


def test_clauses_from_probabilities_no_cue_match_has_no_evidence():
    """A detection that can't be grounded stays as a clause with evidence=None (the Unknown queue)."""
    [clause] = _clauses(_matrix(1, {(0, "Parties"): 0.8}))
    assert (clause.category, clause.evidence) == ("Parties", None)


def test_clauses_from_probabilities_overlapping_chunks_keep_one_row():
    """Two overlapping chunks quoting the same sentence produce one row, from the first chunk."""
    chunks = [_chunk(), _chunk(TEXT[:31], index=1)]
    matrix = _matrix(2, {(0, "Uncapped Liability"): 0.8, (1, "Uncapped Liability"): 0.9})
    clauses = _clauses(matrix, chunks)
    assert [(c.chunk_index, c.probability) for c in clauses] == [(0, pytest.approx(0.8))]


def test_clauses_from_probabilities_matrix_can_span_contracts():
    """Rows from different contracts are grounded against their own contract's text."""
    other = "Audit may happen."
    chunks = [_chunk(), _chunk(other, contract_id="c2")]
    matrix = _matrix(2, {(0, "Uncapped Liability"): 0.8, (1, "Audit Rights"): 0.7})
    clauses = clauses_from_probabilities({"c1": TEXT, "c2": other}, chunks, matrix, THRESHOLDS, CUES)
    assert [(c.contract_id, c.category, c.evidence.quote) for c in clauses] == [
        ("c1", "Uncapped Liability", "Licensee liability is uncapped."),
        ("c2", "Audit Rights", "Audit may happen."),
    ]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_clauses_from_probabilities_returns_plain_floats(dtype):
    """float32 model output and float64 both produce ordinary Python floats on the row."""
    [clause] = _clauses(_matrix(1, {(0, "Audit Rights"): 0.75}).astype(dtype))
    assert type(clause.probability) is float
    assert clause.probability == 0.75


def test_clauses_from_probabilities_no_chunks_returns_empty():
    """An empty matrix means no clauses."""
    assert clauses_from_probabilities({}, [], np.zeros((0, len(CATEGORIES))), THRESHOLDS, CUES) == []


# ----- Bad matrices -----
@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((2, len(CATEGORIES))),        # more rows than chunks
        np.zeros((1, len(CATEGORIES) - 1)),    # a category column is missing
        np.zeros(len(CATEGORIES)),             # one-dimensional
        _matrix(1, {(0, "Parties"): 1.2}),     # above 1, e.g. raw logits
        _matrix(1, {(0, "Parties"): -0.1}),    # below 0
        _matrix(1, {(0, "Parties"): np.nan}),  # would otherwise silently vanish
    ],
)
def test_clauses_from_probabilities_rejects_bad_matrix(matrix):
    """A matrix that doesn't line up with the chunks or isn't probabilities raises ValueError."""
    with pytest.raises(ValueError):
        _clauses(matrix)


# ----- Reading the cache -----
def test_load_cached_probabilities_round_trips(tmp_path):
    """A saved matrix comes back unchanged."""
    matrix = _matrix(1, {(0, "Audit Rights"): 0.75})
    np.save(tmp_path / "probs.npy", matrix)
    assert np.array_equal(load_cached_probabilities(tmp_path / "probs.npy"), matrix)


def test_load_cached_probabilities_refuses_pickles(tmp_path):
    """A pickled object array is refused, since loading one could run arbitrary code."""
    np.save(tmp_path / "probs.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        load_cached_probabilities(tmp_path / "probs.npy")


def test_cached_probabilities_rescore_under_new_weights(tmp_path):
    """One saved cache scores under two weight versions: the rating and version change, no model involved."""
    np.save(tmp_path / "probs.npy", _matrix(1, {(0, "Uncapped Liability"): 0.8}))
    clauses = _clauses(load_cached_probabilities(tmp_path / "probs.npy"))
    # 0.4*0.9 + 0.2*0.5 = 0.46
    [before] = score_contract_clauses(clauses, WEIGHTS)
    [after] = score_contract_clauses(clauses, WEIGHTS.model_copy(update={"weight_version_id": "test-v2", "high_threshold": 0.4}))
    assert before.composite_score == after.composite_score == pytest.approx(0.46)
    assert (before.rating, before.weight_version_id) == ("Medium", "test-v1")
    assert (after.rating, after.weight_version_id) == ("High", "test-v2")
    assert before.probability == pytest.approx(0.8)


def test_below_threshold_category_counts_as_absent():
    """Insurance below its threshold isn't detected, so the 'Insurance absent' rule still fires."""
    matrix = _matrix(1, {(0, "Uncapped Liability"): 0.8, (0, "Insurance"): 0.4})
    [score] = score_contract_clauses(_clauses(matrix), WEIGHTS)
    assert score.signals["contract_context"] == pytest.approx(0.5)
