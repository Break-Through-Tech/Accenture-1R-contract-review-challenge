import pytest

from src.evidence.schema import Evidence
from src.risk.schema import ClauseInput, WeightSet
from src.risk.scoring import rate, score_clause, score_contract_clauses

WEIGHTS = WeightSet(
    weight_version_id="test-v1",
    created_by="tests",
    signal_weights={"category_severity": 0.4, "detection_confidence": 0.3, "risk_language": 0.2, "cooccurrence": 0.1},
    medium_threshold=0.4,
    high_threshold=0.7,
    category_severity={"Uncapped Liability": 0.9, "Parties": 0.1},
    default_severity=0.5,
    cooccurrence_rules=[
        {"target_category": "Uncapped Liability", "other_category": "Cap On Liability", "adjustment": -0.3},
    ],
    risk_phrases=["unlimited", "sole discretion", "irrevocable", "without cause"],
)

QUOTE = "Liability is unlimited and irrevocable."


def _clause(
    category: str = "Uncapped Liability",
    probability: float = 0.8,
    quote: str | None = QUOTE,
    contract_id: str = "c1",
    chunk_index: int = 0,
) -> ClauseInput:
    """Build a ClauseInput, with an Evidence for quote when given."""
    ev = Evidence(quote=quote, start=0, end=len(quote)) if quote else None
    return ClauseInput(
        contract_id=contract_id, chunk_index=chunk_index, category=category, probability=probability, evidence=ev
    )


# ----- Rating boundaries (medium 0.4, high 0.7) -----
@pytest.mark.parametrize(
    "score, expected",
    [
        (0.0, "Low"),
        (0.39, "Low"),
        (0.4, "Medium"),
        (0.41, "Medium"),
        (0.69, "Medium"),
        (0.7, "High"),
        (0.71, "High"),
        (1.0, "High"),
    ],
)
def test_rate_threshold_boundaries(score, expected):
    """Scores at a threshold take the higher rating; just below take the lower one."""
    assert rate(score, WEIGHTS) == expected


# ----- Golden clause score -----
def test_score_clause_golden_values():
    """Hand-computed: 0.4*0.9 + 0.3*0.8 + 0.2*(2/4) + 0.1*(0.5-0.3) = 0.72 -> High."""
    score = score_clause(_clause(), {"Uncapped Liability", "Cap On Liability"}, WEIGHTS)
    assert score.signals == pytest.approx(
        {"category_severity": 0.9, "detection_confidence": 0.8, "risk_language": 0.5, "cooccurrence": 0.2}
    )
    assert score.composite_score == pytest.approx(0.72)
    assert (score.rating, score.has_evidence) == ("High", True)
    assert (score.contract_id, score.chunk_index, score.category) == ("c1", 0, "Uncapped Liability")


def test_score_clause_without_evidence_still_scored():
    """Unknown-queue clauses get risk_language 0 and has_evidence False: 0.36 + 0.24 + 0 + 0.05 = 0.65."""
    score = score_clause(_clause(quote=None), {"Uncapped Liability"}, WEIGHTS)
    assert score.signals["risk_language"] == 0.0
    assert score.composite_score == pytest.approx(0.65)
    assert (score.rating, score.has_evidence) == ("Medium", False)


def test_score_clause_composite_stays_in_unit_range():
    """All-max signals with weights summing to 1 only up to float error never exceed 1."""
    ws = WeightSet(
        weight_version_id="max",
        created_by="tests",
        signal_weights={"category_severity": 0.1, "detection_confidence": 0.2, "risk_language": 0.3, "cooccurrence": 0.4},
        medium_threshold=0.4,
        high_threshold=0.7,
        default_severity=1.0,
        cooccurrence_rules=[
            {"target_category": "Uncapped Liability", "other_category": "Cap On Liability", "adjustment": 1.0}
        ],
        risk_phrases=["unlimited"],
    )
    score = score_clause(_clause(probability=1.0), {"Cap On Liability"}, ws)
    assert score.composite_score == pytest.approx(1.0)
    assert score.composite_score <= 1.0


# ----- Contract-level scoring -----
def test_score_contract_clauses_uses_per_contract_categories():
    """Cap On Liability in c1 lowers c1's Uncapped Liability, but not c2's."""
    clauses = [
        _clause(contract_id="c1"),
        _clause("Cap On Liability", contract_id="c1", chunk_index=1),
        _clause(contract_id="c2"),
    ]
    scores = score_contract_clauses(clauses, WEIGHTS)
    assert [s.signals["cooccurrence"] for s in scores] == pytest.approx([0.2, 0.5, 0.5])
    assert [(s.contract_id, s.chunk_index) for s in scores] == [("c1", 0), ("c1", 1), ("c2", 0)]


def test_score_contract_clauses_empty_returns_empty():
    """No clauses means no scores."""
    assert score_contract_clauses([], WEIGHTS) == []


def test_score_contract_clauses_accepts_generator():
    """A one-shot iterable is consumed once and still gets every clause's contract context."""
    clauses = (c for c in [_clause(), _clause("Cap On Liability", chunk_index=1)])
    assert [s.signals["cooccurrence"] for s in score_contract_clauses(clauses, WEIGHTS)] == pytest.approx([0.2, 0.5])


# ----- Determinism and versioning -----
def test_score_contract_clauses_deterministic():
    """Scoring the same input twice gives identical output."""
    clauses = [_clause(), _clause("Parties", 0.3, None, chunk_index=1), _clause("Cap On Liability", chunk_index=2)]
    assert score_contract_clauses(clauses, WEIGHTS) == score_contract_clauses(clauses, WEIGHTS)


def test_new_weight_set_changes_rating_only_via_weights():
    """Same clause, same contract: raising high_threshold above 0.72 turns High into Medium."""
    v2 = WEIGHTS.model_copy(update={"weight_version_id": "test-v2", "high_threshold": 0.8})
    contract = {"Uncapped Liability", "Cap On Liability"}
    assert score_clause(_clause(), contract, WEIGHTS).rating == "High"
    assert score_clause(_clause(), contract, v2).rating == "Medium"


@pytest.mark.parametrize("version", ["test-v1", "advisor-2026-11-01"])
def test_every_score_records_weight_version_id(version):
    """Every ClauseScore carries the weight_version_id it was scored with."""
    ws = WEIGHTS.model_copy(update={"weight_version_id": version})
    clauses = [_clause(), _clause("Parties", chunk_index=1), _clause(contract_id="c2", quote=None)]
    assert {s.weight_version_id for s in score_contract_clauses(clauses, ws)} == {version}
