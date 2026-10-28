import pytest

from src.evidence.schema import Evidence
from src.risk.schema import ClauseInput, WeightSet, load_weight_set
from src.risk.scoring import rate, score_clause, score_contract_clauses

WEIGHTS = WeightSet(
    weight_version_id="test-v1",
    created_by="tests",
    signal_weights={"category_severity": 0.4, "one_sidedness": 0.2, "magnitude": 0.2, "contract_context": 0.2},
    medium_threshold=0.4,
    high_threshold=0.7,
    category_severity={"Uncapped Liability": 0.9, "Parties": 0.1},
    default_severity=0.5,
    cooccurrence_rules=[
        {"target_category": "Uncapped Liability", "other_category": "Cap On Liability", "adjustment": -0.3},
        {"target_category": "Uncapped Liability", "other_category": "Insurance", "when": "absent", "adjustment": 0.5},
    ],
    aggravating_phrases=["sole discretion", "without notice"],
    mitigating_phrases=["each party"],
    scope_phrases=["perpetual"],
)

QUOTE = "Licensor may, at its sole discretion and without notice, grant a perpetual waiver."


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


# ----- Golden clause scores -----
def test_score_clause_golden_values():
    """Hand-computed, no Insurance: 0.4*0.9 + 0.2*1 + 0.2*1 + 0.2*0.5 = 0.86 -> High."""
    score = score_clause(_clause(), {"Uncapped Liability"}, WEIGHTS)
    assert score.signals == pytest.approx(
        {"category_severity": 0.9, "one_sidedness": 1.0, "magnitude": 1.0, "contract_context": 0.5}
    )
    assert score.composite_score == pytest.approx(0.86)
    assert (score.rating, score.has_evidence, score.probability) == ("High", True, 0.8)
    assert (score.contract_id, score.chunk_index, score.category) == ("c1", 0, "Uncapped Liability")


def test_score_clause_protections_lower_rating():
    """Same clause with Cap On Liability and Insurance: 0.4*0.63 + 0.2 + 0.2 + 0 = 0.652 -> Medium."""
    score = score_clause(_clause(), {"Uncapped Liability", "Cap On Liability", "Insurance"}, WEIGHTS)
    assert score.composite_score == pytest.approx(0.652)
    assert score.rating == "Medium"


def test_score_clause_plain_low_severity_clause_is_low():
    """A plain Parties clause scores on severity alone: 0.4*0.1 = 0.04 -> Low (no constant floor)."""
    score = score_clause(_clause("Parties", quote="Acme Corp and Beta LLC."), {"Parties"}, WEIGHTS)
    assert score.composite_score == pytest.approx(0.04)
    assert score.rating == "Low"


def test_score_clause_without_evidence_still_scored():
    """Unknown-queue clauses get 0 for quote-based signals and has_evidence False: 0.36 + 0.1 = 0.46."""
    score = score_clause(_clause(quote=None), {"Uncapped Liability"}, WEIGHTS)
    assert (score.signals["one_sidedness"], score.signals["magnitude"]) == (0.0, 0.0)
    assert score.composite_score == pytest.approx(0.46)
    assert (score.rating, score.has_evidence) == ("Medium", False)


def test_score_clause_probability_does_not_change_score():
    """Detection confidence is carried for display but is not a risk signal."""
    low, high = (score_clause(_clause(probability=p), {"Uncapped Liability"}, WEIGHTS) for p in (0.3, 0.99))
    assert low.composite_score == high.composite_score
    assert (low.probability, high.probability) == (0.3, 0.99)


def test_score_clause_composite_stays_in_unit_range():
    """All-max signals with weights summing to 1 only up to float error never exceed 1."""
    ws = WEIGHTS.model_copy(update={
        "signal_weights": {"category_severity": 0.1, "one_sidedness": 0.2, "magnitude": 0.3, "contract_context": 0.4},
        "category_severity": {"Uncapped Liability": 1.0},
    })
    ws = WeightSet.model_validate(ws.model_dump() | {"cooccurrence_rules": [
        {"target_category": "Uncapped Liability", "other_category": "Insurance", "when": "absent", "adjustment": 1.0}
    ]})
    score = score_clause(_clause(), {"Uncapped Liability"}, ws)
    assert score.composite_score == pytest.approx(1.0)
    assert score.composite_score <= 1.0


def test_default_weights_spread_ratings():
    """The shipped v1-placeholder weights can produce all three ratings (v0 rated everything Medium)."""
    ws = load_weight_set()
    contract = {"Parties", "Non-Compete", "Uncapped Liability"}
    ratings = [
        score_clause(_clause("Parties", quote="Acme Corp and Beta LLC."), contract, ws).rating,
        score_clause(_clause("Non-Compete", quote="Shall not compete for two (2) years."), contract, ws).rating,
        score_clause(_clause(), contract, ws).rating,
    ]
    assert ratings == ["Low", "Medium", "High"]


# ----- Contract-level scoring -----
def test_score_contract_clauses_uses_per_contract_categories():
    """Cap On Liability in c1 lowers c1's Uncapped Liability severity, but not c2's."""
    clauses = [
        _clause(contract_id="c1"),
        _clause("Cap On Liability", contract_id="c1", chunk_index=1),
        _clause(contract_id="c2"),
    ]
    scores = score_contract_clauses(clauses, WEIGHTS)
    assert [s.signals["category_severity"] for s in scores] == pytest.approx([0.63, 0.5, 0.9])
    assert [(s.contract_id, s.chunk_index) for s in scores] == [("c1", 0), ("c1", 1), ("c2", 0)]


def test_score_contract_clauses_empty_returns_empty():
    """No clauses means no scores."""
    assert score_contract_clauses([], WEIGHTS) == []


def test_score_contract_clauses_accepts_generator():
    """A one-shot iterable is consumed once and still gets every clause's contract context."""
    clauses = (c for c in [_clause(), _clause("Insurance", chunk_index=1)])
    assert [s.signals["contract_context"] for s in score_contract_clauses(clauses, WEIGHTS)] == [0.0, 0.0]


# ----- Determinism and versioning -----
def test_score_contract_clauses_deterministic():
    """Scoring the same input twice gives identical output."""
    clauses = [_clause(), _clause("Parties", 0.3, None, chunk_index=1), _clause("Cap On Liability", chunk_index=2)]
    assert score_contract_clauses(clauses, WEIGHTS) == score_contract_clauses(clauses, WEIGHTS)


def test_new_weight_set_changes_rating_only_via_weights():
    """Same clause, same contract: raising high_threshold above 0.86 turns High into Medium."""
    v2 = WEIGHTS.model_copy(update={"weight_version_id": "test-v2", "high_threshold": 0.9})
    contract = {"Uncapped Liability"}
    assert score_clause(_clause(), contract, WEIGHTS).rating == "High"
    assert score_clause(_clause(), contract, v2).rating == "Medium"


@pytest.mark.parametrize("version", ["test-v1", "advisor-2026-11-01"])
def test_every_score_records_weight_version_id(version):
    """Every ClauseScore carries the weight_version_id it was scored with."""
    ws = WEIGHTS.model_copy(update={"weight_version_id": version})
    clauses = [_clause(), _clause("Parties", chunk_index=1), _clause(contract_id="c2", quote=None)]
    assert {s.weight_version_id for s in score_contract_clauses(clauses, ws)} == {version}
