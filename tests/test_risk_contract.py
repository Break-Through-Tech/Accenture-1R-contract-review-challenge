import pytest

from src.risk.contract import _triage_score, aggregate_contracts, sensitivity
from src.risk.schema import ClauseInput, ClauseScore, WeightSet
from src.risk.scoring import rate

# Severity is the only weighted signal, so a clause's composite is just its category's severity
WEIGHTS = WeightSet(
    weight_version_id="test-v1",
    created_by="tests",
    signal_weights={"category_severity": 1.0, "one_sidedness": 0.0, "magnitude": 0.0, "contract_context": 0.0},
    medium_threshold=0.35,
    high_threshold=0.55,
    category_severity={"Uncapped Liability": 0.9, "Parties": 0.1},
    default_severity=0.5,
)


def _score(contract_id: str, category: str, composite: float, has_evidence: bool = True,
           weight_set: WeightSet = WEIGHTS) -> ClauseScore:
    """Build a ClauseScore with this composite, rated under weight_set."""
    return ClauseScore(
        contract_id=contract_id, chunk_index=0, category=category, probability=0.9,
        signals={"category_severity": composite, "one_sidedness": 0.0, "magnitude": 0.0, "contract_context": 0.0},
        composite_score=composite, rating=rate(composite, weight_set), has_evidence=has_evidence,
        weight_version_id=weight_set.weight_version_id, weight_set_sha256=weight_set.content_sha256,
    )


def _clause(contract_id: str, category: str) -> ClauseInput:
    """Build a ClauseInput with no evidence (severity doesn't read the quote)."""
    return ClauseInput(contract_id=contract_id, chunk_index=0, category=category, probability=0.9)


# ----- Triage score formula: 0.5 * max + 0.5 * mean of top 3 (padded with 0), the defaults -----
@pytest.mark.parametrize(
    "composites, expected",
    [
        ([], 0.0),                         # nothing detected
        ([0.6], 0.4),                      # (0.6 + 0.6/3) / 2
        ([0.5, 0.5], 0.416666667),         # (0.5 + 1.0/3) / 2
        ([0.9, 0.6, 0.3], 0.75),           # (0.9 + 1.8/3) / 2
        ([0.9, 0.6, 0.3, 0.2], 0.75),      # a 4th clause is ignored
        ([1.0, 1.0, 1.0], 1.0),            # stays in [0, 1]
    ],
)
def test_triage_score_golden_values(composites, expected):
    """The worst clause counts half and the mean of the top 3 counts half."""
    assert (WEIGHTS.triage_top_k, WEIGHTS.triage_max_weight) == (3, 0.5)
    assert _triage_score(composites, WEIGHTS) == pytest.approx(expected)


@pytest.mark.parametrize(
    "top_k, max_weight, expected",
    [
        (1, 0.5, 0.9),          # top 1 only: the worst clause alone
        (2, 0.5, 0.825),        # 0.45 + 0.5 * 1.5/2
        (3, 1.0, 0.9),          # all weight on the worst clause
        (3, 0.0, 0.6),          # all weight on the mean of the top 3
        (5, 0.5, 0.65),         # 0.45 + 0.5 * 2.0/5, two empty slots
    ],
)
def test_triage_score_reads_settings_from_the_weight_set(top_k, max_weight, expected):
    """triage_top_k and triage_max_weight come from the weight version, not the code."""
    ws = WEIGHTS.model_copy(update={"weight_version_id": "test-k", "triage_top_k": top_k, "triage_max_weight": max_weight})
    assert _triage_score([0.9, 0.6, 0.3, 0.2], ws) == pytest.approx(expected)


def test_aggregate_contracts_keeps_triage_top_k_clauses():
    """top_clauses holds the triage_top_k worst clauses."""
    ws = WEIGHTS.model_copy(update={"weight_version_id": "test-k1", "triage_top_k": 1})
    scores = [_score("c1", "Audit Rights", 0.4, weight_set=ws), _score("c1", "Uncapped Liability", 0.9, weight_set=ws)]
    [contract] = aggregate_contracts(scores, ws)
    assert [s.category for s in contract.top_clauses] == ["Uncapped Liability"]


def test_triage_score_breadth_breaks_ties_on_the_worst_clause():
    """Two contracts with the same worst clause: the one with more risky clauses ranks higher."""
    assert _triage_score([0.6, 0.5], WEIGHTS) > _triage_score([0.6], WEIGHTS)


# ----- Aggregation -----
def test_aggregate_contracts_keeps_worst_clause_per_category():
    """A category detected in several chunks counts once, at its highest composite."""
    scores = [_score("c1", "Audit Rights", 0.3), _score("c1", "Audit Rights", 0.6)]
    [contract] = aggregate_contracts(scores, WEIGHTS)
    assert contract.triage_score == pytest.approx(0.4)
    assert [s.composite_score for s in contract.top_clauses] == [0.6]
    assert (contract.high_count, contract.medium_count, contract.low_count) == (1, 0, 0)


def test_aggregate_contracts_counts_ratings_and_unknowns():
    """Counts are per distinct category; Unknown-queue clauses still count toward the score."""
    scores = [
        _score("c1", "Uncapped Liability", 0.9),
        _score("c1", "Audit Rights", 0.4, has_evidence=False),
        _score("c1", "Governing Law", 0.2),
        _score("c1", "Parties", 0.1),
    ]
    [contract] = aggregate_contracts(scores, WEIGHTS)
    assert (contract.high_count, contract.medium_count, contract.low_count, contract.unknown_count) == (1, 1, 2, 1)
    assert [s.category for s in contract.top_clauses] == ["Uncapped Liability", "Audit Rights", "Governing Law"]
    assert contract.triage_score == pytest.approx(0.7)  # (0.9 + 1.5/3) / 2
    assert (contract.weight_version_id, contract.weight_set_sha256) == ("test-v1", WEIGHTS.content_sha256)


def test_aggregate_contracts_ranks_riskiest_first_and_ties_share_a_rank():
    """Sorted by score, then contract_id; equal scores share a rank and the next rank skips."""
    scores = [_score("b", "Audit Rights", 0.6), _score("a", "Audit Rights", 0.6), _score("c", "Uncapped Liability", 0.9)]
    contracts = aggregate_contracts(scores, WEIGHTS, contract_ids=["d"])
    assert [(c.contract_id, c.rank) for c in contracts] == [("c", 1), ("a", 2), ("b", 2), ("d", 4)]


def test_aggregate_contracts_includes_contracts_with_nothing_detected():
    """A listed contract with no clauses scores 0 instead of disappearing from the ranking."""
    [contract] = aggregate_contracts([], WEIGHTS, contract_ids=["empty"])
    assert (contract.contract_id, contract.triage_score, contract.rank, contract.top_clauses) == ("empty", 0.0, 1, [])


def test_aggregate_contracts_rejects_scores_from_another_weight_version():
    """Mixing weight versions would make the ranking unreproducible, so it raises."""
    other = WEIGHTS.model_copy(update={"weight_version_id": "test-v2", "default_severity": 0.4})
    with pytest.raises(ValueError):
        aggregate_contracts([_score("c1", "Audit Rights", 0.5, weight_set=other)], WEIGHTS)


# ----- Sensitivity analysis -----
SWAPPED = WEIGHTS.model_copy(update={
    "weight_version_id": "test-v2", "category_severity": {"Uncapped Liability": 0.1, "Parties": 0.9},
})
CLAUSES = [_clause("c1", "Uncapped Liability"), _clause("c2", "Parties")]


def test_sensitivity_ranks_each_contract_under_each_weight_set():
    """Swapping two severities swaps the two contracts; an empty contract stays last."""
    ranks = sensitivity(CLAUSES, [WEIGHTS, SWAPPED], contract_ids=["c3"])
    assert ranks.to_dict() == {"test-v1": {"c1": 1, "c2": 2, "c3": 3}, "test-v2": {"c1": 2, "c2": 1, "c3": 3}}
    # Spearman on ranks (1,2,3) vs (2,1,3): 1 - 6*2 / (3*8) = 0.5
    assert ranks.corr(method="spearman").loc["test-v1", "test-v2"] == pytest.approx(0.5)


def test_sensitivity_threshold_change_keeps_the_ranking():
    """Moving rating thresholds changes clause buckets, not composites, so contract ranks hold."""
    stricter = WEIGHTS.model_copy(update={"weight_version_id": "test-v3", "high_threshold": 0.95})
    ranks = sensitivity(CLAUSES, [WEIGHTS, stricter])
    assert ranks["test-v1"].tolist() == ranks["test-v3"].tolist()


def test_sensitivity_rejects_duplicate_versions():
    """Two weight sets with one id would overwrite each other's column."""
    with pytest.raises(ValueError):
        sensitivity(CLAUSES, [WEIGHTS, WEIGHTS])
