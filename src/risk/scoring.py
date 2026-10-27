"""
Turns detected clauses into Low / Medium / High risk ratings using a versioned WeightSet.

How everything comes together:
rate(): maps a composite score to a rating using the WeightSet's thresholds
score_clause(): computes every signal in signals.SIGNALS for one clause, combines them as a
                weighted sum, rates it, and stamps the result with the weight_version_id
score_contract_clauses(): entry point other pipeline stages call. Works out which categories
                          were detected in each contract (for the cooccurrence signal) and
                          scores every clause

Everything here is pure: no I/O, no randomness, no global state. Scoring only reads cached
probabilities and Evidence, so rescoring under a new weight version never reruns the model.

Inputs from other modules:
'ClauseInput' rows built from cached classifier probabilities plus the GroundedEvidence /
Unknown output of evidence.extractor.extract_for_predictions (evidence=None for Unknowns)
'WeightSet' from schema.load_weight_set()

PLACEHOLDER: the signals and the default weights are placeholders pending advisor sign-off.
Clauses with no evidence are still scored (risk_language is 0 for them) and flagged with
has_evidence=False; whether they should count toward the contract score is an open question.
"""
from collections import defaultdict
from collections.abc import Iterable, Set

from src.risk.schema import ClauseInput, ClauseScore, Rating, WeightSet
from src.risk.signals import SIGNALS


def rate(composite_score: float, weight_set: WeightSet) -> Rating:
    """Return High if score >= high_threshold, Medium if >= medium_threshold, else Low."""
    if composite_score >= weight_set.high_threshold:
        return "High"
    if composite_score >= weight_set.medium_threshold:
        return "Medium"
    return "Low"


def score_clause(clause: ClauseInput, contract_categories: Set[str], weight_set: WeightSet) -> ClauseScore:
    """
    Score one clause.

    clause: the detected clause (probability + optional Evidence)
    contract_categories: every category detected in the same contract
    weight_set: the weights, thresholds and lookup tables to score with
    """
    signals = {name: fn(clause, contract_categories, weight_set) for name, fn in SIGNALS.items()}

    # Weighted sum; weights sum to 1 so this stays in [0, 1] up to float error, which the clip removes
    composite = sum(weight_set.signal_weights[name] * value for name, value in signals.items())
    composite = min(max(composite, 0.0), 1.0)

    return ClauseScore(
        contract_id=clause.contract_id,
        chunk_index=clause.chunk_index,
        category=clause.category,
        signals=signals,
        composite_score=composite,
        rating=rate(composite, weight_set),
        has_evidence=clause.evidence is not None,
        weight_version_id=weight_set.weight_version_id,
    )


def score_contract_clauses(clauses: Iterable[ClauseInput], weight_set: WeightSet) -> list[ClauseScore]:
    """
    Pipeline entry point: score every clause, across one or many contracts, in input order.
    Each contract's detected-category set is built from the clauses themselves.
    """
    clauses = list(clauses)
    categories_by_contract: dict[str, set[str]] = defaultdict(set)
    for clause in clauses:
        categories_by_contract[clause.contract_id].add(clause.category)
    return [score_clause(c, categories_by_contract[c.contract_id], weight_set) for c in clauses]
