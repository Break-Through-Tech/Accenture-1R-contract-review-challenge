"""
Each risk signal is one small pure function that turns a detected clause into a float in [0, 1].
scoring.py combines them into a composite score using the WeightSet's signal_weights.

PLACEHOLDER: these four signals are placeholders until they are settled with the advisor.
    category_severity: how risky the clause type is on its own (lookup table in the WeightSet)
    detection_confidence: how sure the classifier is that the clause is there (its probability)
    risk_language: whether the evidence quote contains risky wording (0 if there is no evidence)
    cooccurrence: whether other clauses detected in the same contract raise or lower this one's risk

Every signal function has the same signature, (clause, contract_categories, weight_set) -> float,
so scoring.py can loop over the SIGNALS registry without knowing which signals exist.

How to add, remove or rename a signal:
1. Write (or delete / rename) the function here and update the SIGNALS registry
2. Update SignalName in schema.py to match (WeightSet then requires a weight for it)
3. Update signal_weights in every weight set JSON you still use, as a new weight version
4. Add tests for the new function in tests/test_risk_signals.py

Inputs from other modules:
'ClauseInput.evidence' is the Evidence from src/evidence/ (None for Unknown-queue clauses)
'contract_categories' is every category detected in the clause's contract (built by scoring.py)
"""
from collections.abc import Callable, Set

from src.evidence.cues import normalize_category
from src.risk.schema import SIGNAL_NAMES, ClauseInput, WeightSet

# Neutral starting point for the cooccurrence signal before any rule applies
NEUTRAL_COOCCURRENCE = 0.5


def _clip(value: float) -> float:
    """Clamp a value into [0, 1]."""
    return min(max(value, 0.0), 1.0)


def category_severity(clause: ClauseInput, contract_categories: Set[str], weight_set: WeightSet) -> float:
    """Look up the category's severity (names normalized like cues.normalize_category), else default_severity."""
    severity = {normalize_category(name): value for name, value in weight_set.category_severity.items()}
    return severity.get(normalize_category(clause.category), weight_set.default_severity)


def detection_confidence(clause: ClauseInput, contract_categories: Set[str], weight_set: WeightSet) -> float:
    """Return the classifier probability for this (chunk, category), clipped to [0, 1]."""
    return _clip(clause.probability)


def risk_language(clause: ClauseInput, contract_categories: Set[str], weight_set: WeightSet) -> float:
    """
    Return the fraction of risk_phrases found in the evidence quote (case-insensitive), capped at 1.0.
    Returns 0.0 when the clause has no evidence or the weight set has no phrases.
    """
    # shortcut: plain substring match ("unlimited" also hits "unlimitedly"), switch to the cue-style word regex if that matters
    if clause.evidence is None or not weight_set.risk_phrases:
        return 0.0
    quote = clause.evidence.quote.lower()
    hits = sum(phrase.lower() in quote for phrase in weight_set.risk_phrases)
    return min(hits / len(weight_set.risk_phrases), 1.0)


def cooccurrence(clause: ClauseInput, contract_categories: Set[str], weight_set: WeightSet) -> float:
    """
    Start at a neutral 0.5 and add the adjustment of every rule whose target is this clause's category
    and whose other_category was detected in the same contract, then clip to [0, 1].
    """
    target = normalize_category(clause.category)
    detected = {normalize_category(name) for name in contract_categories}
    score = NEUTRAL_COOCCURRENCE + sum(
        rule.adjustment
        for rule in weight_set.cooccurrence_rules
        if normalize_category(rule.target_category) == target
        and normalize_category(rule.other_category) in detected
    )
    return _clip(score)


SignalFn = Callable[[ClauseInput, Set[str], WeightSet], float]

# Signal name -> function; scoring.py computes every signal listed here
SIGNALS: dict[str, SignalFn] = {
    "category_severity": category_severity,
    "detection_confidence": detection_confidence,
    "risk_language": risk_language,
    "cooccurrence": cooccurrence,
}

# Fail at import time if the registry and SignalName drift apart
assert set(SIGNALS) == set(SIGNAL_NAMES), f"SIGNALS {sorted(SIGNALS)} != SignalName {sorted(SIGNAL_NAMES)}"
