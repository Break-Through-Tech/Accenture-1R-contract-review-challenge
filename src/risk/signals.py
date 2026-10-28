"""
Each risk signal is one small pure function that turns a detected clause into a float in [0, 1].
scoring.py combines them into a composite score using the WeightSet's signal_weights.

PLACEHOLDER: these four signals are proposals until they are settled with the advisor.
    category_severity: how bad this clause type is on its own (advisor lookup table), lowered
                       when a protective clause elsewhere in the contract softens it
    one_sidedness: whether the quote hands one party unchecked power ("sole discretion",
                   "without notice") rather than mutual, qualified terms ("each party", "reasonable")
    magnitude: how big the obligation is: broad-scope words ("perpetual", "worldwide") or a
               duration past the category's cutoffs (e.g. a 3-year non-compete)
    contract_context: whether the rest of the contract makes this clause worse, through risky
                      combinations or missing protections (e.g. Uncapped Liability with no Insurance)

Apart from category_severity, every signal is 0 when nothing risky is found, so a plain clause
scores on its severity alone and ratings spread across Low / Medium / High. one_sidedness and
magnitude read the evidence quote, so they are 0 for Unknown-queue clauses (evidence=None).

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
import re
from collections.abc import Callable, Iterable, Set
from functools import lru_cache

from src.evidence.cues import normalize_category
from src.risk.schema import SIGNAL_NAMES, ClauseInput, CooccurrenceRule, WeightSet

_DAYS_PER_UNIT = {"day": 1, "week": 7, "month": 30, "year": 365}
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18,
    "twenty": 20, "twenty-four": 24, "thirty": 30, "forty-five": 45, "sixty": 60, "ninety": 90,
}
# "90 days", "thirty (30) days", "forty (40) days", "five-year", "12 calendar months"
_DURATION = re.compile(
    rf"\b(\d+|{'|'.join(sorted(_NUMBER_WORDS, key=len, reverse=True))})\)?[\s-]*(?:\(\d+\)[\s-]*)?"
    r"(?:calendar\s+|business\s+)?(day|week|month|year)s?\b"
)


def _clip(value: float) -> float:
    """Clamp a value into [0, 1]."""
    return min(max(value, 0.0), 1.0)


@lru_cache(maxsize=None)
def _phrase_pattern(phrase: str) -> re.Pattern:
    """Match a phrase at a word start, case-insensitively ("each party" hits "Each party", not "breach party")."""
    return re.compile(rf"\b{re.escape(phrase)}", re.IGNORECASE)


def _count_phrases(text: str, phrases: Iterable[str]) -> int:
    """Count how many distinct phrases appear in text."""
    return sum(bool(_phrase_pattern(p).search(text)) for p in phrases)


def _durations_in_days(text: str) -> list[float]:
    """Return every duration written in text ("thirty (30) days", "5 years"), converted to days."""
    found = []
    for number, unit in _DURATION.findall(text.lower()):
        value = int(number) if number.isdigit() else _NUMBER_WORDS[number]
        found.append(value * _DAYS_PER_UNIT[unit])
    return found


def _matching_rules(clause: ClauseInput, contract_categories: Set[str], weight_set: WeightSet) -> list[CooccurrenceRule]:
    """Return the cooccurrence rules that target this clause and whose present/absent condition holds."""
    target = normalize_category(clause.category)
    detected = {normalize_category(name) for name in contract_categories}
    return [
        rule for rule in weight_set.cooccurrence_rules
        if normalize_category(rule.target_category) == target
        and (normalize_category(rule.other_category) in detected) == (rule.when == "present")
    ]


def category_severity(clause: ClauseInput, contract_categories: Set[str], weight_set: WeightSet) -> float:
    """
    Look up the category's severity (names normalized like cues.normalize_category), else default_severity,
    then scale it down by every matching negative cooccurrence rule: severity * (1 + sum of negatives).
    """
    severity = {normalize_category(name): value for name, value in weight_set.category_severity.items()}
    base = severity.get(normalize_category(clause.category), weight_set.default_severity)
    relief = sum(r.adjustment for r in _matching_rules(clause, contract_categories, weight_set) if r.adjustment < 0)
    return _clip(base * (1 + relief))


def one_sidedness(clause: ClauseInput, contract_categories: Set[str], weight_set: WeightSet) -> float:
    """
    Return aggravating / (aggravating + mitigating) distinct phrase hits in the evidence quote.
    0.0 when there is no evidence or no aggravating phrase; mitigating wording alone never adds risk.
    """
    if clause.evidence is None:
        return 0.0
    aggravating = _count_phrases(clause.evidence.quote, weight_set.aggravating_phrases)
    if aggravating == 0:
        return 0.0
    mitigating = _count_phrases(clause.evidence.quote, weight_set.mitigating_phrases)
    return aggravating / (aggravating + mitigating)


def magnitude(clause: ClauseInput, contract_categories: Set[str], weight_set: WeightSet) -> float:
    """
    Return 1.0 if the quote contains a broad-scope phrase, else the worst duration score from the
    category's MagnitudeRules (linear between low_days and high_days). 0.0 without evidence or matches.
    """
    # shortcut: durations only, no dollar amounts (a cap's size only means something relative to fees); add when the advisor defines amount cutoffs
    if clause.evidence is None:
        return 0.0
    quote = clause.evidence.quote
    if _count_phrases(quote, weight_set.scope_phrases):
        return 1.0

    target = normalize_category(clause.category)
    rules = [r for r in weight_set.magnitude_rules if normalize_category(r.category) == target]
    scores = [0.0]
    for days in _durations_in_days(quote):
        for rule in rules:
            fraction = (days - rule.low_days) / (rule.high_days - rule.low_days)
            scores.append(_clip(fraction if rule.longer_is_riskier else 1 - fraction))
    return max(scores)


def contract_context(clause: ClauseInput, contract_categories: Set[str], weight_set: WeightSet) -> float:
    """
    Start at 0 and add every matching positive cooccurrence rule (a risky combination or a missing
    protection), clipped to [0, 1]. Negative rules are handled by category_severity instead.
    """
    rules = _matching_rules(clause, contract_categories, weight_set)
    return _clip(sum(r.adjustment for r in rules if r.adjustment > 0))


SignalFn = Callable[[ClauseInput, Set[str], WeightSet], float]

# Signal name -> function; scoring.py computes every signal listed here
SIGNALS: dict[str, SignalFn] = {
    "category_severity": category_severity,
    "one_sidedness": one_sidedness,
    "magnitude": magnitude,
    "contract_context": contract_context,
}

# Fail at import time if the registry and SignalName drift apart
assert set(SIGNALS) == set(SIGNAL_NAMES), f"SIGNALS {sorted(SIGNALS)} != SignalName {sorted(SIGNAL_NAMES)}"
