import pytest

from src.evidence.schema import Evidence
from src.risk.schema import SIGNAL_NAMES, ClauseInput, WeightSet
from src.risk.signals import (
    SIGNALS,
    category_severity,
    cooccurrence,
    detection_confidence,
    risk_language,
)

WEIGHTS = WeightSet(
    weight_version_id="test-v1",
    created_by="tests",
    signal_weights={"category_severity": 0.4, "detection_confidence": 0.3, "risk_language": 0.2, "cooccurrence": 0.1},
    medium_threshold=0.4,
    high_threshold=0.7,
    category_severity={"Uncapped Liability": 0.9, "Ip Ownership Assignment": 0.6},
    default_severity=0.5,
    cooccurrence_rules=[
        {"target_category": "Uncapped Liability", "other_category": "Cap On Liability", "adjustment": -0.3},
        {"target_category": "Uncapped Liability", "other_category": "Insurance", "adjustment": 0.1},
        {"target_category": "Parties", "other_category": "Insurance", "adjustment": 0.8},
        {"target_category": "Parties", "other_category": "Exclusivity", "adjustment": -0.9},
    ],
    risk_phrases=["unlimited", "sole discretion", "irrevocable", "without cause"],
)


def _clause(category: str = "Uncapped Liability", probability: float = 0.8, quote: str | None = None) -> ClauseInput:
    """Build a ClauseInput, with an Evidence for quote when given."""
    ev = Evidence(quote=quote, start=0, end=len(quote)) if quote else None
    return ClauseInput(contract_id="c1", chunk_index=0, category=category, probability=probability, evidence=ev)


# ----- Registry -----
def test_signals_registry_matches_signal_names():
    """Every SignalName has a function and every function is a SignalName."""
    assert set(SIGNALS) == set(SIGNAL_NAMES)


@pytest.mark.parametrize("quote", [None, "Liability is unlimited, irrevocable and without cause."])
def test_signals_all_return_unit_floats(quote):
    """Every signal returns a float in [0, 1], with and without evidence."""
    for fn in SIGNALS.values():
        assert 0.0 <= fn(_clause(quote=quote), {"Insurance"}, WEIGHTS) <= 1.0


# ----- category_severity -----
@pytest.mark.parametrize(
    "category, expected",
    [
        ("Uncapped Liability", 0.9),       # listed
        ("uncapped liability", 0.9),       # normalized like cues.normalize_category
        ("IP Ownership Assignment", 0.6),  # "IP" vs "Ip" spelling
        ("Parties", 0.5),                  # not listed -> default_severity
    ],
)
def test_category_severity_lookup(category, expected):
    """Listed categories use their severity (case/punctuation-insensitive), others use the default."""
    assert category_severity(_clause(category), set(), WEIGHTS) == expected


# ----- detection_confidence -----
@pytest.mark.parametrize("probability", [0.0, 0.37, 1.0])
def test_detection_confidence_returns_probability(probability):
    """The signal is the classifier probability itself."""
    assert detection_confidence(_clause(probability=probability), set(), WEIGHTS) == probability


def test_detection_confidence_clips_out_of_range():
    """A probability that slipped past validation (model_construct) is clipped to [0, 1]."""
    over = ClauseInput.model_construct(contract_id="c1", chunk_index=0, category="Parties", probability=1.3)
    assert detection_confidence(over, set(), WEIGHTS) == 1.0


# ----- risk_language -----
@pytest.mark.parametrize(
    "quote, expected",
    [
        (None, 0.0),                                                    # no evidence
        ("Liability is capped at fees paid.", 0.0),                     # no phrases
        ("Liability is UNLIMITED.", 0.25),                              # 1 of 4, case-insensitive
        ("Unlimited and irrevocable, at its sole discretion.", 0.75),   # 3 of 4
        ("Unlimited, unlimited, unlimited.", 0.25),                     # repeats count once
        ("Unlimited, irrevocable, sole discretion, without cause.", 1.0),
    ],
)
def test_risk_language_fraction_of_phrases(quote, expected):
    """The signal is the fraction of distinct risk phrases in the quote, 0 without evidence."""
    assert risk_language(_clause(quote=quote), set(), WEIGHTS) == pytest.approx(expected)


def test_risk_language_no_phrases_returns_zero():
    """A weight set with no risk phrases gives 0 instead of dividing by zero."""
    ws = WEIGHTS.model_copy(update={"risk_phrases": []})
    assert risk_language(_clause(quote="unlimited"), set(), ws) == 0.0


# ----- cooccurrence -----
@pytest.mark.parametrize(
    "category, detected, expected",
    [
        ("Uncapped Liability", set(), 0.5),                              # no rules fire -> neutral
        ("Uncapped Liability", {"Cap On Liability"}, 0.2),               # 0.5 - 0.3
        ("Uncapped Liability", {"cap on liability"}, 0.2),               # names normalized
        ("Uncapped Liability", {"Cap On Liability", "Insurance"}, 0.3),  # 0.5 - 0.3 + 0.1
        ("Parties", {"Insurance"}, 1.0),                                 # 0.5 + 0.8 clipped to 1
        ("Parties", {"Exclusivity"}, 0.0),                               # 0.5 - 0.9 clipped to 0
        ("Governing Law", {"Cap On Liability", "Insurance"}, 0.5),       # no rule targets it
    ],
)
def test_cooccurrence_applies_matching_rules(category, detected, expected):
    """Rules fire only for this clause's category and detected other categories; the result is clipped."""
    assert cooccurrence(_clause(category), detected, WEIGHTS) == pytest.approx(expected)
