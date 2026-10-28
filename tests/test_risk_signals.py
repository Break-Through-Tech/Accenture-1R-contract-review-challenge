import pytest

from src.evidence.schema import Evidence
from src.risk.schema import SIGNAL_NAMES, ClauseInput, WeightSet
from src.risk.signals import (
    SIGNALS,
    _durations_in_days,
    category_severity,
    contract_context,
    magnitude,
    one_sidedness,
)

WEIGHTS = WeightSet(
    weight_version_id="test-v1",
    created_by="tests",
    signal_weights={"category_severity": 0.4, "one_sidedness": 0.2, "magnitude": 0.2, "contract_context": 0.2},
    medium_threshold=0.4,
    high_threshold=0.7,
    category_severity={"Uncapped Liability": 0.9, "Ip Ownership Assignment": 0.6},
    default_severity=0.5,
    cooccurrence_rules=[
        {"target_category": "Uncapped Liability", "other_category": "Cap On Liability", "adjustment": -0.3},
        {"target_category": "Uncapped Liability", "other_category": "Insurance", "when": "absent", "adjustment": 0.5},
        {"target_category": "Uncapped Liability", "other_category": "Liquidated Damages", "adjustment": 0.3},
        {"target_category": "Parties", "other_category": "Insurance", "adjustment": 0.8},
        {"target_category": "Parties", "other_category": "Exclusivity", "adjustment": 0.7},
        {"target_category": "Parties", "other_category": "Audit Rights", "adjustment": -0.7},
        {"target_category": "Parties", "other_category": "Exclusivity", "adjustment": -0.6},
    ],
    aggravating_phrases=["sole discretion", "without notice", "for any reason"],
    mitigating_phrases=["each party", "reasonable"],
    scope_phrases=["perpetual", "worldwide"],
    magnitude_rules=[
        {"category": "Non-Compete", "low_days": 365, "high_days": 1095},
        {"category": "Warranty Duration", "low_days": 90, "high_days": 365, "longer_is_riskier": False},
    ],
)


def _clause(category: str = "Uncapped Liability", quote: str | None = None) -> ClauseInput:
    """Build a ClauseInput, with an Evidence for quote when given."""
    ev = Evidence(quote=quote, start=0, end=len(quote)) if quote else None
    return ClauseInput(contract_id="c1", chunk_index=0, category=category, probability=0.8, evidence=ev)


# ----- Registry -----
def test_signals_registry_matches_signal_names():
    """Every SignalName has a function and every function is a SignalName."""
    assert set(SIGNALS) == set(SIGNAL_NAMES)


@pytest.mark.parametrize("quote", [None, "At its sole discretion, a perpetual, worldwide right for five (5) years."])
def test_signals_all_return_unit_floats(quote):
    """Every signal returns a float in [0, 1], with and without evidence."""
    for fn in SIGNALS.values():
        assert 0.0 <= fn(_clause(quote=quote), {"Liquidated Damages"}, WEIGHTS) <= 1.0


# ----- category_severity -----
@pytest.mark.parametrize(
    "category, expected",
    [
        ("Uncapped Liability", 0.9),       # listed
        ("uncapped liability", 0.9),       # normalized like cues.normalize_category
        ("IP Ownership Assignment", 0.6),  # "IP" vs "Ip" spelling
        ("Governing Law", 0.5),            # not listed -> default_severity
    ],
)
def test_category_severity_lookup(category, expected):
    """Listed categories use their severity (case/punctuation-insensitive), others use the default."""
    assert category_severity(_clause(category), {"Insurance"}, WEIGHTS) == expected


@pytest.mark.parametrize(
    "category, detected, expected",
    [
        ("Uncapped Liability", {"Cap On Liability"}, 0.63),         # 0.9 * (1 - 0.3)
        ("Uncapped Liability", {"Liquidated Damages"}, 0.9),        # positive rules don't touch severity
        ("Parties", {"Audit Rights"}, 0.15),                        # 0.5 * (1 - 0.7)
        ("Parties", {"Audit Rights", "Exclusivity"}, 0.0),          # 0.5 * (1 - 1.3) clipped to 0
    ],
)
def test_category_severity_scaled_by_negative_rules(category, detected, expected):
    """A protective clause elsewhere in the contract scales severity down; the result is clipped to 0."""
    assert category_severity(_clause(category), detected, WEIGHTS) == pytest.approx(expected)


# ----- one_sidedness -----
@pytest.mark.parametrize(
    "quote, expected",
    [
        (None, 0.0),                                                          # no evidence
        ("Liability is capped at fees paid.", 0.0),                           # no phrases
        ("Each party acts in a reasonable manner.", 0.0),                     # mitigating only adds nothing
        ("At its SOLE DISCRETION.", 1.0),                                     # 1 aggravating, case-insensitive
        ("Sole discretion, sole discretion.", 1.0),                           # repeats count once
        ("At its sole discretion, acting reasonable.", 0.5),                  # 1 / (1 + 1)
        ("For any reason, at its sole discretion, without notice; each party.", 0.75),  # 3 / (3 + 1)
        ("In breach party B may act at its sole discretion.", 1.0),           # "breach party" is not "each party"
    ],
)
def test_one_sidedness_ratio_of_phrases(quote, expected):
    """The signal is aggravating / (aggravating + mitigating) distinct phrase hits, 0 without aggravation."""
    assert one_sidedness(_clause(quote=quote), set(), WEIGHTS) == pytest.approx(expected)


# ----- magnitude -----
@pytest.mark.parametrize(
    "text, expected",
    [
        ("within 90 days", [90]),
        ("thirty (30) days", [30]),
        ("forty (40) days", [40]),
        ("five (5) years", [1825]),
        ("a five-year term", [1825]),
        ("twelve (12) calendar months", [360]),
        ("twenty-four months", [720]),
        ("one year and 2 weeks", [365, 14]),
        ("payable monthly or yearly", []),
        ("no durations here", []),
    ],
)
def test_durations_in_days_parses_contract_wording(text, expected):
    """Digit, word and 'word (digit)' durations are found and converted to days (month = 30, year = 365)."""
    assert _durations_in_days(text) == expected


@pytest.mark.parametrize(
    "category, quote, expected",
    [
        ("Non-Compete", None, 0.0),                                     # no evidence
        ("Non-Compete", "for six (6) months", 0.0),                     # 180 <= low_days
        ("Non-Compete", "for two (2) years", 0.5),                      # (730 - 365) / 730
        ("Non-Compete", "for five years", 1.0),                         # past high_days, clipped
        ("Non-Compete", "for one year, then 3 years", 1.0),             # worst duration wins
        ("Warranty Duration", "warranted for 30 days", 1.0),            # shorter is riskier
        ("Warranty Duration", "warranted for one year", 0.0),
        ("Governing Law", "for five years", 0.0),                       # no rule for this category
        ("Governing Law", "a perpetual, worldwide license", 1.0),       # scope phrase, any category
        ("Non-Compete", "No duration or scope words.", 0.0),
    ],
)
def test_magnitude_scores_scope_and_durations(category, quote, expected):
    """Scope phrases score 1; durations are scored against the category's rule; otherwise 0."""
    assert magnitude(_clause(category, quote), set(), WEIGHTS) == pytest.approx(expected)


# ----- contract_context -----
@pytest.mark.parametrize(
    "category, detected, expected",
    [
        ("Uncapped Liability", {"Insurance"}, 0.0),                               # nothing fires -> 0, not 0.5
        ("Uncapped Liability", set(), 0.5),                                       # Insurance absent
        ("Uncapped Liability", {"Insurance", "Liquidated Damages"}, 0.3),         # risky combination
        ("Uncapped Liability", {"Liquidated Damages"}, 0.8),                      # 0.5 + 0.3
        ("Uncapped Liability", {"Insurance", "Cap On Liability"}, 0.0),           # negative rules ignored here
        ("Parties", {"Insurance", "Exclusivity"}, 1.0),                           # 0.8 + 0.7 clipped to 1
        ("Governing Law", {"Liquidated Damages"}, 0.0),                           # no rule targets it
    ],
)
def test_contract_context_adds_positive_rules(category, detected, expected):
    """Starts at 0 and adds matching positive present/absent rules, clipped to 1."""
    assert contract_context(_clause(category), detected, WEIGHTS) == pytest.approx(expected)
