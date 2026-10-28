import json

import pytest
from pydantic import ValidationError

from src.evidence.schema import Evidence
from src.risk.schema import DEFAULT_WEIGHTS_PATH, ClauseInput, WeightSet, load_weight_set

VALID = {
    "weight_version_id": "test-v1",
    "created_by": "tests",
    "signal_weights": {
        "category_severity": 0.4,
        "one_sidedness": 0.3,
        "magnitude": 0.2,
        "contract_context": 0.1,
    },
    "medium_threshold": 0.4,
    "high_threshold": 0.7,
    "category_severity": {"Uncapped Liability": 0.9},
    "default_severity": 0.5,
    "cooccurrence_rules": [
        {"target_category": "Uncapped Liability", "other_category": "Cap On Liability", "adjustment": -0.3}
    ],
    "aggravating_phrases": ["sole discretion"],
    "mitigating_phrases": ["each party"],
    "scope_phrases": ["perpetual"],
    "magnitude_rules": [{"category": "Non-Compete", "low_days": 180, "high_days": 1095}],
}


def _with(**changes) -> dict:
    """Return a copy of VALID with the given top-level fields replaced."""
    return {**VALID, **changes}


# ----- Valid weight sets -----
def test_weight_set_valid_constructs():
    """A well-formed weight set builds, keeps its version id, and defaults rules to 'present' / longer riskier."""
    ws = WeightSet(**VALID)
    assert ws.weight_version_id == "test-v1"
    assert ws.cooccurrence_rules[0].when == "present"
    assert ws.magnitude_rules[0].longer_is_riskier is True


def test_weight_set_sum_within_tolerance_constructs():
    """Weights that sum to 1 only up to float error (0.1 + 0.2 + 0.3 + 0.4) are accepted."""
    weights = {"category_severity": 0.1, "one_sidedness": 0.2, "magnitude": 0.3, "contract_context": 0.4}
    assert WeightSet(**_with(signal_weights=weights))


def test_default_weights_file_loads():
    """The shipped placeholder weight set validates and carries the placeholder version id."""
    ws = load_weight_set(DEFAULT_WEIGHTS_PATH)
    assert ws.weight_version_id == "v1-placeholder"
    assert (ws.medium_threshold, ws.high_threshold) == (0.35, 0.55)


def test_load_weight_set_invalid_file_raises(tmp_path):
    """A weight set JSON with medium_threshold above high_threshold is rejected by the loader."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(_with(medium_threshold=0.9)), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_weight_set(path)


# ----- Invalid weight sets -----
@pytest.mark.parametrize(
    "changes",
    [
        # weights don't sum to 1
        {"signal_weights": {**VALID["signal_weights"], "contract_context": 0.2}},
        {"signal_weights": {**VALID["signal_weights"], "contract_context": 0.0}},
        # a signal is missing or unknown (including the retired v0 signals)
        {"signal_weights": {"category_severity": 0.5, "one_sidedness": 0.5}},
        {"signal_weights": {**VALID["signal_weights"], "detection_confidence": 0.0}},
        # thresholds out of order or equal
        {"medium_threshold": 0.7, "high_threshold": 0.4},
        {"medium_threshold": 0.5, "high_threshold": 0.5},
        # values outside [0, 1]
        {"signal_weights": {**VALID["signal_weights"], "category_severity": 1.2, "one_sidedness": -0.5}},
        {"medium_threshold": -0.1},
        {"high_threshold": 1.1},
        {"default_severity": 1.5},
        {"category_severity": {"Uncapped Liability": -0.1}},
        {"cooccurrence_rules": [{"target_category": "Parties", "other_category": "Insurance", "adjustment": 1.5}]},
        # bad rule fields
        {"cooccurrence_rules": [{"target_category": "Parties", "other_category": "Insurance", "when": "maybe", "adjustment": 0.1}]},
        {"magnitude_rules": [{"category": "Non-Compete", "low_days": 365, "high_days": 365}]},
        {"magnitude_rules": [{"category": "Non-Compete", "low_days": -1, "high_days": 365}]},
        # unknown category names and misspelled fields
        {"category_severity": {"Uncaped Liability": 0.9}},
        {"cooccurrence_rules": [{"target_category": "Parties", "other_category": "Nope", "adjustment": 0.1}]},
        {"magnitude_rules": [{"category": "Non Compete Clause", "low_days": 1, "high_days": 2}]},
        {"hgh_threshold": 0.7},
        {"risk_phrases": ["unlimited"]},
        # empty ids and empty phrases
        {"weight_version_id": ""},
        {"aggravating_phrases": [""]},
        {"scope_phrases": [""]},
    ],
)
def test_weight_set_invalid_raises(changes):
    """Bad sums, missing signals, out-of-order thresholds, out-of-range values and typos are rejected."""
    with pytest.raises(ValidationError):
        WeightSet(**_with(**changes))


# ----- ClauseInput -----
def test_clause_input_without_evidence_constructs():
    """Unknown-queue clauses (evidence=None) are valid inputs."""
    clause = ClauseInput(contract_id="c", chunk_index=0, category="Parties", probability=0.5)
    assert clause.evidence is None


def test_clause_input_with_evidence_constructs():
    """A clause can carry an Evidence from the evidence stage."""
    ev = Evidence(quote="abc", start=0, end=3)
    assert ClauseInput(contract_id="c", chunk_index=0, category="Parties", probability=0.5, evidence=ev).evidence == ev


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_clause_input_probability_out_of_range_raises(probability):
    """Probabilities outside [0, 1] are rejected."""
    with pytest.raises(ValidationError):
        ClauseInput(contract_id="c", chunk_index=0, category="Parties", probability=probability)
