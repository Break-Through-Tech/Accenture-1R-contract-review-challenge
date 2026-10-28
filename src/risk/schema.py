"""
Risk scoring is a judgement layered on top of detection: the classifier says a clause is
there, and this stage says how worried a reviewer should be about it. Keeping the two
separate means the scoring logic can be retuned without retraining the model.

This module holds the Pydantic models that cross the risk-scoring stage boundary:
- WeightSet: the advisor-owned, versioned config that scoring reads (signal weights,
  rating thresholds, category severities, context rules, phrase lists, magnitude rules)
- ClauseInput: one detected clause, i.e. a cached classifier probability plus the
  Evidence from src/evidence/ (or None if the clause went to the Unknown queue)
- ClauseScore: the scored row, which always records the weight_version_id it used

PLACEHOLDER: the four signals in SignalName are proposals until they are settled with
the advisor. See signals.py for what each one measures and how to add, remove or rename one.

Detection confidence is deliberately not a signal: a confident detection is not a riskier
clause. The classifier's per-class thresholds already gate which clauses reach scoring,
and ClauseScore carries the probability so the UI can still show it.

Creating a new weight version:
1. Copy src/risk/weights/default_weights_v1.json to a new file
2. Give it a new, unique weight_version_id and set created_by
3. Edit the values; load_weight_set() validates bounds, the weight sum and threshold order
Never edit a version in place: scores record their weight_version_id, so changing the
values behind an existing id would make old scores unreproducible.
"""
import math
from pathlib import Path
from typing import Annotated, Literal, get_args

from typing_extensions import Self
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.evidence.cues import normalize_category
from src.evidence.schema import Evidence
from src.labels.categories import CATEGORIES

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "default_weights_v1.json"

# PLACEHOLDER signals pending advisor sign-off; every name here needs a function in signals.SIGNALS
SignalName = Literal["category_severity", "one_sidedness", "magnitude", "contract_context"]
SIGNAL_NAMES: tuple[str, ...] = get_args(SignalName)

Rating = Literal["Low", "Medium", "High"]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
Phrase = Annotated[str, Field(min_length=1)]

# Allowed slack when checking that signal weights sum to 1.0 (JSON floats rarely sum exactly)
WEIGHT_SUM_TOLERANCE = 1e-6

_KNOWN_CATEGORIES = {normalize_category(c) for c in CATEGORIES}


def _check_category(name: str) -> str:
    """Reject category names that don't match any CUAD category (typos would silently fall to default)."""
    if normalize_category(name) not in _KNOWN_CATEGORIES:
        raise ValueError(f"Unknown CUAD category: {name!r}")
    return name


class CooccurrenceRule(BaseModel):
    """
    A rule that fires for target_category when other_category is present in (or absent from) the
    same contract. A positive adjustment raises the target's contract_context signal (e.g. Uncapped
    Liability with no Insurance); a negative one lowers its category_severity, because a protection
    elsewhere in the contract softens the clause's impact (e.g. Uncapped Liability with a Cap On Liability).
    """
    model_config = ConfigDict(extra="forbid")

    target_category: str
    other_category: str
    when: Literal["present", "absent"] = "present"
    adjustment: float = Field(ge=-1.0, le=1.0)

    @field_validator("target_category", "other_category")
    @classmethod
    def check_categories(cls, name: str) -> str:
        return _check_category(name)


class MagnitudeRule(BaseModel):
    """
    How long is too long for one category's durations, in days. A duration at or below low_days
    scores 0 and one at or above high_days scores 1, linearly in between. Set longer_is_riskier
    to False when a short duration is the risk (e.g. a 30-day warranty).
    """
    model_config = ConfigDict(extra="forbid")

    category: str
    low_days: float = Field(ge=0.0)
    high_days: float
    longer_is_riskier: bool = True

    @field_validator("category")
    @classmethod
    def check_category(cls, name: str) -> str:
        return _check_category(name)

    @model_validator(mode="after")
    def check_range(self) -> Self:
        if self.low_days >= self.high_days:
            raise ValueError("low_days must be below high_days")
        return self


class WeightSet(BaseModel):
    """
    One immutable, versioned set of scoring parameters, owned by the advisor.
    Unknown keys are rejected so a misspelled field in the JSON can't be silently ignored.
    """
    model_config = ConfigDict(extra="forbid")

    weight_version_id: str = Field(min_length=1)
    created_by: str = Field(min_length=1)
    description: str = ""
    signal_weights: dict[SignalName, UnitFloat]
    medium_threshold: UnitFloat
    high_threshold: UnitFloat
    category_severity: dict[str, UnitFloat] = {}
    default_severity: UnitFloat
    cooccurrence_rules: list[CooccurrenceRule] = []
    aggravating_phrases: list[Phrase] = []
    mitigating_phrases: list[Phrase] = []
    scope_phrases: list[Phrase] = []
    magnitude_rules: list[MagnitudeRule] = []

    @field_validator("category_severity")
    @classmethod
    def check_severity_categories(cls, severity: dict[str, float]) -> dict[str, float]:
        for name in severity:
            _check_category(name)
        return severity

    # Every signal needs a weight, the weights must sum to 1, and Medium must sit below High
    @model_validator(mode="after")
    def check_weights_and_thresholds(self) -> Self:
        missing = set(SIGNAL_NAMES) - set(self.signal_weights)
        if missing:
            raise ValueError(f"signal_weights is missing: {sorted(missing)}")
        total = sum(self.signal_weights.values())
        if not math.isclose(total, 1.0, abs_tol=WEIGHT_SUM_TOLERANCE):
            raise ValueError(f"signal_weights must sum to 1.0, got {total}")
        if self.medium_threshold >= self.high_threshold:
            raise ValueError("medium_threshold must be below high_threshold")
        return self


class ClauseInput(BaseModel):
    """
    One detected clause to score: the cached classifier probability for (chunk, category)
    plus its grounded Evidence, or None when the clause went to the Unknown queue.
    Only pass clauses that cleared the classifier's per-class threshold: every input counts
    as detected when contract_context checks which categories are present.
    """
    contract_id: str
    chunk_index: int = Field(ge=0)
    category: str
    probability: UnitFloat
    evidence: Evidence | None = None


class ClauseScore(BaseModel):
    """A scored clause: each signal value, their weighted composite, the rating, and the weight version used."""
    contract_id: str
    chunk_index: int
    category: str
    probability: UnitFloat
    signals: dict[SignalName, UnitFloat]
    composite_score: UnitFloat
    rating: Rating
    has_evidence: bool
    weight_version_id: str


def load_weight_set(path: str | Path = DEFAULT_WEIGHTS_PATH) -> WeightSet:
    """Read a weight set JSON file and validate it; raises pydantic.ValidationError if it is invalid."""
    return WeightSet.model_validate_json(Path(path).read_text(encoding="utf-8"))
