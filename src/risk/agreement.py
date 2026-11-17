"""
Measures how well the rule-based risk scores agree with the advisor's own judgement of the same
clauses. This is the risk layer's success criterion: with no ground-truth risk labels, the advisor's
hand-ranked clauses are the reference.

How everything comes together:
1. scripts/build_advisor_packet.py samples scored clauses and writes a blind packet (no model
   output) plus a key (PacketKey) holding everything needed to rescore each clause without a model
2. The advisor fills in a bucket (Low / Medium / High), a rank (1 = riskiest) and notes per item
3. load_returned_packet() joins the returned packet with its key into AdvisorRanking records,
   rejecting malformed input with a message that names the item and the problem
4. evaluate_agreement() pairs each judged clause with the model's score (stored in the key, or
   rescored under another WeightSet) and computes:
   - Spearman correlation, with ties given average ranks (buckets are heavily tied by design)
   - bucket agreement: exact rate, adjacent rate, 3x3 confusion matrix, weighted kappa
   - precision and recall of the model's High bucket against the advisor's High bucket
   - percentile bootstrap confidence intervals over clauses for every metric
   - a sweep of the Medium and High thresholds, with the range over which agreement holds

Everything except the two loaders is pure. Scoring only reads the key, so rescoring under a new
weight version never reruns the classifier.

Inputs from other modules:
'WeightSet' from schema.load_weight_set(), 'score_clause' from scoring.py
"""
import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.stats import rankdata

from src.evidence.cues import normalize_category
from src.evidence.schema import Evidence
from src.labels.categories import CATEGORIES
from src.risk.schema import ClauseInput, ClauseScore, Rating, WeightSet
from src.risk.scoring import score_clause

BUCKETS: tuple[Rating, ...] = ("Low", "Medium", "High")
BUCKET_ORDINAL: dict[str, int] = {bucket: i for i, bucket in enumerate(BUCKETS)}

# Columns the advisor packet must carry for the loader to join it with its key
PACKET_ID_COLUMN = "packet_id"
ITEM_ID_COLUMN = "item_id"
ADVISOR_COLUMNS = ("advisor_bucket", "advisor_rank", "advisor_notes")
REQUIRED_PACKET_COLUMNS = (PACKET_ID_COLUMN, ITEM_ID_COLUMN, "category", "quote", *ADVISOR_COLUMNS)

# Spearman of 0.7 or more is the usual reading of "strong"; the criterion asks for strong correlation
STRONG_SPEARMAN = 0.7
# An interval wider than this is too wide to support a conclusion about a rate or correlation
WIDE_INTERVAL = 0.3

_CATEGORY_BY_NORMALIZED = {normalize_category(category): category for category in CATEGORIES}


def _canonical_category(name: str) -> str:
    """Return the canonical CUAD spelling of a category name; an unknown name raises ValueError."""
    try:
        return _CATEGORY_BY_NORMALIZED[normalize_category(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown CUAD category: {name!r}") from exc


def _check_span(start: int, end: int, quote: str) -> None:
    """Raise ValueError unless [start:end] is a non-empty span the quote exactly fills."""
    if end <= start:
        raise ValueError("end must be after start")
    if len(quote) != end - start:
        raise ValueError(f"quote length ({len(quote)}) must equal end - start ({end - start})")


# ----- Records that cross the stage boundary -----
class AdvisorRanking(BaseModel):
    """
    One advisor judgement of one clause. Field names match the advisor_rankings table.
    advisor_rank: 1 is the riskiest; ties are allowed. advisor_bucket: Low, Medium or High.
    Either may be missing (bucket-only or rank-only judgements), but not both.
    start and end are absolute, end-exclusive offsets of the quote in the contract text.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str = Field(min_length=1)
    category: str
    start: int = Field(ge=0)
    end: int
    quote: str
    advisor_rank: int | None = Field(default=None, ge=1)
    advisor_bucket: Rating | None = None
    reviewer: str = Field(min_length=1)
    ranking_set: str = Field(min_length=1)

    @field_validator("category")
    @classmethod
    def check_category(cls, name: str) -> str:
        return _canonical_category(name)

    # The quote must fill the span exactly, and a record must carry some judgement
    @model_validator(mode="after")
    def check_span_and_judgement(self) -> Self:
        _check_span(self.start, self.end, self.quote)
        if self.advisor_rank is None and self.advisor_bucket is None:
            raise ValueError("an advisor ranking needs advisor_rank, advisor_bucket or both")
        return self


class PacketKeyItem(BaseModel):
    """
    The hidden half of one packet item: everything needed to rebuild its ClauseInput and rescore
    it under any weight version (contract_categories feeds the contract_context signal), plus the
    model's score at generation time. Never shown to the advisor.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    category: str
    start: int = Field(ge=0)
    end: int
    quote: str
    probability: float = Field(ge=0.0, le=1.0)
    contract_categories: tuple[str, ...]
    composite_score: float = Field(ge=0.0, le=1.0)
    rating: Rating
    signals: dict[str, float]
    weight_version_id: str = Field(min_length=1)
    weight_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("category")
    @classmethod
    def check_category(cls, name: str) -> str:
        return _canonical_category(name)

    @field_validator("contract_categories")
    @classmethod
    def check_contract_categories(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({_canonical_category(name) for name in names}))

    # The span must be grounded-shaped, and the clause's own category is always detected in its contract
    @model_validator(mode="after")
    def check_item(self) -> Self:
        _check_span(self.start, self.end, self.quote)
        if self.category not in self.contract_categories:
            raise ValueError(f"item {self.item_id}: contract_categories must include {self.category!r}")
        return self

    def to_clause_input(self) -> ClauseInput:
        """Rebuild the ClauseInput the model's score came from, with the quote as grounded Evidence."""
        return ClauseInput(
            contract_id=self.contract_id, chunk_index=self.chunk_index, category=self.category,
            probability=self.probability, evidence=Evidence(quote=self.quote, start=self.start, end=self.end),
        )


class PacketKey(BaseModel):
    """
    The key file for one advisor packet: generation settings and one PacketKeyItem per item.
    packet_id is printed on every packet row, so a returned packet can't be joined to the wrong key.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str = Field(min_length=1)
    weight_version_id: str = Field(min_length=1)
    weight_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    medium_threshold: float = Field(ge=0.0, le=1.0)
    high_threshold: float = Field(ge=0.0, le=1.0)
    settings: dict = Field(default_factory=dict)
    items: tuple[PacketKeyItem, ...] = Field(min_length=1)

    # Item ids and clause identities must be unique, and every item scored under the key's weight set
    @model_validator(mode="after")
    def check_items(self) -> Self:
        ids = [item.item_id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("item_id values in the key must be unique")
        spans = [(i.contract_id, i.category, i.start, i.end) for i in self.items]
        if len(set(spans)) != len(spans):
            raise ValueError("two key items share the same contract, category and span")
        for item in self.items:
            if (item.weight_version_id, item.weight_set_sha256) != (self.weight_version_id, self.weight_set_sha256):
                raise ValueError(f"item {item.item_id} was scored under a different weight set than the key")
        return self


class JudgedItem(BaseModel):
    """One packet item the advisor judged: the AdvisorRanking record plus the free-text notes."""
    item_id: str
    ranking: AdvisorRanking
    notes: str = ""


class ReturnedPacket(BaseModel):
    """A returned packet joined with its key. skipped lists item ids left with no bucket and no rank."""
    packet_id: str
    judged: list[JudgedItem]
    skipped: list[str] = Field(default_factory=list)


def load_key(path: str | Path) -> PacketKey:
    """Read and validate a packet key JSON file; raises pydantic.ValidationError if it is malformed."""
    return PacketKey.model_validate_json(Path(path).read_text(encoding="utf-8"))


# ----- Loading a returned packet -----
def _blank(value: object) -> bool:
    """True for an empty cell (None, NaN or whitespace only)."""
    return value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip()


def _collapse(text: object) -> str:
    """Collapse runs of whitespace to one space, so a quote survives a spreadsheet round trip."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _parse_bucket(value: object, item_id: str) -> Rating | None:
    """Read a bucket cell: Low, Medium or High in any case ("Med" accepted); blank gives None."""
    if _blank(value):
        return None
    text = str(value).strip().lower()
    text = "medium" if text == "med" else text
    for bucket in BUCKETS:
        if text == bucket.lower():
            return bucket
    raise ValueError(f"item {item_id}: advisor_bucket must be Low, Medium or High, got {value!r}")


def _parse_rank(value: object, item_id: str, item_count: int) -> int | None:
    """Read a rank cell: a whole number from 1 to item_count ("3.0" from a spreadsheet is fine)."""
    if _blank(value):
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        raise ValueError(f"item {item_id}: advisor_rank must be a whole number, got {value!r}") from None
    if not number.is_integer() or not 1 <= number <= item_count:
        raise ValueError(f"item {item_id}: advisor_rank must be a whole number from 1 to {item_count}, got {value!r}")
    return int(number)


def read_packet_csv(path: str | Path) -> pd.DataFrame:
    """Read a returned packet CSV as text (UTF-8, falling back to Windows-1252 for Excel saves)."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="cp1252")


def join_packet(packet: pd.DataFrame, key: PacketKey, reviewer: str, ranking_set: str | None = None) -> ReturnedPacket:
    """
    Join a returned packet (one row per item) with its key into AdvisorRanking records.

    Raises ValueError, naming the item, for: missing columns, a packet_id that isn't the key's,
    duplicate, unknown or missing item ids, a category or quote that no longer matches the key
    (rows shuffled or edited by hand), and an unreadable bucket or rank. Rows with neither a
    bucket nor a rank are skipped, not rejected, and listed in ReturnedPacket.skipped.
    ranking_set defaults to the key's packet_id.
    """
    # The packet must have the columns the join reads, and belong to this key; fully blank rows are ignored
    missing = [column for column in REQUIRED_PACKET_COLUMNS if column not in packet.columns]
    if missing:
        raise ValueError(f"returned packet is missing columns: {missing}")
    packet = packet[~packet.map(_blank).all(axis=1)]
    packet_ids = {str(value).strip() for value in packet[PACKET_ID_COLUMN] if not _blank(value)}
    if packet_ids != {key.packet_id}:
        raise ValueError(f"returned packet has packet_id {sorted(packet_ids)}, but the key is {key.packet_id!r}")

    # Every key item must appear exactly once, and nothing else
    ids = [str(value).strip() for value in packet[ITEM_ID_COLUMN]]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise ValueError(f"returned packet repeats item ids: {duplicated}")
    by_id = {item.item_id: item for item in key.items}
    unknown, absent = sorted(set(ids) - set(by_id)), sorted(set(by_id) - set(ids))
    if unknown:
        raise ValueError(f"returned packet has item ids not in the key: {unknown}")
    if absent:
        raise ValueError(f"returned packet is missing items from the key: {absent}")

    # Build one record per judged row, checking the row still describes its key item
    judged, skipped = [], []
    for item_id, (_, row) in zip(ids, packet.iterrows()):
        item = by_id[item_id]
        same_category = _canonical_category(str(row["category"])) == item.category
        if not same_category or _collapse(row["quote"]) != _collapse(item.quote):
            raise ValueError(f"item {item_id}: category or quote doesn't match the key (rows edited or misaligned?)")
        bucket = _parse_bucket(row["advisor_bucket"], item_id)
        rank = _parse_rank(row["advisor_rank"], item_id, len(key.items))
        if bucket is None and rank is None:
            skipped.append(item_id)
            continue
        ranking = AdvisorRanking(
            contract_id=item.contract_id, category=item.category, start=item.start, end=item.end, quote=item.quote,
            advisor_rank=rank, advisor_bucket=bucket, reviewer=reviewer, ranking_set=ranking_set or key.packet_id,
        )
        notes = "" if _blank(row["advisor_notes"]) else str(row["advisor_notes"]).strip()
        judged.append(JudgedItem(item_id=item_id, ranking=ranking, notes=notes))
    if not judged:
        raise ValueError("returned packet has no judged items: every bucket and rank cell is blank")
    return ReturnedPacket(packet_id=key.packet_id, judged=judged, skipped=sorted(skipped))


def load_returned_packet(path: str | Path, key: PacketKey, reviewer: str, ranking_set: str | None = None) -> ReturnedPacket:
    """Read a returned packet CSV and join it with its key (see join_packet for the checks)."""
    return join_packet(read_packet_csv(path), key, reviewer, ranking_set)


# ----- Rescoring from the key -----
def rescore_key(key: PacketKey, weight_set: WeightSet) -> dict[str, ClauseScore]:
    """Score every key item under weight_set from the stored inputs alone; returns item_id -> ClauseScore."""
    return {item.item_id: score_clause(item.to_clause_input(), set(item.contract_categories), weight_set)
            for item in key.items}


def bucket_for(score: float, medium_threshold: float, high_threshold: float) -> Rating:
    """The rating scoring.rate() would give score under these thresholds (High >= high, Medium >= medium)."""
    if score >= high_threshold:
        return "High"
    if score >= medium_threshold:
        return "Medium"
    return "Low"


# ----- Metrics -----
def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """
    Spearman correlation: the Pearson correlation of the two average-rank vectors, so tied values
    share the mean of the ranks they span. None when either side is constant or has under 2 values.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        return None
    rx, ry = rankdata(x, method="average"), rankdata(y, method="average")
    dx, dy = rx - rx.mean(), ry - ry.mean()
    denominator = math.sqrt(float(dx @ dx) * float(dy @ dy))
    return None if denominator == 0 else float(dx @ dy) / denominator


def confusion_matrix(advisor: Sequence[str], model: Sequence[str]) -> list[list[int]]:
    """3x3 counts with rows = advisor bucket and columns = model bucket, both in Low, Medium, High order."""
    matrix = [[0] * len(BUCKETS) for _ in BUCKETS]
    for a, m in zip(advisor, model, strict=True):
        matrix[BUCKET_ORDINAL[a]][BUCKET_ORDINAL[m]] += 1
    return matrix


def exact_rate(advisor: Sequence[str], model: Sequence[str]) -> float | None:
    """Share of clauses where model and advisor give the same bucket; None for no clauses."""
    pairs = list(zip(advisor, model, strict=True))
    return sum(a == m for a, m in pairs) / len(pairs) if pairs else None


def adjacent_rate(advisor: Sequence[str], model: Sequence[str]) -> float | None:
    """Share of clauses where the buckets are equal or one step apart (only Low vs High misses)."""
    pairs = list(zip(advisor, model, strict=True))
    return sum(abs(BUCKET_ORDINAL[a] - BUCKET_ORDINAL[m]) <= 1 for a, m in pairs) / len(pairs) if pairs else None


def weighted_kappa(advisor: Sequence[str], model: Sequence[str], scheme: str = "linear") -> float | None:
    """
    Cohen's weighted kappa over the ordered buckets: 1 is perfect agreement, 0 is what chance would
    give with these bucket frequencies, below 0 is worse than chance. A one-step disagreement gets
    partial credit (linear: 1 - |i-j|/2, quadratic: 1 - (i-j)^2/4). None when chance agreement is
    already 1 (both sides used one bucket only), where kappa is undefined.
    """
    if scheme not in ("linear", "quadratic"):
        raise ValueError(f"scheme must be 'linear' or 'quadratic', got {scheme!r}")
    counts = np.array(confusion_matrix(advisor, model), dtype=float)
    total = counts.sum()
    if total == 0:
        return None
    # Agreement weights: 1 on the diagonal, falling with distance between buckets
    k = len(BUCKETS)
    distance = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    weights = 1 - (distance if scheme == "linear" else distance ** 2)
    observed = float((weights * counts).sum() / total)
    expected = float(weights @ (counts.sum(axis=0) / total) @ (counts.sum(axis=1) / total))
    return None if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def high_precision(advisor: Sequence[str], model: Sequence[str]) -> float | None:
    """Of the clauses the model calls High, the share the advisor also calls High; None if the model calls none High."""
    called = [a for a, m in zip(advisor, model, strict=True) if m == "High"]
    return sum(a == "High" for a in called) / len(called) if called else None


def high_recall(advisor: Sequence[str], model: Sequence[str]) -> float | None:
    """Of the clauses the advisor calls High, the share the model also calls High; None if the advisor calls none High."""
    wanted = [m for a, m in zip(advisor, model, strict=True) if a == "High"]
    return sum(m == "High" for m in wanted) / len(wanted) if wanted else None


class MetricEstimate(BaseModel):
    """
    A point estimate with its percentile bootstrap interval over clauses.
    n: clauses the metric used. n_boot_valid: resamples where the metric was defined (a resample
    can draw, say, no High clauses). value and bounds are None when the metric is undefined.
    """
    value: float | None
    ci_low: float | None
    ci_high: float | None
    confidence: float
    n: int
    n_boot_valid: int

    @property
    def ci_width(self) -> float | None:
        """Width of the interval, or None if there is no interval."""
        return None if self.ci_low is None or self.ci_high is None else self.ci_high - self.ci_low


def bootstrap_estimate(
    statistic: Callable[..., float | None],
    columns: Sequence[Sequence],
    n_boot: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> MetricEstimate:
    """
    Compute statistic(*columns) and a percentile bootstrap interval: resample clauses (rows across
    every column together) with replacement n_boot times and take the central confidence share of
    the defined results. Seeded, so the same inputs always give the same interval.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    arrays = [np.asarray(column, dtype=object) for column in columns]
    n = len(arrays[0]) if arrays else 0
    value = statistic(*[list(a) for a in arrays]) if n else None

    # Resample rows; keep only resamples where the statistic is defined
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(n_boot if n else 0):
        rows = rng.integers(0, n, size=n)
        result = statistic(*[list(a[rows]) for a in arrays])
        if result is not None and not math.isnan(result):
            replicates.append(result)
    if value is None or not replicates:
        return MetricEstimate(value=value, ci_low=None, ci_high=None, confidence=confidence, n=n,
                              n_boot_valid=len(replicates))
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(replicates, [tail, 100 - tail])
    return MetricEstimate(value=value, ci_low=float(low), ci_high=float(high), confidence=confidence, n=n,
                          n_boot_valid=len(replicates))


# ----- Threshold sweep -----
class ThresholdPoint(BaseModel):
    """Agreement with the advisor's buckets at one (medium_threshold, high_threshold) setting."""
    medium_threshold: float
    high_threshold: float
    exact_rate: float
    weighted_kappa: float | None
    high_count: int
    medium_count: int
    low_count: int
    changed_vs_base: int


class ThresholdSweep(BaseModel):
    """
    The grid of threshold settings, and how stable agreement is around the base setting.
    A setting counts as stable when its exact rate is within tolerance of the base rate.
    medium_stable_range / high_stable_range: the unbroken run of stable values along one threshold,
    holding the other at its base value. best_points: the settings with the highest exact rate.
    """
    base_medium: float
    base_high: float
    tolerance: float
    base_exact_rate: float
    points: list[ThresholdPoint]
    stable_share: float
    medium_stable_range: tuple[float, float]
    high_stable_range: tuple[float, float]
    best_exact_rate: float
    best_points: list[tuple[float, float]]


def threshold_grid(base: float, step: float = 0.025, half_width: float = 0.15) -> list[float]:
    """Values base + k * step for |k * step| <= half_width, inside [0, 1], so the base is always on the grid."""
    if step <= 0 or half_width < 0:
        raise ValueError("step must be positive and half_width non-negative")
    steps = round(half_width / step)
    values = (round(base + k * step, 9) for k in range(-steps, steps + 1))
    return [v for v in values if 0.0 <= v <= 1.0]


def _stable_run(values: Sequence[float], stable: Sequence[bool], base: float) -> tuple[float, float]:
    """The lowest and highest value of the unbroken run of stable values that contains base."""
    i = min(range(len(values)), key=lambda j: abs(values[j] - base))
    low = high = i
    while low > 0 and stable[low - 1]:
        low -= 1
    while high < len(values) - 1 and stable[high + 1]:
        high += 1
    return values[low], values[high]


def threshold_sweep(
    scores: Sequence[float],
    advisor_buckets: Sequence[str],
    base_medium: float,
    base_high: float,
    step: float = 0.025,
    half_width: float = 0.15,
    tolerance: float = 0.05,
) -> ThresholdSweep:
    """
    Re-bucket the model's composite scores at every (medium, high) grid setting with medium < high
    and compare with the advisor's buckets. Thresholds don't change composite scores, so no rescoring
    is needed. changed_vs_base counts clauses whose bucket differs from the base setting's.
    """
    if len(scores) != len(advisor_buckets) or not scores:
        raise ValueError("scores and advisor_buckets must be non-empty and the same length")
    base = [bucket_for(s, base_medium, base_high) for s in scores]
    base_rate = exact_rate(advisor_buckets, base)

    # Every grid setting where Medium sits below High
    mediums, highs = threshold_grid(base_medium, step, half_width), threshold_grid(base_high, step, half_width)
    points = []
    for medium in mediums:
        for high in highs:
            if medium >= high:
                continue
            buckets = [bucket_for(s, medium, high) for s in scores]
            points.append(ThresholdPoint(
                medium_threshold=medium, high_threshold=high, exact_rate=exact_rate(advisor_buckets, buckets),
                weighted_kappa=weighted_kappa(advisor_buckets, buckets),
                high_count=buckets.count("High"), medium_count=buckets.count("Medium"), low_count=buckets.count("Low"),
                changed_vs_base=sum(a != b for a, b in zip(buckets, base)),
            ))

    # Stability: within tolerance of the base rate, overall and along each axis through the base
    by_setting = {(p.medium_threshold, p.high_threshold): p for p in points}
    is_stable = {key: p.exact_rate >= base_rate - tolerance - 1e-12 for key, p in by_setting.items()}
    medium_axis = [m for m in mediums if (m, base_high) in by_setting]
    high_axis = [h for h in highs if (base_medium, h) in by_setting]
    best = max(p.exact_rate for p in points)
    return ThresholdSweep(
        base_medium=base_medium, base_high=base_high, tolerance=tolerance, base_exact_rate=base_rate, points=points,
        stable_share=sum(is_stable.values()) / len(points),
        medium_stable_range=_stable_run(medium_axis, [is_stable[m, base_high] for m in medium_axis], base_medium),
        high_stable_range=_stable_run(high_axis, [is_stable[base_medium, h] for h in high_axis], base_high),
        best_exact_rate=best,
        best_points=[(p.medium_threshold, p.high_threshold) for p in points if p.exact_rate == best],
    )


# ----- The full evaluation -----
class ItemComparison(BaseModel):
    """One judged clause, the model's score and rating beside the advisor's bucket, rank and notes."""
    item_id: str
    contract_id: str
    category: str
    composite_score: float
    rating: Rating
    advisor_bucket: Rating | None
    advisor_rank: int | None
    notes: str


class AgreementReport(BaseModel):
    """
    Agreement between the model's risk scores (under one weight version) and one advisor's judgements.
    spearman_rank uses clauses with an advisor rank; spearman_bucket uses clauses with a bucket.
    Bucket metrics use clauses with a bucket. scores_reproduced is set only when rescoring under the
    key's own weight set: False means the scoring code changed since the packet was generated.
    """
    packet_id: str
    reviewer: str
    ranking_set: str
    weight_version_id: str
    weight_set_sha256: str
    rescored: bool
    scores_reproduced: bool | None
    medium_threshold: float
    high_threshold: float
    n_items: int
    n_judged: int
    n_skipped: int
    n_with_rank: int
    n_with_bucket: int
    model_bucket_counts: dict[str, int]
    advisor_bucket_counts: dict[str, int]
    spearman_rank: MetricEstimate | None
    spearman_bucket: MetricEstimate | None
    exact_rate: MetricEstimate | None
    adjacent_rate: MetricEstimate | None
    weighted_kappa_linear: MetricEstimate | None
    weighted_kappa_quadratic: MetricEstimate | None
    high_precision: MetricEstimate | None
    high_recall: MetricEstimate | None
    confusion_matrix: list[list[int]] | None
    threshold_sweep: ThresholdSweep | None
    caveats: list[str]
    items: list[ItemComparison]


def _fmt(value: float | None) -> str:
    """A metric for a sentence: two decimals, or 'undefined'."""
    return "undefined" if value is None else f"{value:.2f}"


def _caveats(report: dict, n_boot: int, confidence: float) -> list[str]:
    """Plain-language statements about what this sample size does and doesn't support."""
    notes = [
        (f"{report['n_judged']} judged clauses. Every interval below is a {confidence:.0%} bootstrap interval "
         f"over clauses ({n_boot} resamples); read the interval, not the point estimate."),
        ("The packet is stratified to over-sample High and Medium clauses and spread categories, so these rates "
         "describe agreement on this sample, not the share of all clauses the model buckets correctly."),
    ]
    # Spearman against the 'strong' bar
    for name, label in (("spearman_rank", "Spearman (ranks)"), ("spearman_bucket", "Spearman (buckets)")):
        est = report[name]
        if est is None or est.value is None:
            continue
        if est.ci_low is None:
            notes.append(f"{label} = {_fmt(est.value)} has no interval; too few clauses to judge it.")
        elif est.ci_low >= STRONG_SPEARMAN:
            notes.append(f"{label} = {_fmt(est.value)}, interval [{_fmt(est.ci_low)}, {_fmt(est.ci_high)}]: "
                         f"the whole interval is at or above {STRONG_SPEARMAN}, so strong agreement is supported.")
        elif est.ci_low > 0:
            notes.append(f"{label} = {_fmt(est.value)}, interval [{_fmt(est.ci_low)}, {_fmt(est.ci_high)}]: "
                         f"a positive association is supported, but this sample can't confirm it reaches {STRONG_SPEARMAN} "
                         "(strong).")
        elif est.ci_high < 0:
            notes.append(f"{label} = {_fmt(est.value)}, interval [{_fmt(est.ci_low)}, {_fmt(est.ci_high)}]: "
                         "the whole interval is below 0, so the model tends to rank clauses opposite to the advisor.")
        else:
            notes.append(f"{label} = {_fmt(est.value)}, interval [{_fmt(est.ci_low)}, {_fmt(est.ci_high)}]: "
                         "the interval includes 0, so this sample can't rule out no association.")
    # Rates and kappa: flag wide intervals
    for name, label in (("exact_rate", "Exact bucket agreement"), ("weighted_kappa_linear", "Weighted kappa (linear)"),
                        ("high_precision", "High precision"), ("high_recall", "High recall")):
        est = report[name]
        if est is not None and est.ci_width is not None and est.ci_width > WIDE_INTERVAL:
            notes.append(f"{label} = {_fmt(est.value)} with interval [{_fmt(est.ci_low)}, {_fmt(est.ci_high)}] "
                         f"(width {est.ci_width:.2f}): too wide to support a conclusion on its own.")
        elif est is not None and est.value is None:
            notes.append(f"{label} is undefined on this sample (no clauses in its denominator).")
    sweep = report["threshold_sweep"]
    if sweep is not None:
        notes.append("The threshold sweep shows sensitivity, not a tuning result: picking the best thresholds on "
                     f"these {report['n_with_bucket']} clauses would overfit them, and one clause moves the exact "
                     f"rate by {1 / report['n_with_bucket']:.3f}.")
    return notes


def evaluate_agreement(
    returned: ReturnedPacket,
    key: PacketKey,
    weight_set: WeightSet | None = None,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
    sweep_step: float = 0.025,
    sweep_half_width: float = 0.15,
    sweep_tolerance: float = 0.05,
) -> AgreementReport:
    """
    Compare the advisor's judgements with the model's scores and return every metric with its interval.

    weight_set: None uses the scores stored in the key (the weight version the packet was built with);
    a WeightSet rescores every clause from the key without a model, and its thresholds set the buckets.
    """
    if returned.packet_id != key.packet_id:
        raise ValueError(f"returned packet {returned.packet_id!r} doesn't belong to key {key.packet_id!r}")
    by_id = {item.item_id: item for item in key.items}

    # Model scores: stored in the key, or rescored from it
    if weight_set is None:
        scores = {i.item_id: (i.composite_score, i.rating) for i in key.items}
        version, sha, medium, high, reproduced = (key.weight_version_id, key.weight_set_sha256, key.medium_threshold,
                                                  key.high_threshold, None)
    else:
        rescored = rescore_key(key, weight_set)
        scores = {item_id: (s.composite_score, s.rating) for item_id, s in rescored.items()}
        version, sha = weight_set.weight_version_id, weight_set.content_sha256
        medium, high = weight_set.medium_threshold, weight_set.high_threshold
        reproduced = None
        if sha == key.weight_set_sha256:
            reproduced = all(math.isclose(scores[i.item_id][0], i.composite_score, abs_tol=1e-9)
                             and scores[i.item_id][1] == i.rating for i in key.items)

    # One comparison row per judged clause
    strangers = [j.item_id for j in returned.judged if j.item_id not in by_id]
    if strangers:
        raise ValueError(f"judged items not in the key: {strangers}")
    items = [ItemComparison(
        item_id=j.item_id, contract_id=j.ranking.contract_id, category=j.ranking.category,
        composite_score=scores[j.item_id][0], rating=scores[j.item_id][1], advisor_bucket=j.ranking.advisor_bucket,
        advisor_rank=j.ranking.advisor_rank, notes=j.notes,
    ) for j in returned.judged]
    ranked = [i for i in items if i.advisor_rank is not None]
    bucketed = [i for i in items if i.advisor_bucket is not None]

    def estimate(statistic: Callable, columns: Sequence[Sequence]) -> MetricEstimate | None:
        """Bootstrap one metric, or None when it has no clauses to use."""
        if not columns[0]:
            return None
        return bootstrap_estimate(statistic, columns, n_boot=n_boot, seed=seed, confidence=confidence)

    # Spearman: model score against advisor riskiness (minus the rank, or the bucket's order)
    spearman_rank = estimate(spearman, [[i.composite_score for i in ranked], [-i.advisor_rank for i in ranked]])
    spearman_bucket = estimate(spearman, [[i.composite_score for i in bucketed],
                                          [BUCKET_ORDINAL[i.advisor_bucket] for i in bucketed]])

    # Bucket agreement, High precision / recall, and the threshold sweep
    advisor_b, model_b = [i.advisor_bucket for i in bucketed], [i.rating for i in bucketed]
    columns = [advisor_b, model_b]
    report = {
        "packet_id": key.packet_id, "reviewer": returned.judged[0].ranking.reviewer,
        "ranking_set": returned.judged[0].ranking.ranking_set, "weight_version_id": version,
        "weight_set_sha256": sha, "rescored": weight_set is not None, "scores_reproduced": reproduced,
        "medium_threshold": medium, "high_threshold": high, "n_items": len(key.items), "n_judged": len(items),
        "n_skipped": len(returned.skipped), "n_with_rank": len(ranked), "n_with_bucket": len(bucketed),
        "model_bucket_counts": {b: sum(i.rating == b for i in items) for b in BUCKETS},
        "advisor_bucket_counts": {b: sum(i.advisor_bucket == b for i in items) for b in BUCKETS},
        "spearman_rank": spearman_rank, "spearman_bucket": spearman_bucket,
        "exact_rate": estimate(exact_rate, columns), "adjacent_rate": estimate(adjacent_rate, columns),
        "weighted_kappa_linear": estimate(weighted_kappa, columns),
        "weighted_kappa_quadratic": estimate(lambda a, m: weighted_kappa(a, m, "quadratic"), columns),
        "high_precision": estimate(high_precision, columns), "high_recall": estimate(high_recall, columns),
        "confusion_matrix": confusion_matrix(advisor_b, model_b) if bucketed else None,
        "threshold_sweep": threshold_sweep([i.composite_score for i in bucketed], advisor_b, medium, high,
                                           sweep_step, sweep_half_width, sweep_tolerance) if bucketed else None,
        "items": items,
    }
    report["caveats"] = _caveats(report, n_boot, confidence)
    return AgreementReport(**report)


def write_key(key: PacketKey, path: str | Path) -> None:
    """Write a packet key as indented JSON (UTF-8)."""
    Path(path).write_text(json.dumps(key.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
