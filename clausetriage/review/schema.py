"""
Pydantic models at the boundary of the human-in-the-loop layer: what the review queue returns, what a
reviewer submits, how repeated verdicts resolve, and what the training-data export writes.

All of them serialise to JSON with model_dump(mode="json"), so the API layer can return them as is.
Timestamps are timezone-aware UTC.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

from clausetriage.labels.categories import CATEGORY_TO_INDEX
from clausetriage.store.records import DetectionKind, Verdict

# Queue status of one item: nobody has reviewed it, reviewers agree on its label, or they disagree
ReviewStatus = Literal["pending", "reviewed", "conflict"]
# Outcome of resolving every reviewer's current verdict on one (chunk, category)
ResolutionStatus = Literal["resolved", "conflict"]
# Why resolved feedback was left out of a training export
ExclusionReason = Literal["conflict", "evaluation_contract", "not_in_training_manifest", "offset_mismatch"]

# The label each verdict asserts for its (chunk, category): accept and missing say present, reject says absent
VERDICT_LABEL: dict[str, int] = {"accept": 1, "missing": 1, "reject": 0}


def _check_category(category: str) -> str:
    """Reject anything that is not an exact canonical CUAD category name."""
    if category not in CATEGORY_TO_INDEX:
        raise ValueError(f"unknown CUAD category {category!r}")
    return category


# ----- Feedback -----
class FeedbackSubmission(BaseModel):
    """
    One verdict a reviewer submits on a (chunk, category) under the model version they reviewed.
    The reviewer name is stripped; timestamp, when given, must be timezone-aware (None means now).
    Whether the verdict fits what the model predicted is checked against the store on submit.
    """
    chunk_id: str = Field(min_length=1)
    category: str
    model_version: str = Field(min_length=1)
    verdict: Verdict
    reviewer: str
    note: str | None = None
    timestamp: datetime | None = None

    # Names and categories must be exact; a timestamp without an offset is ambiguous
    @field_validator("reviewer")
    @classmethod
    def _strip_reviewer(cls, value: str) -> str:
        """Strip the reviewer name and reject an empty one."""
        if not value.strip():
            raise ValueError("reviewer must be a non-empty string")
        return value.strip()

    @field_validator("category")
    @classmethod
    def _canonical_category(cls, value: str) -> str:
        """Only the 41 canonical CUAD category names are accepted."""
        return _check_category(value)

    @field_validator("timestamp")
    @classmethod
    def _aware_timestamp(cls, value: datetime | None) -> datetime | None:
        """Require an explicit UTC offset and convert to UTC."""
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)


class ReviewerVerdict(BaseModel):
    """One stored verdict, as shown in a resolution or an export row."""
    feedback_id: int
    reviewer: str
    verdict: Verdict
    model_version: str
    timestamp: datetime
    note: str | None = None


class Resolution(BaseModel):
    """
    Every reviewer's current verdict on one (chunk, category) and what they add up to. A reviewer's
    current verdict is their latest one (log order); earlier ones stay in history. Reviewers whose
    current verdicts imply the same label resolve to it; different labels are a conflict with label None.
    """
    contract_id: str
    chunk_id: str
    category: str
    status: ResolutionStatus
    label: Literal[0, 1] | None
    current: list[ReviewerVerdict]
    history: list[ReviewerVerdict]

    @property
    def reviewers(self) -> list[str]:
        """Reviewers with a current verdict, sorted."""
        return [vote.reviewer for vote in self.current]


# ----- Review queue -----
class QueueCandidate(BaseModel):
    """
    One (chunk, category) a queue source says needs a human, before the queue adds chunk text,
    pages and review status. source names the producer ("unknown", "conflict"; later e.g. "uncertainty").
    """
    source: str
    contract_id: str
    chunk_id: str
    chunk_index: int
    category: str
    probability: float
    threshold: float | None = None
    detection: DetectionKind | None
    reason: str
    reason_text: str


class ReviewItem(QueueCandidate):
    """
    Everything a reviewer needs to decide one queue item without opening anything else.
    chunk_start/chunk_end are absolute offsets (contract text[chunk_start:chunk_end] == chunk_text);
    first_page/last_page are 1-based and None when the contract has no page offsets. evidence_* are
    set only for a grounded prediction. allowed_verdicts lists what this item accepts: accept/reject
    for a prediction; missing/reject for a category the model did not predict but a reviewer flagged.
    """
    item_id: str
    model_version: str
    source_name: str | None = None
    chunk_start: int
    chunk_end: int
    chunk_text: str
    first_page: int | None = None
    last_page: int | None = None
    evidence_quote: str | None = None
    evidence_start: int | None = None
    evidence_end: int | None = None
    allowed_verdicts: list[Verdict]
    status: ReviewStatus
    resolved_label: Literal[0, 1] | None = None
    current_verdicts: list[ReviewerVerdict] = Field(default_factory=list)


class ReviewQueuePage(BaseModel):
    """
    One page of the review queue. total counts items after the status filter; status_counts counts
    every item before it. Items are in queue order (see clausetriage/review/queue.py).
    """
    model_version: str
    contract_id: str | None = None
    status: ReviewStatus | None = None
    offset: int
    limit: int
    total: int
    status_counts: dict[str, int]
    items: list[ReviewItem]


# ----- Training-data export -----
class TrainingManifest(BaseModel):
    """
    Which contracts may contribute training examples, identified by id and by the SHA-256 of their
    exact text. evaluation lists held-out contracts that must never be exported, even under another id.
    """
    source: str
    source_sha256: str | None = None
    split_method: str
    seed: int | None = None
    val_fraction: float | None = None
    training: dict[str, str]
    evaluation: dict[str, str]

    # A contract can't be both trainable and held out, by id or by content
    @model_validator(mode="after")
    def _disjoint(self) -> Self:
        """Training and evaluation must share no contract id and no text hash."""
        shared_ids = set(self.training) & set(self.evaluation)
        shared_text = set(self.training.values()) & set(self.evaluation.values())
        if shared_ids or shared_text:
            raise ValueError(f"training and evaluation overlap ({len(shared_ids)} ids, {len(shared_text)} texts)")
        return self

    @property
    def manifest_sha256(self) -> str:
        """SHA-256 of the manifest's canonical JSON, recorded with every export."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TrainingExample(BaseModel):
    """
    One labelled chunk example from resolved feedback: chunk text at absolute contract offsets
    (contract text[start:end] == chunk_text), one category and its 0/1 label, plus provenance.
    contract_sha256 identifies the contract text, so a training run can check it against its
    evaluation contracts. votes are the reviewers' current verdicts that produced the label.
    """
    contract_id: str
    contract_sha256: str
    chunk_id: str
    chunk_index: int
    start: int
    end: int
    chunk_text: str
    category: str
    category_index: int
    label: Literal[0, 1]
    verdicts: list[Verdict]
    reviewers: list[str]
    model_versions: list[str]
    first_reviewed_at: datetime
    last_reviewed_at: datetime
    votes: list[ReviewerVerdict]


class ExcludedFeedback(BaseModel):
    """A (chunk, category) with feedback that the export left out, and why."""
    contract_id: str
    chunk_id: str
    category: str
    reason: ExclusionReason
    detail: str
    votes: list[ReviewerVerdict]


class FeedbackExport(BaseModel):
    """
    A training-data export: examples in (contract, chunk, category) order, the feedback left out and
    why, and the manifest and filters that produced it. test_data_used is always False.
    """
    model_version: str | None = None
    manifest_source: str
    manifest_sha256: str
    split_method: str
    seed: int | None = None
    val_fraction: float | None = None
    test_data_used: Literal[False] = False
    example_count: int
    positive_count: int
    excluded_counts: dict[str, int]
    examples: list[TrainingExample]
    excluded: list[ExcludedFeedback]
