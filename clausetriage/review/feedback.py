"""
Reviewer feedback: the validated write path and how repeated or conflicting verdicts resolve.

Verdicts, on one (chunk, category) under the model version the reviewer looked at:
    accept   the model predicted the category in this chunk and it is there      -> label 1
    reject   the category is not in this chunk, though the model predicted it or
             another reviewer flagged it as missing                              -> label 0
    missing  the model did not predict the category in this chunk, but it is there -> label 1
"Predicted" means a stored detection: grounded evidence or an Unknown. A second reviewer who agrees
with a "missing" also says missing; one who disagrees says reject. accept without a detection, reject
with neither a detection nor a claimed miss, missing on a detection, unknown chunks, chunks never run
under the model version and non-canonical categories are all refused by ArtifactStore.add_feedback,
so the API can't bypass them.

Resolution: history is append-only. Each reviewer's current verdict on a (chunk, category) is their
latest one in log order (the store keeps each reviewer's timestamps non-decreasing per key, so this is
also the latest in time). Current verdicts that imply the same label resolve to it; different labels
are a conflict, reported with every reviewer's verdict and never settled by majority or seniority.
A later verdict from one of the disagreeing reviewers can resolve it.

How everything comes together:
submit_feedback():   validate a FeedbackSubmission and append it through the store
resolve_feedback():  pure: group FeedbackRecords by (chunk, category) and resolve each group
list_resolutions():  resolve the stored feedback in scope (model version, contract, status)
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from clausetriage.labels.categories import CATEGORY_TO_INDEX
from clausetriage.review.schema import VERDICT_LABEL, FeedbackSubmission, Resolution, ResolutionStatus, ReviewerVerdict
from clausetriage.store.artifact_store import ArtifactStore
from clausetriage.store.records import FeedbackRecord


def submit_feedback(store: ArtifactStore, submission: FeedbackSubmission | Mapping[str, Any]) -> FeedbackRecord:
    """
    Validate a verdict and append it to the feedback log; returns the stored record.
    Raises pydantic.ValidationError for a malformed submission, KeyError for an unknown chunk and
    ValueError when the verdict does not fit what the model predicted (see the module docstring).
    """
    submission = FeedbackSubmission.model_validate(submission)
    return store.add_feedback(**submission.model_dump())


def _vote(record: FeedbackRecord) -> ReviewerVerdict:
    """The provenance fields of one stored verdict."""
    return ReviewerVerdict(feedback_id=record.id, reviewer=record.reviewer, verdict=record.verdict,
                           model_version=record.model_version, timestamp=record.timestamp, note=record.note)


def resolve_feedback(records: Iterable[FeedbackRecord]) -> list[Resolution]:
    """
    Resolve every (chunk, category) in records: each reviewer's latest verdict (highest id) counts,
    and the key is resolved when those verdicts all imply one label, a conflict otherwise.
    Returned in (contract_id, chunk_id, category column) order; current verdicts by reviewer name.
    """
    # Group the log by key, keeping log order inside each group
    groups: dict[tuple[str, str, str], list[FeedbackRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda r: r.id):
        groups[record.contract_id, record.chunk_id, record.category].append(record)

    resolutions = []
    for (contract_id, chunk_id, category), group in groups.items():
        # A later verdict from the same reviewer replaces their earlier one
        latest = {record.reviewer: record for record in group}
        current = [_vote(latest[reviewer]) for reviewer in sorted(latest)]
        labels = {VERDICT_LABEL[vote.verdict] for vote in current}
        status: ResolutionStatus = "resolved" if len(labels) == 1 else "conflict"
        resolutions.append(Resolution(
            contract_id=contract_id, chunk_id=chunk_id, category=category, status=status,
            label=labels.pop() if status == "resolved" else None,
            current=current, history=[_vote(record) for record in group]))
    return sorted(resolutions, key=lambda r: (r.contract_id, r.chunk_id, CATEGORY_TO_INDEX[r.category]))


def list_resolutions(store: ArtifactStore, *, model_version: str | None = None, contract_id: str | None = None,
                     status: ResolutionStatus | None = None) -> list[Resolution]:
    """
    Resolve the stored feedback in scope. With model_version, only verdicts given under that version
    count; without it, verdicts on the same (chunk, category) under any version are resolved together,
    since the label is a fact about the chunk text. status="conflict" lists the disagreements to settle.
    """
    resolutions = resolve_feedback(store.list_feedback(contract_id=contract_id, model_version=model_version))
    return [r for r in resolutions if status is None or r.status == status]
