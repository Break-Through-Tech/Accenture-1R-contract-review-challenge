"""Tests for clausetriage/review/feedback.py: verdict validation, append-only history, changed and conflicting verdicts."""
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from clausetriage.review.feedback import list_resolutions, resolve_feedback, submit_feedback
from clausetriage.store.records import FeedbackRecord
from tests import review_fixtures as fixtures
from tests.review_fixtures import T0, verdict
from tests.test_artifact_store import MODEL_VERSION

# Bind the shared fixtures under their own names (see tests/test_store_rescoring.py)
make_pipeline = fixtures.make_pipeline
store = fixtures.store
review_store = fixtures.review_store


def record(id: int, reviewer: str, verdict: str, chunk_id: str = "c-1:3", category: str = "Insurance",
           model_version: str = MODEL_VERSION) -> FeedbackRecord:
    """A FeedbackRecord built by hand for the pure resolution tests, timed id minutes after T0."""
    return FeedbackRecord(id=id, contract_id=chunk_id.split(":")[0], chunk_id=chunk_id, category=category,
                          model_version=model_version, verdict=verdict, reviewer=reviewer,
                          timestamp=T0 + timedelta(minutes=id))


# ----- Valid verdicts -----
@pytest.mark.parametrize("chunk_id,category,given", [
    ("c-1:3", "Insurance", "accept"),               # an Unknown
    ("c-1:3", "Insurance", "reject"),
    ("c-1:0", "Governing Law", "accept"),           # a grounded prediction
    ("c-1:0", "Governing Law", "reject"),
    ("c-1:2", "Insurance", "missing"),              # no prediction at all
    ("c-1:0", "Termination For Convenience", "missing"),  # above threshold but dropped as a duplicate
])
def test_submit_feedback_valid_verdict_is_stored(review_store, chunk_id, category, given):
    """Each verdict is accepted on what it can refer to and stored with its contract and UTC timestamp."""
    stored = verdict(review_store, chunk_id, category, given, "ana")
    assert (stored.contract_id, stored.chunk_id, stored.category, stored.verdict, stored.reviewer) == \
        ("c-1", chunk_id, category, given, "ana")
    assert stored.timestamp == T0
    assert review_store.list_feedback() == [stored]


def test_submit_feedback_reject_contradicts_a_claimed_miss(review_store):
    """After ana flags a missed Insurance clause in c-1:2, ben may reject it (and accept is still refused)."""
    verdict(review_store, "c-1:2", "Insurance", "missing", "ana")
    rejected = verdict(review_store, "c-1:2", "Insurance", "reject", "ben", minute=1)
    assert rejected.verdict == "reject"
    with pytest.raises(ValueError, match="needs a prediction"):
        verdict(review_store, "c-1:2", "Insurance", "accept", "ben", minute=2)


def test_submit_feedback_strips_reviewer_and_converts_to_utc(review_store):
    """' ana ' is stored as 'ana', and 11:00+02:00 as 09:00 UTC (SQLite would otherwise drop the offset)."""
    stored = submit_feedback(review_store, dict(
        chunk_id="c-1:3", category="Insurance", model_version=MODEL_VERSION, verdict="accept", reviewer=" ana ",
        timestamp=datetime(2026, 11, 1, 11, 0, tzinfo=timezone(timedelta(hours=2)))))
    (listed,) = review_store.list_feedback()
    assert (stored.reviewer, listed.reviewer) == ("ana", "ana")
    assert stored.timestamp == listed.timestamp == T0
    assert listed.timestamp.tzinfo is not None


def test_submit_feedback_default_timestamp_is_now(review_store):
    """Without a timestamp the store records the current UTC time."""
    before = datetime.now(timezone.utc)
    stored = submit_feedback(review_store, dict(chunk_id="c-1:3", category="Insurance", model_version=MODEL_VERSION,
                                                verdict="accept", reviewer="ana"))
    assert before <= stored.timestamp <= datetime.now(timezone.utc)


# ----- Refused verdicts -----
@pytest.mark.parametrize("chunk_id,category,given,model_version,error,match", [
    ("c-1:2", "Insurance", "accept", MODEL_VERSION, ValueError, "needs a prediction"),
    ("c-1:2", "Insurance", "reject", MODEL_VERSION, ValueError, "needs a prediction"),
    ("c-1:3", "Insurance", "missing", MODEL_VERSION, ValueError, "predicted 'Insurance'.*unknown"),
    ("c-1:0", "Governing Law", "missing", MODEL_VERSION, ValueError, "predicted 'Governing Law'.*grounded"),
    ("c-1:3", "Insurance", "accept", "other-model:1", ValueError, "no stored predictions"),
    ("c-1:3", "Insurance", "missing", "other-model:1", ValueError, "no stored predictions"),
    ("c-9:0", "Insurance", "missing", MODEL_VERSION, KeyError, "unknown chunk"),
])
def test_submit_feedback_on_nonexistent_target_raises(review_store, chunk_id, category, given, model_version,
                                                      error, match):
    """Verdicts that don't fit what the model predicted, or point at nothing, are refused and not stored."""
    with pytest.raises(error, match=match):
        verdict(review_store, chunk_id, category, given, "ana", model_version=model_version)
    assert review_store.list_feedback() == []


@pytest.mark.parametrize("changes", [
    {"verdict": "maybe"},
    {"category": "insurance"},
    {"reviewer": "   "},
    {"timestamp": datetime(2026, 11, 1, 9, 0)},
    {"chunk_id": ""},
])
def test_submit_feedback_malformed_submission_raises(review_store, changes):
    """Bad verdicts, non-canonical categories, blank reviewers, naive timestamps and empty ids fail validation."""
    values = dict(chunk_id="c-1:3", category="Insurance", model_version=MODEL_VERSION, verdict="accept",
                  reviewer="ana", timestamp=T0)
    with pytest.raises(ValidationError):
        submit_feedback(review_store, {**values, **changes})
    assert review_store.list_feedback() == []


def test_submit_feedback_earlier_than_own_last_verdict_raises(review_store):
    """A reviewer's verdicts on one key can't go back in time; another reviewer is unaffected."""
    verdict(review_store, "c-1:3", "Insurance", "accept", "ana", minute=10)
    with pytest.raises(ValueError, match="earlier than 'ana'"):
        verdict(review_store, "c-1:3", "Insurance", "reject", "ana", minute=5)
    verdict(review_store, "c-1:3", "Insurance", "reject", "ben", minute=5)
    assert [(f.reviewer, f.verdict) for f in review_store.list_feedback()] == [("ana", "accept"), ("ben", "reject")]


# ----- Changed verdicts -----
def test_changed_verdict_latest_counts_and_history_is_kept(review_store):
    """ana accepts then rejects: both rows stay, her current verdict is reject, so the label is 0."""
    first = verdict(review_store, "c-1:3", "Insurance", "accept", "ana", minute=0)
    second = verdict(review_store, "c-1:3", "Insurance", "reject", "ana", minute=5)
    (resolution,) = list_resolutions(review_store)
    assert (resolution.status, resolution.label) == ("resolved", 0)
    assert [v.feedback_id for v in resolution.current] == [second.id]
    assert [v.feedback_id for v in resolution.history] == [first.id, second.id]


def test_changed_verdict_with_equal_timestamps_uses_log_order(review_store):
    """Two verdicts at the same instant: the one written last is current."""
    verdict(review_store, "c-1:3", "Insurance", "reject", "ana")
    verdict(review_store, "c-1:3", "Insurance", "accept", "ana")
    (resolution,) = list_resolutions(review_store)
    assert (resolution.label, resolution.current[0].verdict) == (1, "accept")


# ----- Conflicts -----
def test_conflicting_reviewers_are_reported_not_settled(review_store):
    """ana accepts and ben rejects c-2:0 Insurance: a conflict with no label and both current verdicts."""
    verdict(review_store, "c-2:0", "Insurance", "accept", "ana")
    verdict(review_store, "c-2:0", "Insurance", "reject", "ben", minute=1)
    (resolution,) = list_resolutions(review_store, status="conflict")
    assert (resolution.chunk_id, resolution.status, resolution.label) == ("c-2:0", "conflict", None)
    assert [(v.reviewer, v.verdict) for v in resolution.current] == [("ana", "accept"), ("ben", "reject")]
    assert resolution.reviewers == ["ana", "ben"]


def test_conflict_is_not_outvoted(review_store):
    """Two accepts against one reject is still a conflict: no majority rule."""
    for minute, (reviewer, given) in enumerate([("ana", "accept"), ("cy", "accept"), ("ben", "reject")]):
        verdict(review_store, "c-2:0", "Insurance", given, reviewer, minute=minute)
    (resolution,) = list_resolutions(review_store)
    assert resolution.status == "conflict"


def test_conflict_resolves_when_a_reviewer_changes_their_mind(review_store):
    """ben's later accept replaces his reject, and the key resolves to 1."""
    verdict(review_store, "c-2:0", "Insurance", "accept", "ana")
    verdict(review_store, "c-2:0", "Insurance", "reject", "ben", minute=1)
    verdict(review_store, "c-2:0", "Insurance", "accept", "ben", minute=2)
    (resolution,) = list_resolutions(review_store)
    assert (resolution.status, resolution.label, len(resolution.history)) == ("resolved", 1, 3)


def test_missing_claim_contradicted_is_a_conflict(review_store):
    """ana says Insurance is missing from c-1:2, ben rejects that: conflict on a key the model never predicted."""
    verdict(review_store, "c-1:2", "Insurance", "missing", "ana")
    verdict(review_store, "c-1:2", "Insurance", "reject", "ben", minute=1)
    (resolution,) = list_resolutions(review_store)
    assert (resolution.chunk_id, resolution.status) == ("c-1:2", "conflict")


# ----- Pure resolution -----
@pytest.mark.parametrize("verdicts,status,label", [
    ([("ana", "accept")], "resolved", 1),
    ([("ana", "reject")], "resolved", 0),
    ([("ana", "missing")], "resolved", 1),
    ([("ana", "accept"), ("ben", "missing")], "resolved", 1),
    ([("ana", "accept"), ("ben", "reject")], "conflict", None),
    ([("ana", "accept"), ("ana", "reject")], "resolved", 0),
    ([("ana", "reject"), ("ben", "accept"), ("ana", "accept")], "resolved", 1),
])
def test_resolve_feedback_golden_outcomes(verdicts, status, label):
    """Hand-worked outcomes: latest verdict per reviewer, accept/missing = 1, reject = 0, disagreement = conflict."""
    records = [record(i, reviewer, given) for i, (reviewer, given) in enumerate(verdicts, start=1)]
    (resolution,) = resolve_feedback(records)
    assert (resolution.status, resolution.label) == (status, label)


def test_resolve_feedback_uses_ids_not_input_order():
    """Records passed out of order resolve the same as in log order."""
    records = [record(2, "ana", "reject"), record(1, "ana", "accept")]
    (resolution,) = resolve_feedback(records)
    assert (resolution.label, [v.feedback_id for v in resolution.history]) == (0, [1, 2])


def test_resolve_feedback_groups_by_chunk_and_category_in_order():
    """Keys come back by contract, chunk id and category column; model versions of one key resolve together."""
    records = [record(1, "ana", "accept", chunk_id="c-2:0"),
               record(2, "ana", "accept", category="Governing Law"),
               record(3, "ana", "reject", model_version="m:2"),
               record(4, "ben", "accept")]
    resolutions = resolve_feedback(records)
    assert [(r.chunk_id, r.category, r.status) for r in resolutions] == [
        ("c-1:3", "Governing Law", "resolved"), ("c-1:3", "Insurance", "conflict"), ("c-2:0", "Insurance", "resolved")]


def test_list_resolutions_filters_by_model_version(review_store):
    """Feedback under another model version is out of scope when a version is given."""
    verdict(review_store, "c-1:3", "Insurance", "accept", "ana")
    assert list_resolutions(review_store, model_version="other-model:1") == []
    assert len(list_resolutions(review_store, model_version=MODEL_VERSION, contract_id="c-1")) == 1
    assert list_resolutions(review_store, contract_id="c-2") == []
