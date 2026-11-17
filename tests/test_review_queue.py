"""Tests for src/review/queue.py: queue contents, ordering, status and paging."""
import json

import pytest

from src.review.queue import build_review_queue, unknown_candidates
from src.review.schema import QueueCandidate, ReviewQueuePage
from tests import review_fixtures as fixtures
from tests.review_fixtures import TEXT_3, verdict
from tests.test_artifact_store import MODEL_VERSION, TEXT_1

# Bind the shared fixtures under their own names (see tests/test_store_rescoring.py)
make_pipeline = fixtures.make_pipeline
store = fixtures.store
review_store = fixtures.review_store


def keys(page: ReviewQueuePage) -> list[tuple[str, str, str]]:
    """(chunk_id, category, status) of every item on a page, in order."""
    return [(item.chunk_id, item.category, item.status) for item in page.items]


# ----- Contents -----
def test_queue_lists_every_unknown_highest_probability_first(review_store):
    """Three Unknowns, all pending: c-3 (p 0.95) first, then the two 0.7 Insurance items by contract."""
    page = build_review_queue(review_store, MODEL_VERSION)
    assert keys(page) == [("c-3:0", "Uncapped Liability", "pending"), ("c-1:3", "Insurance", "pending"),
                          ("c-2:0", "Insurance", "pending")]
    assert (page.total, page.status_counts) == (3, {"conflict": 0, "pending": 3, "reviewed": 0})


def test_queue_item_carries_what_a_reviewer_needs(review_store):
    """c-1:3 Insurance holds its chunk text at [120:167], probability, threshold, reason and allowed verdicts."""
    item = build_review_queue(review_store, MODEL_VERSION, contract_id="c-1").items[0]
    assert (item.chunk_start, item.chunk_end, item.chunk_text) == (120, 167, TEXT_1[120:167])
    assert item.chunk_text == " The supplier shall maintain adequate coverage."
    assert (item.source, item.detection, item.reason, item.threshold) == ("unknown", "unknown", "no_cue_match", 0.5)
    assert item.probability == pytest.approx(0.7)
    assert "cue words" in item.reason_text
    assert item.allowed_verdicts == ["accept", "reject"]
    assert item.item_id == f"c-1:3|Insurance|{MODEL_VERSION}"
    assert (item.first_page, item.last_page, item.source_name, item.evidence_quote) == (None, None, None, None)


def test_queue_item_pages_come_from_page_offsets(review_store):
    """c-3's only chunk [0:40] starts on page 1 and ends on page 2 (page_starts [0, 20])."""
    item = build_review_queue(review_store, MODEL_VERSION, contract_id="c-3").items[0]
    assert (item.chunk_text, item.first_page, item.last_page, item.source_name) == (TEXT_3, 1, 2, "msa.pdf")


def test_queue_contract_filter_and_unknown_model(review_store):
    """A contract filter keeps only its items; an unknown model version or contract raises KeyError."""
    assert keys(build_review_queue(review_store, MODEL_VERSION, contract_id="c-2")) == [
        ("c-2:0", "Insurance", "pending")]
    assert build_review_queue(review_store, MODEL_VERSION, contract_id="c-empty").items == []
    with pytest.raises(KeyError):
        build_review_queue(review_store, "other-model:1")
    with pytest.raises(KeyError):
        build_review_queue(review_store, MODEL_VERSION, contract_id="c-9")


# ----- Status and ordering -----
def test_queue_status_follows_feedback_and_orders_conflict_pending_reviewed(review_store):
    """ana accepts c-1:3 (reviewed); ana and ben split on c-2:0 (conflict): conflict, pending, reviewed."""
    verdict(review_store, "c-1:3", "Insurance", "accept", "ana")
    verdict(review_store, "c-2:0", "Insurance", "accept", "ana")
    verdict(review_store, "c-2:0", "Insurance", "reject", "ben", minute=1)
    page = build_review_queue(review_store, MODEL_VERSION)
    assert keys(page) == [("c-2:0", "Insurance", "conflict"), ("c-3:0", "Uncapped Liability", "pending"),
                          ("c-1:3", "Insurance", "reviewed")]
    reviewed = page.items[2]
    assert (reviewed.resolved_label, [v.reviewer for v in reviewed.current_verdicts]) == (1, ["ana"])
    assert page.items[0].resolved_label is None
    assert page.status_counts == {"conflict": 1, "pending": 1, "reviewed": 1}


def test_queue_status_only_counts_feedback_under_its_model_version(review_store, make_pipeline):
    """Feedback given under another model version leaves this version's item pending."""
    review_store.put_probabilities("c-1:3", "other-model:1", ["Insurance"], [0.1])
    verdict(review_store, "c-1:3", "Insurance", "missing", "ana", model_version="other-model:1")
    assert keys(build_review_queue(review_store, MODEL_VERSION, contract_id="c-1")) == [
        ("c-1:3", "Insurance", "pending")]


def test_queue_adds_conflicts_on_grounded_predictions_with_evidence(review_store):
    """A split verdict on c-1:0's grounded Governing Law joins the queue first, showing its quote."""
    verdict(review_store, "c-1:0", "Governing Law", "accept", "ana")
    verdict(review_store, "c-1:0", "Governing Law", "reject", "ben", minute=1)
    item = build_review_queue(review_store, MODEL_VERSION).items[0]
    assert (item.chunk_id, item.category, item.status, item.source, item.detection) == \
        ("c-1:0", "Governing Law", "conflict", "conflict", "grounded")
    assert (item.evidence_quote, item.evidence_start, item.evidence_end) == \
        ("This Agreement is governed by Delaware law.", 0, 43)
    assert item.probability == pytest.approx(0.9)
    assert item.reason == "reviewer_conflict"


def test_queue_adds_conflicts_on_claimed_misses(review_store):
    """A contested 'missing' on c-1:2 Insurance is queued with no detection and missing/reject allowed."""
    verdict(review_store, "c-1:2", "Insurance", "missing", "ana")
    verdict(review_store, "c-1:2", "Insurance", "reject", "ben", minute=1)
    item = build_review_queue(review_store, MODEL_VERSION, status="conflict").items[0]
    assert (item.chunk_id, item.detection, item.allowed_verdicts, item.probability) == \
        ("c-1:2", None, ["missing", "reject"], 0.0)


def test_queue_unknown_conflict_listed_once_from_unknown_source(review_store):
    """A conflicted Unknown appears once, attributed to the first source (unknown)."""
    verdict(review_store, "c-2:0", "Insurance", "accept", "ana")
    verdict(review_store, "c-2:0", "Insurance", "reject", "ben", minute=1)
    page = build_review_queue(review_store, MODEL_VERSION)
    matching = [item for item in page.items if item.chunk_id == "c-2:0"]
    assert [(item.source, item.status) for item in matching] == [("unknown", "conflict")]


def test_queue_accepts_another_source(review_store):
    """A custom source (here a fake 'uncertainty' item) is merged into the same queue and ordering."""
    def near_threshold(store, model_version, contract_id):
        """One made-up candidate just below threshold in c-1:2."""
        return [QueueCandidate(source="uncertainty", contract_id="c-1", chunk_id="c-1:2", chunk_index=2,
                               category="Insurance", probability=0.45, threshold=0.5, detection=None,
                               reason="near_threshold", reason_text="Just below threshold.")]
    page = build_review_queue(review_store, MODEL_VERSION, sources=[unknown_candidates, near_threshold])
    assert keys(page)[-1] == ("c-1:2", "Insurance", "pending")
    assert page.items[-1].source == "uncertainty"


# ----- Paging -----
@pytest.mark.parametrize("offset,limit,expected", [
    (0, 2, ["c-3:0", "c-1:3"]),
    (2, 2, ["c-2:0"]),
    (3, 2, []),
    (1, 1, ["c-1:3"]),
])
def test_queue_pages_slice_the_stable_order(review_store, offset, limit, expected):
    """Pages are consecutive slices of the full order; total stays 3."""
    page = build_review_queue(review_store, MODEL_VERSION, offset=offset, limit=limit)
    assert ([item.chunk_id for item in page.items], page.total) == (expected, 3)


def test_queue_status_filter_counts_before_and_after(review_store):
    """status='pending' leaves two items; status_counts still counts all three."""
    verdict(review_store, "c-1:3", "Insurance", "accept", "ana")
    page = build_review_queue(review_store, MODEL_VERSION, status="pending")
    assert ([item.chunk_id for item in page.items], page.total) == (["c-3:0", "c-2:0"], 2)
    assert page.status_counts == {"conflict": 0, "pending": 2, "reviewed": 1}


@pytest.mark.parametrize("offset,limit", [(-1, 10), (0, 0), (0, 501)])
def test_queue_invalid_paging_raises(review_store, offset, limit):
    """Negative offsets and limits outside 1..500 are refused."""
    with pytest.raises(ValueError):
        build_review_queue(review_store, MODEL_VERSION, offset=offset, limit=limit)


def test_queue_page_serializes_to_json(review_store):
    """A page dumps to plain JSON (what the API will return) and validates back to an equal page."""
    verdict(review_store, "c-1:3", "Insurance", "accept", "ana")
    page = build_review_queue(review_store, MODEL_VERSION)
    payload = json.loads(page.model_dump_json())
    assert payload["items"][2]["current_verdicts"][0]["timestamp"] == "2026-11-01T09:00:00Z"
    assert ReviewQueuePage.model_validate(payload) == page
