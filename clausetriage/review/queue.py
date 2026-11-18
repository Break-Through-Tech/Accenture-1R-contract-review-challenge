"""
The review queue: every (chunk, category) under a model version that needs a human, with what the
reviewer needs to decide it and its review status, in a stable order and in pages.

Sources (each a function store, model_version, contract_id -> list[QueueCandidate]):
    unknown_candidates   every Unknown: a prediction above threshold whose evidence could not be grounded
    conflict_candidates  every (chunk, category) whose reviewers disagree under this model version,
                         grounded predictions and missed clauses included (Unknowns are already listed)
A new source (for example uncertainty sampling, chunks just below threshold) is one more function
passed in `sources`; when two sources name the same (chunk, category), the first source wins.

Status comes from feedback given under the same model version (see clausetriage/review/feedback.py):
pending (no verdict yet), reviewed (every reviewer's current verdict implies the same label) or
conflict (they imply different labels).

Order (stable for a given store content):
    1. status: conflict, then pending, then reviewed. Conflicts block export and already hold
       reviewer work; pending items are the backlog; reviewed items need nothing.
    2. probability, highest first: the model is surest a clause is there but could not quote it,
       so these are the likeliest real clauses missing from the register.
    3. contract_id, chunk_index, category column: a fixed tiebreak.
Writing feedback moves an item (pending -> reviewed), so offsets shift between calls; a UI working
through the backlog should ask for status="pending" and re-read page 0 after each verdict.

How everything comes together:
build_review_queue(): gather candidates from every source, add chunk text, pages, evidence and
                      status, sort, filter by status and return one ReviewQueuePage
"""
from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence

from clausetriage.labels.categories import CATEGORY_TO_INDEX
from clausetriage.review.feedback import list_resolutions
from clausetriage.review.schema import QueueCandidate, Resolution, ReviewItem, ReviewQueuePage, ReviewStatus
from clausetriage.store.artifact_store import ArtifactStore, evidence_id_for
from clausetriage.store.records import ChunkRecord, ContractRecord, EvidenceRecord

QueueSource = Callable[[ArtifactStore, str, str | None], list[QueueCandidate]]

MAX_PAGE_SIZE = 500
STATUS_ORDER: dict[str, int] = {"conflict": 0, "pending": 1, "reviewed": 2}

# Plain-language reasons shown to the reviewer
REASON_TEXT = {
    "no_cue_match": "The model predicted this category, but no sentence in the chunk contains the "
                    "category's cue words, so no quote could be grounded.",
    "span_mismatch": "The model predicted this category and a quote was picked, but it did not match "
                     "the contract text at its offsets, so it was not shown.",
    "reviewer_conflict": "Reviewers gave verdicts that disagree on whether this category is in the chunk.",
}


def unknown_candidates(store: ArtifactStore, model_version: str, contract_id: str | None) -> list[QueueCandidate]:
    """Every Unknown prediction under model_version (optionally one contract)."""
    return [QueueCandidate(source="unknown", contract_id=item.contract_id, chunk_id=item.chunk_id,
                           chunk_index=item.chunk_index, category=item.category, probability=item.probability,
                           threshold=item.threshold, detection="unknown", reason=item.reason,
                           reason_text=REASON_TEXT[item.reason])
            for item in store.list_unknowns(model_version, contract_id)]


def conflict_candidates(store: ArtifactStore, model_version: str, contract_id: str | None) -> list[QueueCandidate]:
    """Every (chunk, category) whose verdicts under model_version conflict, whatever the model predicted."""
    thresholds = store.get_model_version(model_version).thresholds
    candidates = []
    for resolution in list_resolutions(store, model_version=model_version, contract_id=contract_id,
                                       status="conflict"):
        chunk = store.get_chunk(resolution.chunk_id)
        probability = store.get_probabilities(chunk.chunk_id, model_version)[resolution.category]
        candidates.append(QueueCandidate(
            source="conflict", contract_id=chunk.contract_id, chunk_id=chunk.chunk_id, chunk_index=chunk.chunk_index,
            category=resolution.category, probability=probability, threshold=thresholds.get(resolution.category),
            detection=store.get_detection(chunk.chunk_id, resolution.category, model_version),
            reason="reviewer_conflict", reason_text=REASON_TEXT["reviewer_conflict"]))
    return candidates


DEFAULT_SOURCES: tuple[QueueSource, ...] = (unknown_candidates, conflict_candidates)


def _page_of(offset: int, page_starts: Sequence[int] | None) -> int | None:
    """1-based page holding text[offset], or None without page offsets (or before the first page)."""
    if not page_starts:
        return None
    return bisect.bisect_right(page_starts, offset) or None


def _status(resolution: Resolution | None) -> ReviewStatus:
    """Queue status from a key's resolution under the queue's model version."""
    if resolution is None:
        return "pending"
    return "reviewed" if resolution.status == "resolved" else "conflict"


def _item(candidate: QueueCandidate, model_version: str, contract: ContractRecord, chunk: ChunkRecord,
          evidence: EvidenceRecord | None, resolution: Resolution | None) -> ReviewItem:
    """Add chunk text, pages, evidence, allowed verdicts and status to a candidate."""
    return ReviewItem(
        **candidate.model_dump(), item_id=evidence_id_for(chunk.chunk_id, candidate.category, model_version),
        model_version=model_version, source_name=contract.source_name, chunk_start=chunk.start,
        chunk_end=chunk.end, chunk_text=chunk.text, first_page=_page_of(chunk.start, contract.page_starts),
        last_page=_page_of(chunk.end - 1, contract.page_starts),
        evidence_quote=evidence.quote if evidence else None, evidence_start=evidence.start if evidence else None,
        evidence_end=evidence.end if evidence else None,
        allowed_verdicts=["accept", "reject"] if candidate.detection else ["missing", "reject"],
        status=_status(resolution), resolved_label=resolution.label if resolution else None,
        current_verdicts=resolution.current if resolution else [])


def _sort_key(item: ReviewItem) -> tuple:
    """Queue order: status, then probability (highest first), then contract, chunk and category column."""
    return (STATUS_ORDER[item.status], -item.probability, item.contract_id, item.chunk_index,
            CATEGORY_TO_INDEX[item.category])


def build_review_queue(store: ArtifactStore, model_version: str, *, contract_id: str | None = None,
                       status: ReviewStatus | None = None, offset: int = 0, limit: int = 50,
                       sources: Sequence[QueueSource] = DEFAULT_SOURCES) -> ReviewQueuePage:
    """
    One page of the review queue for a model version (optionally one contract and one status).
    KeyError if the model version or contract was never stored; ValueError for a bad offset or
    limit (1..MAX_PAGE_SIZE). See the module docstring for sources, status and order.
    """
    if offset < 0 or not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"offset must be >= 0 and limit in 1..{MAX_PAGE_SIZE}")
    store.get_model_version(model_version)
    if contract_id is not None:
        store.get_contract(contract_id)

    # Candidates from every source; the first source to name a (chunk, category) keeps it
    candidates: dict[tuple[str, str], QueueCandidate] = {}
    for source in sources:
        for candidate in source(store, model_version, contract_id):
            candidates.setdefault((candidate.chunk_id, candidate.category), candidate)

    # What the reviewer sees: contract metadata, chunk text, evidence for grounded predictions, status
    resolution_of = {(r.chunk_id, r.category): r
                     for r in list_resolutions(store, model_version=model_version, contract_id=contract_id)}
    contracts, chunks, evidence = {}, {}, {}
    for cid in sorted({c.contract_id for c in candidates.values()}):
        contracts[cid] = store.get_contract(cid)
        chunks.update({chunk.chunk_id: chunk for chunk in store.get_chunks(cid)})
        evidence.update({(e.chunk_id, e.category): e for e in store.get_evidence(cid, model_version)})
    items = sorted((_item(c, model_version, contracts[c.contract_id], chunks[c.chunk_id], evidence.get(key),
                          resolution_of.get(key)) for key, c in candidates.items()), key=_sort_key)

    # Counts before the status filter, then one page
    counts = {name: sum(item.status == name for item in items) for name in STATUS_ORDER}
    selected = [item for item in items if status is None or item.status == status]
    return ReviewQueuePage(model_version=model_version, contract_id=contract_id, status=status, offset=offset,
                           limit=limit, total=len(selected), status_counts=counts,
                           items=selected[offset:offset + limit])
