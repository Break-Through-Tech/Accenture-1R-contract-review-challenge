"""
Review endpoints: the queue of predictions that need a human, recording verdicts, and how the
recorded verdicts resolve. Each handler calls one ClauseTriageService method, the same ones the
reviewer page calls.

Not one of create_app()'s built-in routers: python -m src.api passes it in through
create_app(service, routers=[review.router]), as any extra router would be.
"""
from typing import Annotated

from fastapi import APIRouter, Query

from src.api.routers import ServiceDep
from src.api.schemas import FeedbackRequest
from src.review.queue import MAX_PAGE_SIZE
from src.review.schema import Resolution, ResolutionStatus, ReviewQueuePage, ReviewStatus
from src.store.records import FeedbackRecord

router = APIRouter(tags=["review"])

ModelVersion = Annotated[str | None, Query(description="Model version (default: the server's)")]
ContractFilter = Annotated[str | None, Query(description="Only this contract (default: every contract)")]


@router.get("/review/queue", response_model=ReviewQueuePage)
def review_queue(service: ServiceDep, contract_id: ContractFilter = None,
                 status: Annotated[ReviewStatus | None, Query(description="pending, reviewed or conflict")] = None,
                 offset: Annotated[int, Query(ge=0)] = 0,
                 limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
                 model_version: ModelVersion = None) -> ReviewQueuePage:
    """
    One page of the review queue: Unknown predictions (no quote could be grounded) and reviewer
    conflicts, each with its chunk text, pages, reason, allowed verdicts and review status.
    Conflicts first, then pending, then reviewed; highest probability first within each.
    """
    return service.review_queue(contract_id, status, offset, limit, model_version)


@router.post("/feedback", status_code=201, response_model=FeedbackRecord)
def submit_feedback(request: FeedbackRequest, service: ServiceDep) -> FeedbackRecord:
    """
    Record a verdict on a (chunk, category): accept or reject a prediction, or flag a clause the
    model missed. A verdict that doesn't fit what the model predicted is refused with 400 and the
    reason; nothing is written.
    """
    return service.submit_feedback(request)


@router.get("/feedback", response_model=list[FeedbackRecord])
def list_feedback(service: ServiceDep, contract_id: ContractFilter = None,
                  chunk_id: Annotated[str | None, Query(description="Only this chunk")] = None,
                  category: Annotated[str | None, Query(description="Only this CUAD category")] = None,
                  model_version: ModelVersion = None) -> list[FeedbackRecord]:
    """Every recorded verdict under the model version, oldest first (the log is append-only)."""
    return service.list_feedback(contract_id, chunk_id, category, model_version)


@router.get("/feedback/resolutions", response_model=list[Resolution])
def list_resolutions(service: ServiceDep, contract_id: ContractFilter = None,
                     status: Annotated[ResolutionStatus | None, Query(description="resolved or conflict")] = None,
                     model_version: ModelVersion = None) -> list[Resolution]:
    """Each reviewed (chunk, category) with every reviewer's current verdict: resolved to a label, or a conflict."""
    return service.list_resolutions(contract_id, status, model_version)
