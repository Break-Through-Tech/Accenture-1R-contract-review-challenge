"""
Pydantic models the artifact store returns from its read methods.

These cross the boundary between the store and its readers (the FastAPI layer, the reviewer UI,
the clause-register export), so readers never see SQLAlchemy rows. Clause and contract scores
are returned as the risk stage's own ClauseScore and ContractScore models.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from src.risk.contract import ContractScore
from src.risk.schema import ClauseScore

Verdict = Literal["accept", "reject", "missing"]
Bucket = Literal["Low", "Medium", "High"]
UnknownReason = Literal["no_cue_match", "span_mismatch"]


class ContractSummary(BaseModel):
    """One stored contract without its text, for listings."""
    contract_id: str
    text_sha256: str
    text_length: int
    source_name: str | None = None
    source_type: str | None = None
    page_count: int | None = None
    ocr_used: bool | None = None
    created_at: datetime


class ContractRecord(ContractSummary):
    """One stored contract with its exact text and page-start offsets (None when not an upload)."""
    text: str
    page_starts: tuple[int, ...] | None = None


class ChunkRecord(BaseModel):
    """One stored chunk; text == contract text[start:end]."""
    chunk_id: str
    contract_id: str
    chunk_index: int
    start: int
    end: int
    text: str


class ModelVersionRecord(BaseModel):
    """What the store knows about a model version: enough to rescore without loading the model."""
    model_version: str
    model_name: str
    seed: int | None = None
    thresholds: dict[str, float]
    created_at: datetime


class WeightVersionRecord(BaseModel):
    """A stored weight version's metadata; load the WeightSet itself with get_weight_version."""
    weight_version_id: str
    created_by: str
    description: str
    content_sha256: str
    created_at: datetime


class EvidenceRecord(BaseModel):
    """A grounded quote for a detected clause; start/end are absolute contract offsets."""
    evidence_id: str
    contract_id: str
    chunk_id: str
    chunk_index: int
    category: str
    model_version: str
    quote: str
    start: int
    end: int


class UnknownItem(BaseModel):
    """
    One Unknown-queue item: a prediction that cleared its threshold but has no grounded evidence.
    chunk_start/chunk_end locate the chunk to show the reviewer; threshold is None if the model
    version was stored without thresholds.
    """
    contract_id: str
    chunk_id: str
    chunk_index: int
    chunk_start: int
    chunk_end: int
    category: str
    model_version: str
    probability: float
    threshold: float | None = None
    reason: UnknownReason


class FeedbackRecord(BaseModel):
    """A reviewer's verdict on one (chunk, category) prediction."""
    id: int
    contract_id: str
    chunk_id: str
    category: str
    model_version: str
    verdict: Verdict
    reviewer: str
    timestamp: datetime
    note: str | None = None


class AdvisorRankingRecord(BaseModel):
    """One advisor-ranked clause span; advisor_rank 1 is the riskiest."""
    id: int
    contract_id: str
    category: str
    start: int
    end: int
    quote: str
    advisor_rank: int | None = None
    advisor_bucket: Bucket | None = None
    reviewer: str
    ranking_set: str
    timestamp: datetime


class ContractDetail(BaseModel):
    """
    Everything a reviewer sees for one contract under one model version and weight version.
    contract_score is None if the contract was never scored under that pair; its rank is
    among every contract stored under the same pair.
    """
    contract: ContractRecord
    model_version: str
    weight_version_id: str
    chunks: list[ChunkRecord]
    evidence: list[EvidenceRecord]
    unknowns: list[UnknownItem]
    clause_scores: list[ClauseScore]
    contract_score: ContractScore | None = None
