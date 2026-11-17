"""
Request and response models of the API. They are also what ClauseTriageService returns in-process,
so the UI gets the same shapes whether it calls the service directly or over HTTP.

Store records (ContractRecord, ChunkRecord, EvidenceRecord, UnknownItem, WeightVersionRecord,
ModelVersionRecord) and the risk stage's ClauseScore, ContractScore and WeightSet are reused as
they are, so there is one definition of each.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.ingest.documents import MAX_TEXT_CHARS
from src.risk.contract import ContractScore
from src.risk.schema import ClauseScore, WeightSet
from src.store.records import (ChunkRecord, ContractRecord, EvidenceRecord, ModelVersionRecord, UnknownItem, Verdict,
                               WeightVersionRecord)

# Longest display name accepted for pasted text or kept from an uploaded file name
MAX_NAME_CHARS = 120
# Weight version ids appear in URLs and file names, so new ones are limited to these characters
WEIGHT_VERSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
# Longest note a reviewer can attach to a verdict
MAX_NOTE_CHARS = 2000


# ----- Requests -----
class PastedTextRequest(BaseModel):
    """Contract text pasted by a reviewer, scored exactly as given; name is optional and display-only."""
    text: str = Field(max_length=MAX_TEXT_CHARS)
    name: str | None = Field(default=None, max_length=MAX_NAME_CHARS)


class PreviewRequest(BaseModel):
    """
    A candidate weight set to rank stored contracts with, saving nothing. contract_ids limits the
    preview to those contracts (default: every contract stored under the model version);
    model_version defaults to the server's.
    """
    weight_set: WeightSet
    contract_ids: list[str] | None = None
    model_version: str | None = None


class FeedbackRequest(BaseModel):
    """
    A reviewer's verdict on one (chunk, category), as a client sends it. chunk_id comes from a review
    queue item or a contract's chunks; model_version is the one the reviewer looked at (default: the
    server's). The reviewer name is self-declared (there are no accounts). The server sets the time.
    """
    chunk_id: str = Field(min_length=1)
    category: str
    verdict: Verdict
    reviewer: str = Field(max_length=MAX_NAME_CHARS)
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)
    model_version: str | None = None


# ----- Rankings -----
class RankedContract(BaseModel):
    """One contract's place in a ranking, with its rating counts and where it came from."""
    contract_id: str
    rank: int
    triage_score: float
    high_count: int
    medium_count: int
    low_count: int
    unknown_count: int
    source_name: str | None = None
    source_type: str | None = None
    page_count: int | None = None


class ContractRanking(BaseModel):
    """
    Contracts ranked for review under one model version and weight version (rank 1 = open first).
    saved is False for a preview, whose scores were not stored.
    """
    model_version: str
    weight_version_id: str
    weight_set_sha256: str
    ranked_out_of: int
    saved: bool = True
    contracts: list[RankedContract]


# ----- One contract -----
class ChunkDetail(ChunkRecord):
    """A stored chunk with the raw probability of every CUAD category under the model version."""
    probabilities: dict[str, float]


class Prediction(BaseModel):
    """One detection: a (chunk, category) whose probability cleared its threshold, grounded or not."""
    chunk_index: int
    category: str
    probability: float
    threshold: float
    has_evidence: bool


class ContractDetailResponse(BaseModel):
    """
    Everything stored for one contract under one model version and weight version: the text and
    source metadata, chunks with probabilities, detections, grounded evidence, the Unknown queue,
    clause scores, and the contract score with its rank among every contract under the same pair.
    """
    contract: ContractRecord
    model_version: str
    model_name: str
    weight_version_id: str
    weight_set_sha256: str
    thresholds: dict[str, float]
    chunks: list[ChunkDetail]
    predictions: list[Prediction]
    evidence: list[EvidenceRecord]
    unknowns: list[UnknownItem]
    clause_scores: list[ClauseScore]
    contract_score: ContractScore
    rank: int
    ranked_out_of: int


# ----- Weights and models -----
class WeightVersionList(BaseModel):
    """Every stored weight version, oldest first, and the one new uploads are scored with."""
    default_weight_version_id: str
    versions: list[WeightVersionRecord]


class WeightVersionDetail(BaseModel):
    """One stored weight version: its metadata (created_by, created_at, hash) and the full weight set."""
    version: WeightVersionRecord
    is_default: bool
    weight_set: WeightSet


class WeightVersionCreated(BaseModel):
    """
    Result of POST /weights. created is False when an identical version already existed (nothing
    changed). rescored_contracts counts the contracts scored under the version for model_version.
    """
    created: bool
    version: WeightVersionRecord
    model_version: str
    rescored_contracts: int


class ModelVersionList(BaseModel):
    """Every model version with stored results, and the one this server scores uploads with."""
    server_model_version: str
    versions: list[ModelVersionRecord]


class HealthResponse(BaseModel):
    """Liveness plus the model and default weight version this server uses."""
    status: str = "ok"
    model_version: str
    default_weight_version_id: str
