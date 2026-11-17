"""
The plain-Python layer behind every endpoint. Handlers stay thin: they parse the request, call one
method here and return its Pydantic model. The reviewer UI can call the same methods in-process.

How everything comes together:
submit_upload(), submit_text(), submit_document(): ingest a contract and queue its scoring job
run_job(): runs the pipeline and saves the result to the artifact store (FastAPI runs it in the
           background); the job becomes complete only after the store has the result
get_job(), list_jobs(): scoring status
list_ranked(), get_contract_detail(), get_register(): reads, each under a model and weight version
list_weights(), get_weights(), create_weights(), preview(): weight versions and re-ranking
review_queue(), submit_feedback(), list_feedback(), list_resolutions(): the human-in-the-loop
           operations of src/review/, with its errors turned into the service's

Scores under a weight version are filled in from cached probabilities, never by rerunning the
model: create_weights() rescores every stored contract when it stores a new version, and every read
first rescores any contract that was stored after that (ensure_scored). Rescoring reuses
store.rescoring.rescore_contracts, so stored scores equal what the pipeline would give.

Writes to the store go through one lock, so a background job, a weight post and a read that
fills in scores never race each other. Run one server process per database file.
"""
from __future__ import annotations

import logging
import re
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError

from src.api.jobs import JobRecord, JobStore
from src.api.schemas import (MAX_NAME_CHARS, WEIGHT_VERSION_ID_PATTERN, ChunkDetail, ContractDetailResponse,
                             ContractRanking, FeedbackRequest, ModelVersionList, Prediction, RankedContract,
                             WeightVersionCreated, WeightVersionDetail, WeightVersionList)
from src.evidence.extractor import GroundedEvidence, Unknown
from src.evidence.schema import Evidence
from src.inference import ContractInferencePipeline
from src.ingest.documents import (MAX_FILE_BYTES, MB, SUPPORTED_SUFFIXES, Document, document_from_bytes,
                                  pasted_document)
from src.register.export import build_register
from src.register.schema import ContractRegister
from src.review.feedback import list_resolutions as resolve_stored_feedback
from src.review.feedback import submit_feedback as record_feedback
from src.review.queue import MAX_PAGE_SIZE, STATUS_ORDER, build_review_queue
from src.review.schema import FeedbackSubmission, Resolution, ResolutionStatus, ReviewQueuePage, ReviewStatus
from src.risk.contract import ContractScore
from src.risk.schema import WeightSet
from src.store.artifact_store import ArtifactStore
from src.store.records import FeedbackRecord
from src.store.rescoring import rescore_contracts

logger = logging.getLogger(__name__)

# Shown for a failed job whose error isn't a reviewer-readable ValueError
INTERNAL_ERROR = "Scoring failed because of an internal error. The server log has the details."
_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9]+")
_PATH_SEPARATORS = re.compile(r"[\\/]")
MAX_SLUG_CHARS = 60


# ----- Errors the endpoints turn into HTTP status codes -----
class ServiceError(Exception):
    """Base class for errors whose message is safe to show to the client."""


class NotFoundError(ServiceError):
    """An unknown contract, job, model version or weight version (HTTP 404)."""


class ConflictError(ServiceError):
    """A request that would change something immutable, such as an existing weight version (HTTP 409)."""


class InvalidInputError(ServiceError):
    """Input that can't be ingested or used, with a message saying why (HTTP 400)."""


class UnsupportedTypeError(InvalidInputError):
    """An upload whose extension isn't a supported document type (HTTP 415)."""


class TooLargeError(InvalidInputError):
    """An upload over the size limit (HTTP 413)."""


# ----- Names and ids from untrusted input -----
def safe_filename(filename: str | None) -> str:
    """
    A display-safe version of an uploaded file name: directory parts (either slash) dropped, Unicode
    normalised (NFC), control and other invisible characters removed, whitespace collapsed, and the
    stem cut to MAX_NAME_CHARS. The extension is kept so the file type can still be checked.
    """
    name = _PATH_SEPARATORS.split(filename or "")[-1]
    name = "".join(ch for ch in unicodedata.normalize("NFC", name) if unicodedata.category(ch)[0] != "C")
    name = " ".join(name.split())
    stem, suffix = (name[:-len(Path(name).suffix)], Path(name).suffix) if Path(name).suffix else (name, "")
    return stem[:MAX_NAME_CHARS].strip() + suffix


def validation_message(error: ValidationError) -> str:
    """A pydantic ValidationError as one readable line: where and why for each problem, never the input."""
    problems = []
    for item in error.errors():
        where = ".".join(str(part) for part in item.get("loc", ()))
        message = str(item.get("msg", "")).removeprefix("Value error, ")
        problems.append(f"{where}: {message}" if where else message)
    return "; ".join(problems)


def upload_contract_id(name: str, text_sha256: str) -> str:
    """
    The contract id of an upload: an ASCII slug of its name (letters and digits, other runs turned
    into '-', at most MAX_SLUG_CHARS) plus the first 12 hex characters of its text's SHA-256. The
    same file uploaded again gets the same id; a different text under the same name gets another.
    Safe in URLs and file names.
    """
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = _UNSAFE_ID_CHARS.sub("-", ascii_name).strip("-")[:MAX_SLUG_CHARS].rstrip("-") or "contract"
    return f"{slug}-{text_sha256[:12]}"


class ClauseTriageService:
    """
    Scores contracts into an ArtifactStore and serves what is stored. One pipeline and one model
    version score new uploads, under the pipeline's weight set (the default weight version); reads
    can name any stored model version and weight version.
    """

    def __init__(self, store: ArtifactStore, pipeline: ContractInferencePipeline, model_version: str,
                 jobs: JobStore | None = None) -> None:
        """
        Wrap a store and pipeline. model_version is the registry-qualified key the pipeline's model
        is stored under. Stores the default weight version and settles jobs a stopped server left
        unfinished (see JobStore.recover).
        """
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version must be a non-empty string")
        self.store, self.pipeline, self.model_version = store, pipeline, model_version
        self.jobs = jobs or JobStore(store.engine)
        self._lock = threading.RLock()
        self._pending: dict[tuple[str, str], tuple[Document, str | None, str | None]] = {}
        with self._lock:
            store.put_weight_version(self.default_weight_set)
            self.jobs.recover(self._is_stored)

    @property
    def default_weight_set(self) -> WeightSet:
        """The weight set new uploads are scored with (the pipeline's)."""
        return self.pipeline.weight_set

    # ----- Submitting contracts -----
    def submit_upload(self, data: bytes, filename: str | None) -> tuple[JobRecord, bool]:
        """
        Ingest an uploaded file and queue its scoring. Returns (job, queued): queued is False when the
        same file is already scored or being scored, and nothing new needs to run.
        Raises UnsupportedTypeError, TooLargeError or InvalidInputError with a reviewer-readable reason.
        """
        name = safe_filename(filename)
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            kind = f"type {suffix!r}" if suffix else "without an extension"
            raise UnsupportedTypeError(f"Can't read a file {kind}; upload a {', '.join(SUPPORTED_SUFFIXES)} file.")
        if len(data) > MAX_FILE_BYTES:
            raise TooLargeError(f"This file is over the {MAX_FILE_BYTES // MB} MB limit.")
        try:
            document = document_from_bytes(data, name)
        except ValueError as error:
            raise InvalidInputError(str(error)) from error
        return self.submit_document(document, source_name=name)

    def submit_text(self, text: str, name: str | None = None) -> tuple[JobRecord, bool]:
        """Queue pasted text for scoring, unchanged; see submit_upload for the return value."""
        display = safe_filename(name) if name and name.strip() else "Pasted text"
        try:
            document = pasted_document(text, name=display or "Pasted text")
        except (TypeError, ValueError) as error:
            raise InvalidInputError(str(error)) from error
        return self.submit_document(document, source_name=display)

    def submit_document(self, document: Document, *, contract_id: str | None = None, source_name: str | None = None,
                        source_type: str | None = None) -> tuple[JobRecord, bool]:
        """
        Queue a Document for scoring under the server's model version and default weights.
        contract_id defaults to upload_contract_id(document.name, document.sha256); source_type
        defaults to document.source_type. Returns (job, queued) as submit_upload does.
        """
        try:
            document.text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise InvalidInputError("The text contains characters that aren't valid Unicode.") from error
        contract_id = contract_id or upload_contract_id(document.name, document.sha256)
        source_type = source_type or document.source_type
        key = (contract_id, self.model_version)
        with self._lock:
            # The same contract already scored, or already on its way: nothing new to run
            job = self.jobs.get(*key)
            if job is not None and job.status in ("queued", "running"):
                return job, False
            if self._is_stored(*key):
                self._check_same_text(contract_id, document.text)
                if job is None or job.status != "complete":
                    self.jobs.queue(*key, self.default_weight_set.weight_version_id,
                                    source_name=source_name, source_type=source_type)
                    job = self.jobs.finish(*key, "complete")
                return job, False

            # New (or previously failed): keep the document in memory until the job runs
            self._pending[key] = (document, source_name, source_type)
            job = self.jobs.queue(*key, self.default_weight_set.weight_version_id,
                                  source_name=source_name, source_type=source_type)
            return job, True

    def _check_same_text(self, contract_id: str, text: str) -> None:
        """ConflictError if the store holds different text under contract_id."""
        if self.store.get_contract(contract_id).text != text:
            raise ConflictError(f"Contract {contract_id!r} is already stored with different text.")

    def run_job(self, contract_id: str) -> JobRecord | None:
        """
        Score a queued contract and save the result; never raises. The job becomes complete only
        once the store has committed the result, and failed (with a reason) on any error. A job that
        isn't queued is returned unchanged (None if there is no job).
        """
        key = (contract_id, self.model_version)
        if not self.jobs.claim(*key):
            return self.jobs.get(*key)
        completed = False
        try:
            pending = self._pending.pop(key, None)
            if pending is None:
                raise ValueError("The uploaded text is no longer in memory (was the server restarted?). "
                                 "Upload the contract again.")
            document, source_name, source_type = pending
            result = self.pipeline.predict_contract(contract_id, document.text)
            with self._lock:
                self.store.save_inference_result(
                    result, text=document.text, model_version=self.model_version,
                    thresholds=self.pipeline.bundle.thresholds, weight_set=self.default_weight_set,
                    source_name=source_name, source_type=source_type,
                    page_starts=document.page_starts or None,
                    ocr_used=document.ocr if document.source_type == "pdf" else None,
                )
                if not self._is_stored(*key):
                    raise RuntimeError(f"the store has no result for {contract_id!r} after saving it")
                job = self.jobs.finish(*key, "complete")
            completed = True
            return job
        except Exception as error:  # any failure is recorded on the job, never reported as complete
            logger.exception("Scoring %r failed", contract_id)
            message = str(error) if isinstance(error, (ValueError, ServiceError)) else INTERNAL_ERROR
            return self.jobs.finish(*key, "failed", message)
        finally:
            # KeyboardInterrupt, SystemExit and the like: still never leave the job running
            if not completed and (job := self.jobs.get(*key)) is not None and job.status == "running":
                self.jobs.finish(*key, "failed", "Scoring was interrupted before it finished.")

    def _is_stored(self, contract_id: str, model_version: str) -> bool:
        """Whether the store holds the pipeline result for contract_id under model_version."""
        return contract_id in self.store.list_run_contract_ids(model_version)

    # ----- Job status -----
    def get_job(self, contract_id: str) -> JobRecord:
        """
        The scoring status of a contract under the server's model version. A contract stored without
        a job (for example by another tool) reports complete. NotFoundError if neither exists.
        """
        key = (contract_id, self.model_version)
        job = self.jobs.get(*key)
        if job is not None:
            return job
        if self._is_stored(*key):
            with self._lock:
                record = self.store.get_contract(contract_id)
                self.jobs.queue(*key, self.default_weight_set.weight_version_id,
                                source_name=record.source_name, source_type=record.source_type)
                return self.jobs.finish(*key, "complete")
        raise NotFoundError(f"No contract or scoring job {contract_id!r} under model version {self.model_version!r}.")

    def list_jobs(self) -> list[JobRecord]:
        """Every scoring job under the server's model version, most recent first."""
        return self.jobs.list(self.model_version)

    # ----- Reads -----
    def _resolve(self, model_version: str | None, weight_version_id: str | None) -> tuple[str, WeightSet]:
        """Default and check a (model version, weight version) pair; NotFoundError if either is unknown."""
        model_version = model_version or self.model_version
        weight_version_id = weight_version_id or self.default_weight_set.weight_version_id
        if model_version != self.model_version and not self._model_version_stored(model_version):
            raise NotFoundError(f"Unknown model version {model_version!r}.")
        return model_version, self._weight_set(weight_version_id)

    def _model_version_stored(self, model_version: str) -> bool:
        """Whether the store knows model_version."""
        return any(record.model_version == model_version for record in self.store.list_model_versions())

    def _weight_set(self, weight_version_id: str) -> WeightSet:
        """A stored WeightSet; NotFoundError if it doesn't exist."""
        try:
            return self.store.get_weight_version(weight_version_id)
        except KeyError as error:
            raise NotFoundError(f"Unknown weight version {weight_version_id!r}.") from error

    def ensure_scored(self, model_version: str, weight_set: WeightSet) -> int:
        """
        Score, from cached probabilities, every contract stored under model_version that has no score
        under weight_set yet, and save the scores. Returns how many contracts were scored.
        """
        with self._lock:
            run = set(self.store.list_run_contract_ids(model_version))
            scored = {s.contract_id for s in self.store.list_contract_scores(model_version,
                                                                              weight_set.weight_version_id)}
            missing = sorted(run - scored)
            if missing:
                rescore_contracts(self.store, model_version, weight_set, missing)
            return len(missing)

    def _ranked(self, model_version: str, weight_set: WeightSet) -> list[ContractScore]:
        """Every contract ranked under the pair, after filling in missing scores."""
        self.ensure_scored(model_version, weight_set)
        return self.store.list_contract_scores(model_version, weight_set.weight_version_id)

    def _ranking(self, model_version: str, weight_set: WeightSet, scores: Sequence[ContractScore],
                 saved: bool) -> ContractRanking:
        """Wrap ranked contract scores with each contract's source metadata."""
        summaries = {summary.contract_id: summary for summary in self.store.list_contracts()}
        contracts = []
        for score in scores:
            summary = summaries.get(score.contract_id)
            contracts.append(RankedContract(
                contract_id=score.contract_id, rank=score.rank, triage_score=score.triage_score,
                high_count=score.high_count, medium_count=score.medium_count, low_count=score.low_count,
                unknown_count=score.unknown_count, source_name=summary.source_name if summary else None,
                source_type=summary.source_type if summary else None,
                page_count=summary.page_count if summary else None))
        return ContractRanking(model_version=model_version, weight_version_id=weight_set.weight_version_id,
                               weight_set_sha256=weight_set.content_sha256, ranked_out_of=len(scores), saved=saved,
                               contracts=contracts)

    def list_ranked(self, model_version: str | None = None, weight_version_id: str | None = None) -> ContractRanking:
        """Every stored contract ranked for review (rank 1 = open first) under the model and weight version."""
        model_version, weight_set = self._resolve(model_version, weight_version_id)
        return self._ranking(model_version, weight_set, self._ranked(model_version, weight_set), saved=True)

    def _scored_contract(self, contract_id: str, model_version: str | None,
                         weight_version_id: str | None) -> tuple[str, WeightSet, list[ContractScore], ContractScore]:
        """Resolve the pair, fill in scores, and return the ranking and this contract's score; NotFoundError if absent."""
        model_version, weight_set = self._resolve(model_version, weight_version_id)
        if contract_id not in self.store.list_run_contract_ids(model_version):
            raise NotFoundError(f"No scored contract {contract_id!r} under model version {model_version!r}.")
        ranked = self._ranked(model_version, weight_set)
        score = next(score for score in ranked if score.contract_id == contract_id)
        return model_version, weight_set, ranked, score

    def get_contract_detail(self, contract_id: str, model_version: str | None = None,
                            weight_version_id: str | None = None) -> ContractDetailResponse:
        """One contract's text, chunks with probabilities, detections, evidence, Unknowns, scores and rank."""
        model_version, weight_set, ranked, score = self._scored_contract(contract_id, model_version, weight_version_id)
        detail = self.store.get_contract_detail(contract_id, model_version, weight_set.weight_version_id)
        model = self.store.get_model_version(model_version)
        chunks = [ChunkDetail(**chunk.model_dump(), probabilities=self.store.get_probabilities(chunk.chunk_id,
                                                                                               model_version))
                  for chunk in detail.chunks]
        probability_of = {(chunk.chunk_index, category): value
                          for chunk in chunks for category, value in chunk.probabilities.items()}
        grounded = {(e.chunk_index, e.category) for e in detail.evidence}
        predictions = [Prediction(chunk_index=clause.chunk_index, category=clause.category,
                                  probability=probability_of[clause.chunk_index, clause.category],
                                  threshold=model.thresholds[clause.category],
                                  has_evidence=(clause.chunk_index, clause.category) in grounded)
                       for clause in detail.clause_scores]
        return ContractDetailResponse(
            contract=detail.contract, model_version=model_version, model_name=model.model_name,
            weight_version_id=weight_set.weight_version_id, weight_set_sha256=weight_set.content_sha256,
            thresholds=model.thresholds, chunks=chunks, predictions=predictions, evidence=detail.evidence,
            unknowns=detail.unknowns, clause_scores=detail.clause_scores, contract_score=score, rank=score.rank,
            ranked_out_of=len(ranked))

    def get_register(self, contract_id: str, model_version: str | None = None,
                     weight_version_id: str | None = None) -> ContractRegister:
        """
        The contract's clause register built by register.export.build_register from stored data,
        ranked among every contract under the pair, with every quote re-checked against the stored text.
        """
        model_version, weight_set, ranked, score = self._scored_contract(contract_id, model_version, weight_version_id)
        contract = self.store.get_contract(contract_id)
        model = self.store.get_model_version(model_version)
        evidence = self.store.get_evidence(contract_id, model_version)
        unknowns = self.store.list_unknowns(model_version, contract_id)
        result = {
            "contract_id": contract_id, "model_name": model.model_name, "seed": model.seed,
            "grounded": [GroundedEvidence(contract_id=contract_id, chunk_index=e.chunk_index, category=e.category,
                                          evidence=Evidence(quote=e.quote, start=e.start, end=e.end))
                         for e in evidence],
            "unknown": [Unknown(contract_id=contract_id, chunk_index=u.chunk_index, category=u.category,
                                reason=u.reason) for u in unknowns],
            "clause_scores": self.store.get_clause_scores(contract_id, model_version, weight_set.weight_version_id),
        }
        return build_register(result, score, text=contract.text, page_starts=contract.page_starts or (),
                              ranked_out_of=len(ranked))

    def list_model_versions(self) -> ModelVersionList:
        """Every stored model version and the one this server scores with."""
        return ModelVersionList(server_model_version=self.model_version, versions=self.store.list_model_versions())

    # ----- Weights -----
    def list_weights(self) -> WeightVersionList:
        """Every stored weight version (oldest first) and the default one."""
        return WeightVersionList(default_weight_version_id=self.default_weight_set.weight_version_id,
                                 versions=self.store.list_weight_versions())

    def get_weights(self, weight_version_id: str) -> WeightVersionDetail:
        """One stored weight version with its metadata; NotFoundError if unknown."""
        weight_set = self._weight_set(weight_version_id)
        record = next(r for r in self.store.list_weight_versions() if r.weight_version_id == weight_version_id)
        return WeightVersionDetail(version=record, weight_set=weight_set,
                                   is_default=weight_version_id == self.default_weight_set.weight_version_id)

    def create_weights(self, weight_set: WeightSet) -> WeightVersionCreated:
        """
        Store a new weight version and score every contract stored under the server's model version
        with it, from cached probabilities, in one transaction. Posting the identical content again
        changes nothing (created False). ConflictError if the id exists with other content, and
        InvalidInputError if the id has characters outside WEIGHT_VERSION_ID_PATTERN.
        """
        if not re.fullmatch(WEIGHT_VERSION_ID_PATTERN, weight_set.weight_version_id):
            raise InvalidInputError("weight_version_id must start with a letter or digit and use only letters, "
                                    "digits, '.', '_' and '-' (at most 100 characters).")
        with self._lock:
            try:
                existing = self.store.get_weight_version(weight_set.weight_version_id)
            except KeyError:
                existing = None
            if existing is not None and existing.content_sha256 != weight_set.content_sha256:
                raise ConflictError(f"Weight version {weight_set.weight_version_id!r} already exists with different "
                                    "values. Versions never change; post the edit under a new weight_version_id.")
            if existing is None:
                # Stores the version and its scores together: rescore_contracts saves in one transaction
                scored = rescore_contracts(self.store, self.model_version, weight_set)
                rescored = len(scored)
            else:
                rescored = self.ensure_scored(self.model_version, weight_set)
            record = next(r for r in self.store.list_weight_versions()
                          if r.weight_version_id == weight_set.weight_version_id)
        return WeightVersionCreated(created=existing is None, version=record, model_version=self.model_version,
                                    rescored_contracts=rescored)

    def preview(self, weight_set: WeightSet, contract_ids: Sequence[str] | None = None,
                model_version: str | None = None) -> ContractRanking:
        """
        Rank stored contracts under a candidate weight set without saving anything (saved False).
        NotFoundError if the model version or one of contract_ids has no stored results.
        """
        model_version = model_version or self.model_version
        if model_version != self.model_version and not self._model_version_stored(model_version):
            raise NotFoundError(f"Unknown model version {model_version!r}.")
        if contract_ids is not None:
            run = set(self.store.list_run_contract_ids(model_version))
            unknown = sorted(set(contract_ids) - run)
            if unknown:
                raise NotFoundError(f"No scored contracts {unknown!r} under model version {model_version!r}.")
        scores = rescore_contracts(self.store, model_version, weight_set, contract_ids, save=False)
        return self._ranking(model_version, weight_set, scores, saved=False)

    # ----- Review queue and feedback -----
    def _review_model_version(self, model_version: str | None) -> str:
        """Default (to the server's) and check the model version a review read or verdict refers to."""
        model_version = model_version or self.model_version
        if model_version != self.model_version and not self._model_version_stored(model_version):
            raise NotFoundError(f"Unknown model version {model_version!r}.")
        return model_version

    def _check_contract(self, contract_id: str) -> None:
        """NotFoundError if the store has no contract contract_id."""
        try:
            self.store.get_contract(contract_id)
        except KeyError as error:
            raise NotFoundError(f"No stored contract {contract_id!r}.") from error

    def review_queue(self, contract_id: str | None = None, status: ReviewStatus | None = None, offset: int = 0,
                     limit: int = 50, model_version: str | None = None) -> ReviewQueuePage:
        """
        One page of the review queue (src.review.queue.build_review_queue): Unknown predictions and
        reviewer conflicts under the model version (default: the server's), optionally for one
        contract and one status. An empty page while nothing is scored under the model version.
        NotFoundError for an unknown contract or model version; InvalidInputError for a status
        other than pending/reviewed/conflict, a negative offset or a limit outside 1..MAX_PAGE_SIZE.
        """
        model_version = self._review_model_version(model_version)
        if status is not None and status not in get_args(ReviewStatus):
            raise InvalidInputError(f"status must be one of {', '.join(get_args(ReviewStatus))}.")
        if offset < 0 or not 1 <= limit <= MAX_PAGE_SIZE:
            raise InvalidInputError(f"offset must be 0 or more and limit between 1 and {MAX_PAGE_SIZE}.")
        if contract_id is not None:
            self._check_contract(contract_id)
        if not self._model_version_stored(model_version):
            return ReviewQueuePage(model_version=model_version, contract_id=contract_id, status=status, offset=offset,
                                   limit=limit, total=0, status_counts=dict.fromkeys(STATUS_ORDER, 0), items=[])
        return build_review_queue(self.store, model_version, contract_id=contract_id, status=status, offset=offset,
                                  limit=limit)

    def submit_feedback(self, request: FeedbackRequest | Mapping[str, Any]) -> FeedbackRecord:
        """
        Record one verdict (accept, reject or missing) on a (chunk, category), timestamped now, through
        src.review.feedback.submit_feedback; the store's rules decide what is allowed (see
        docs/feedback-review-queue.md). Returns the stored record. NotFoundError for an unknown chunk
        or model version; InvalidInputError with the reason for a malformed or refused verdict, in
        which case nothing is written.
        """
        try:
            request = FeedbackRequest.model_validate(request)
            submission = FeedbackSubmission(
                chunk_id=request.chunk_id, category=request.category,
                model_version=self._review_model_version(request.model_version), verdict=request.verdict,
                reviewer=request.reviewer, note=(request.note or "").strip() or None)
        except ValidationError as error:
            raise InvalidInputError(validation_message(error)) from error
        with self._lock:
            try:
                return record_feedback(self.store, submission)
            except KeyError as error:
                raise NotFoundError(f"No stored chunk {submission.chunk_id!r}.") from error
            except ValueError as error:
                raise InvalidInputError(str(error)) from error

    def list_feedback(self, contract_id: str | None = None, chunk_id: str | None = None,
                      category: str | None = None, model_version: str | None = None) -> list[FeedbackRecord]:
        """
        Recorded verdicts given under the model version (default: the server's), oldest first,
        optionally for one contract, chunk or category. NotFoundError for an unknown contract.
        """
        model_version = self._review_model_version(model_version)
        if contract_id is not None:
            self._check_contract(contract_id)
        return self.store.list_feedback(contract_id=contract_id, model_version=model_version, chunk_id=chunk_id,
                                        category=category)

    def list_resolutions(self, contract_id: str | None = None, status: ResolutionStatus | None = None,
                         model_version: str | None = None) -> list[Resolution]:
        """
        How the verdicts given under the model version (default: the server's) resolve, per
        (chunk, category): resolved to a label, or a conflict between reviewers. NotFoundError for
        an unknown contract; InvalidInputError for a status other than resolved/conflict.
        """
        model_version = self._review_model_version(model_version)
        if status is not None and status not in get_args(ResolutionStatus):
            raise InvalidInputError(f"status must be one of {', '.join(get_args(ResolutionStatus))}.")
        if contract_id is not None:
            self._check_contract(contract_id)
        return resolve_stored_feedback(self.store, model_version=model_version, contract_id=contract_id, status=status)
