"""
SQLite persistence for every pipeline artifact, so weights can be retuned and contracts rescored
without rerunning the classifier.

How everything comes together:
save_inference_result(): writes one ContractInferencePipeline.predict_contract result (contract,
                         chunks, raw probabilities, evidence, Unknowns, weight version, clause
                         scores, contract score) in a single transaction
get_clause_inputs():     rebuilds the ClauseInput rows scoring reads, from stored probabilities
                         and evidence (see clausetriage/store/rescoring.py for rescoring under new weights)
save_scores():           writes clause and contract scores under a weight version
read methods:            list_contracts, get_contract, get_contract_detail, list_unknowns,
                         list_contract_scores (ranked), ... return the Pydantic models in records.py
add_feedback():          appends a reviewer verdict after checking it fits what the model predicted
                         (the review queue, resolution and export built on it live in clausetriage/review/)

Every write is idempotent: writing the same content again is a no-op, and writing different
content under an existing key raises ValueError and rolls the whole transaction back. Weight
versions are immutable (same id, different content fails). Foreign keys are enforced.

Chunk IDs are f"{contract_id}:{chunk_index}" and evidence IDs f"{chunk_id}|{category}|{model_version}"
when the store derives them. Use a registry-qualified model version such as "clause-classifier:3".
There are no migrations: after a schema change, delete and recreate the local database file.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session, sessionmaker

from clausetriage.evidence.extractor import GroundedEvidence, Unknown
from clausetriage.evidence.schema import Evidence
from clausetriage.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from clausetriage.risk.contract import ContractScore
from clausetriage.risk.schema import ClauseInput, ClauseScore, WeightSet
from clausetriage.store.records import (
    ChunkRecord, ContractDetail, ContractRecord, ContractSummary, DetectionKind,
    EvidenceRecord, FeedbackRecord, ModelVersionRecord, UnknownItem, Verdict, WeightVersionRecord,
)
from clausetriage.store.tables import (
    Base, ChunkProbabilityRow, ChunkRow, ClauseScoreRow, ContractRow,
    ContractScoreRow, EvidenceRow, FeedbackRow, ModelVersionRow, UnknownPredictionRow,
    WeightVersionRow, enable_sqlite_foreign_keys, utc_now,
)

# Stored floats must match to this tolerance for a rewrite to count as identical
FLOAT_TOLERANCE = 1e-12
# A stored detection may sit this far below its threshold (float32 model output vs float64 threshold)
THRESHOLD_TOLERANCE = 1e-6
# Columns that record when a row was written; never part of a row's identity
_TIMESTAMP_FIELDS = frozenset({"created_at", "timestamp"})

__all__ = [
    "ArtifactStore", "Base", "ChunkProbabilityRow", "ChunkRow", "ClauseScoreRow", "ContractRow",
    "ContractScoreRow", "EvidenceRow", "chunk_id_for", "evidence_id_for",
]


def chunk_id_for(contract_id: str, chunk_index: int) -> str:
    """Stable chunk ID the store derives for a pipeline chunk."""
    return f"{contract_id}:{chunk_index}"


def evidence_id_for(chunk_id: str, category: str, model_version: str) -> str:
    """Stable evidence ID the store derives for a grounded (chunk, category) under a model version."""
    return f"{chunk_id}|{category}|{model_version}"


def _sha256(text: str) -> str:
    """Hex SHA-256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    """Deterministic JSON (sorted keys, no whitespace) so equal content always serialises equally."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _same_value(old: Any, new: Any) -> bool:
    """Equality, with a tiny tolerance for floats."""
    if isinstance(old, float) or isinstance(new, float):
        return old is not None and new is not None and math.isclose(old, new, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)
    return old == new


def _merge(session: Session, row_class: type, rows: Iterable[dict[str, Any]], key_fields: Sequence[str],
           existing: Iterable[Any], label: str) -> int:
    """
    Insert each row whose key is new and return how many were inserted. A row whose key already
    exists (stored or earlier in the batch) must hold the same values, or ValueError is raised.
    """
    by_key = {tuple(getattr(row, field) for field in key_fields): row for row in existing}
    inserted = 0
    for values in rows:
        key = tuple(values[field] for field in key_fields)
        row = by_key.get(key)
        if row is None:
            row = row_class(**values)
            session.add(row)
            by_key[key] = row
            inserted += 1
            continue
        changed = [field for field, value in values.items()
                   if field not in _TIMESTAMP_FIELDS and not _same_value(getattr(row, field), value)]
        if changed:
            raise ValueError(f"{label} {key!r} already exists with a different value for {', '.join(changed)}")
    # Flush per table so inserts always follow foreign-key order
    session.flush()
    return inserted


def _check_category(category: str) -> str:
    """Reject anything that is not an exact canonical CUAD category name."""
    if category not in CATEGORY_TO_INDEX:
        raise ValueError(f"unknown CUAD category {category!r}")
    return category


def _check_probability(value: float, label: str) -> float:
    """Return value as a float, rejecting NaN and anything outside [0, 1]."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {value}")
    return value


def _check_page_starts(page_starts: Sequence[int], text_length: int) -> tuple[int, ...]:
    """Page starts must be non-decreasing offsets inside the text."""
    starts = tuple(int(start) for start in page_starts)
    if any(not 0 <= start <= text_length for start in starts):
        raise ValueError("page_starts must lie within the contract text")
    if any(later < earlier for earlier, later in zip(starts, starts[1:])):
        raise ValueError("page_starts must be non-decreasing")
    return starts


def _as_utc(value: datetime) -> datetime:
    """
    A timezone-aware UTC datetime. Naive values are taken as UTC: SQLite drops the offset on the way
    back, and the store only ever writes UTC, so a naive value read from the database is UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _detection_order(chunk_index: int, category: str) -> tuple[int, int]:
    """Pipeline order of a detection: chunk first, then category column (np.argwhere order)."""
    return chunk_index, CATEGORY_TO_INDEX[category]


class ArtifactStore:
    """
    Relational store for contracts, chunks, raw probabilities, evidence, Unknowns, weight versions,
    risk scores and reviewer feedback. Callers supply stable IDs and versions.
    """

    def __init__(self, database: str | Path) -> None:
        """Open (and create if needed) a SQLite database from a file path or a sqlite: URL."""
        database = str(database)
        if database.startswith("sqlite:"):
            url = database
        else:
            path = Path(database).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{path.as_posix()}"
        self.engine = create_engine(url, future=True)
        enable_sqlite_foreign_keys(self.engine)
        self._check_schema()
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def _check_schema(self) -> None:
        """
        Refuse to open a database whose existing tables have other columns than this code expects.
        There are no migrations, so the fix is to delete the file and let the store recreate it.
        """
        inspector = inspect(self.engine)
        existing = set(inspector.get_table_names())
        for name, table in Base.metadata.tables.items():
            if name not in existing:
                continue
            stored = {column["name"] for column in inspector.get_columns(name)}
            if stored != set(table.columns.keys()):
                raise RuntimeError(f"table {name!r} in {self.engine.url} has an outdated schema; the store has no "
                                   "migrations, so delete the database file and rerun to recreate it")

    def close(self) -> None:
        """Release every pooled connection."""
        self.engine.dispose()

    # ----- Contracts and chunks -----
    def put_contract(self, contract_id: str, text: str, *, source_name: str | None = None,
                     source_type: str | None = None, page_starts: Sequence[int] | None = None,
                     ocr_used: bool | None = None) -> None:
        """
        Store a contract's exact text and optional upload metadata (e.g. source_type "pdf",
        page_starts[i] = offset where page i + 1 begins, ocr_used). Rewriting the same text is a
        no-op; different text fails. Metadata is fixed at the first write: a later call that passes
        any metadata must pass exactly the stored values, and a call passing none leaves it as is.
        """
        with self.Session.begin() as session:
            self._put_contract(session, contract_id, text, source_name=source_name, source_type=source_type,
                               page_starts=page_starts, ocr_used=ocr_used)

    def _put_contract(self, session: Session, contract_id: str, text: str, *, source_name: str | None,
                      source_type: str | None, page_starts: Sequence[int] | None,
                      ocr_used: bool | None) -> ContractRow:
        """Insert or verify one contract row inside an open transaction."""
        if not isinstance(contract_id, str) or not contract_id.strip():
            raise ValueError("contract_id must be a non-empty string")
        if not isinstance(text, str):
            raise TypeError("contract text must be a string")
        starts_json = None if page_starts is None else json.dumps(list(_check_page_starts(page_starts, len(text))))
        metadata = (source_name, source_type, starts_json, None if ocr_used is None else bool(ocr_used))
        digest = _sha256(text)

        # Same text is a no-op; metadata, when given, must match what was stored first
        existing = session.get(ContractRow, contract_id)
        if existing is not None:
            if existing.text_sha256 != digest:
                raise ValueError(f"contract {contract_id!r} already exists with different text")
            stored = (existing.source_name, existing.source_type, existing.page_starts_json, existing.ocr_used)
            if any(value is not None for value in metadata) and metadata != stored:
                raise ValueError(f"contract {contract_id!r} already exists with different source metadata")
            return existing
        row = ContractRow(contract_id=contract_id, text_sha256=digest, text=text, source_name=source_name,
                          source_type=source_type, page_starts_json=starts_json, ocr_used=metadata[3])
        session.add(row)
        session.flush()
        return row

    def put_chunk(self, *, chunk_id: str, contract_id: str, chunk_index: int,
                  start: int, end: int, text: str) -> None:
        """Store one chunk; its text must equal the contract text at [start:end]."""
        with self.Session.begin() as session:
            contract = session.get(ContractRow, contract_id)
            if contract is None:
                raise KeyError(f"unknown contract {contract_id!r}")
            row = self._chunk_values(contract, chunk_id, chunk_index, start, end, text)
            _merge(session, ChunkRow, [row], ["chunk_id"], session.scalars(
                select(ChunkRow).where(ChunkRow.chunk_id == chunk_id)), "chunk")

    @staticmethod
    def _chunk_values(contract: ContractRow, chunk_id: str, chunk_index: int, start: int, end: int,
                      text: str) -> dict[str, Any]:
        """Validate a chunk against its contract text and return its column values."""
        if start < 0 or end <= start or end - start != len(text):
            raise ValueError("chunk offsets must be nonempty and match text length")
        if contract.text[start:end] != text:
            raise ValueError("chunk text does not match its source contract offsets")
        return dict(chunk_id=chunk_id, contract_id=contract.contract_id, chunk_index=int(chunk_index),
                    start=int(start), end=int(end), text=text)

    # ----- Model versions and raw probabilities -----
    def put_model_version(self, model_version: str, *, model_name: str, thresholds: Mapping[str, float],
                          seed: int | None = None) -> None:
        """
        Record a model version's name, seed and per-category thresholds (one per CUAD category).
        Immutable: the same version with different content raises ValueError.
        """
        with self.Session.begin() as session:
            self._put_model_version(session, model_version, model_name=model_name, thresholds=thresholds, seed=seed)

    @staticmethod
    def _put_model_version(session: Session, model_version: str, *, model_name: str,
                           thresholds: Mapping[str, float], seed: int | None) -> dict[str, float]:
        """Insert or verify one model-version row inside an open transaction; returns the thresholds."""
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version must be a non-empty string")
        if set(thresholds) != set(CATEGORIES):
            raise ValueError("thresholds must hold exactly one value per CUAD category")
        clean = {category: _check_probability(thresholds[category], f"threshold for {category!r}")
                 for category in CATEGORIES}
        row = dict(model_version=model_version, model_name=model_name,
                   seed=None if seed is None else int(seed), thresholds_json=_canonical_json(clean))
        _merge(session, ModelVersionRow, [row], ["model_version"],
               session.scalars(select(ModelVersionRow).where(ModelVersionRow.model_version == model_version)),
               "model version")
        return clean

    def get_model_version(self, model_version: str) -> ModelVersionRecord:
        """Return a stored model version; KeyError if it was never stored."""
        with self.Session() as session:
            row = session.get(ModelVersionRow, model_version)
            if row is None:
                raise KeyError(f"unknown model version {model_version!r}")
            return self._model_version_record(row)

    def list_model_versions(self) -> list[ModelVersionRecord]:
        """Every stored model version, oldest first."""
        with self.Session() as session:
            rows = session.scalars(select(ModelVersionRow).order_by(ModelVersionRow.created_at,
                                                                    ModelVersionRow.model_version))
            return [self._model_version_record(row) for row in rows]

    @staticmethod
    def _model_version_record(row: ModelVersionRow) -> ModelVersionRecord:
        """Convert a model-version row to its record."""
        return ModelVersionRecord(model_version=row.model_version, model_name=row.model_name, seed=row.seed,
                                  thresholds=json.loads(row.thresholds_json), created_at=row.created_at)

    def put_probabilities(self, chunk_id: str, model_version: str,
                          categories: list[str], probabilities: list[float]) -> None:
        """Store raw probabilities for one chunk; a different value for a stored key raises ValueError."""
        if len(categories) != len(probabilities) or len(set(categories)) != len(categories):
            raise ValueError("categories and probabilities must be aligned and unique")
        rows = [dict(chunk_id=chunk_id, category=_check_category(category), model_version=model_version,
                     probability=_check_probability(value, "probabilities"))
                for category, value in zip(categories, probabilities)]
        with self.Session.begin() as session:
            if not session.get(ChunkRow, chunk_id):
                raise KeyError(f"unknown chunk {chunk_id!r}")
            existing = session.scalars(select(ChunkProbabilityRow).where(
                ChunkProbabilityRow.chunk_id == chunk_id, ChunkProbabilityRow.model_version == model_version))
            _merge(session, ChunkProbabilityRow, rows, ["chunk_id", "category", "model_version"],
                   existing, "prediction")

    def get_probabilities(self, chunk_id: str, model_version: str) -> dict[str, float]:
        """Category -> raw probability for one chunk under one model version (empty if none stored)."""
        with self.Session() as session:
            rows = session.scalars(select(ChunkProbabilityRow).where(
                ChunkProbabilityRow.chunk_id == chunk_id,
                ChunkProbabilityRow.model_version == model_version,
            )).all()
            return {row.category: row.probability for row in rows}

    # ----- Evidence -----
    def add_evidence(self, *, evidence_id: str, chunk_id: str, category: str,
                     quote: str, start: int, end: int, model_version: str) -> None:
        """
        Store one grounded quote; the contract text at [start:end] must equal the quote, and the
        (chunk, category, model_version) probability must already be stored.
        """
        if end <= start or len(quote) != end - start:
            raise ValueError("evidence offsets must be nonempty and match quote length")
        _check_category(category)
        with self.Session.begin() as session:
            chunk = session.get(ChunkRow, chunk_id)
            if not chunk:
                raise KeyError(f"unknown chunk {chunk_id!r}")
            contract = session.get(ContractRow, chunk.contract_id)
            if end > len(contract.text) or contract.text[start:end] != quote:
                raise ValueError("evidence quote does not match its absolute contract offsets")
            if session.get(ChunkProbabilityRow, (chunk_id, category, model_version)) is None:
                raise KeyError(f"no stored probability for {(chunk_id, category, model_version)!r}")
            row = dict(evidence_id=evidence_id, chunk_id=chunk_id, category=category, quote=quote,
                       start=start, end=end, model_version=model_version)
            _merge(session, EvidenceRow, [row], ["evidence_id"],
                   session.scalars(select(EvidenceRow).where(EvidenceRow.evidence_id == evidence_id)), "evidence ID")

    def get_evidence(self, contract_id: str, model_version: str) -> list[EvidenceRecord]:
        """A contract's grounded evidence under one model version, in chunk then category order."""
        with self.Session() as session:
            query = (select(EvidenceRow, ChunkRow).join(ChunkRow, EvidenceRow.chunk_id == ChunkRow.chunk_id)
                     .where(ChunkRow.contract_id == contract_id, EvidenceRow.model_version == model_version))
            rows = session.execute(query).all()
            records = [EvidenceRecord(evidence_id=ev.evidence_id, contract_id=chunk.contract_id,
                                      chunk_id=chunk.chunk_id, chunk_index=chunk.chunk_index, category=ev.category,
                                      model_version=ev.model_version, quote=ev.quote, start=ev.start, end=ev.end)
                       for ev, chunk in rows]
            return sorted(records, key=lambda r: (*_detection_order(r.chunk_index, r.category), r.start))

    # ----- Whole pipeline results -----
    def save_inference_result(self, result: Mapping[str, Any], *, text: str, model_version: str,
                              thresholds: Mapping[str, float], weight_set: WeightSet,
                              source_name: str | None = None, source_type: str | None = None,
                              page_starts: Sequence[int] | None = None,
                              ocr_used: bool | None = None) -> ContractScore:
        """
        Persist one ContractInferencePipeline.predict_contract result atomically and idempotently:
        the contract (with optional upload metadata), its chunks, every raw probability, the model
        version's thresholds, grounded evidence, Unknowns, the weight version, clause scores and
        the contract score. Saving the same result twice is a no-op; any conflict with stored
        content raises ValueError and writes nothing.

        text: the exact contract text the pipeline ran on (the result does not carry it)
        model_version: registry-qualified model key, e.g. "clause-classifier:3"
        thresholds: the bundle's per-category thresholds (bundle.thresholds)
        weight_set: the WeightSet the pipeline scored with; every score must carry its id and hash
        Returns the stored ContractScore, ranked among every contract under this model and weight version.
        """
        contract_id = result["contract_id"]
        with self.Session.begin() as session:
            # Contract, model version and weight version
            contract = self._put_contract(session, contract_id, text, source_name=source_name,
                                          source_type=source_type, page_starts=page_starts, ocr_used=ocr_used)
            clean_thresholds = self._put_model_version(session, model_version, model_name=result["model_name"],
                                                       thresholds=thresholds, seed=result.get("seed"))
            self._put_weight_version(session, weight_set)

            # Chunks and probabilities; a contract keeps one chunking across model versions
            chunk_rows, prob_rows, probability_of = [], [], {}
            for chunk in result["chunks"]:
                chunk_id = chunk_id_for(contract_id, chunk["chunk_index"])
                chunk_rows.append(self._chunk_values(contract, chunk_id, chunk["chunk_index"], chunk["start"],
                                                     chunk["end"], chunk["text"]))
                if set(chunk["probabilities"]) != set(CATEGORIES):
                    raise ValueError(f"chunk {chunk_id!r} must have one probability per CUAD category")
                for category in CATEGORIES:
                    value = _check_probability(chunk["probabilities"][category], f"probability for {category!r}")
                    probability_of[chunk["chunk_index"], category] = value
                    prob_rows.append(dict(chunk_id=chunk_id, category=category, model_version=model_version,
                                          probability=value))
            existing_chunks = session.scalars(select(ChunkRow).where(ChunkRow.contract_id == contract_id)).all()
            if existing_chunks and {c.chunk_index for c in existing_chunks} != {c["chunk_index"] for c in chunk_rows}:
                raise ValueError(f"contract {contract_id!r} is already stored with a different chunking")
            _merge(session, ChunkRow, chunk_rows, ["chunk_id"], existing_chunks, "chunk")
            _merge(session, ChunkProbabilityRow, prob_rows, ["chunk_id", "category", "model_version"],
                   self._scoped(session, ChunkProbabilityRow, contract_id, model_version), "prediction")

            # Detections: grounded evidence and Unknowns, each above its category's threshold
            evidence_rows, unknown_rows, detected = [], [], set()
            records = [(GroundedEvidence.model_validate(g), True) for g in result["grounded"]]
            records += [(Unknown.model_validate(u), False) for u in result["unknown"]]
            for record, grounded in records:
                key = (record.chunk_index, _check_category(record.category))
                if record.contract_id != contract_id or key not in probability_of:
                    raise ValueError(f"detection {key!r} does not belong to a chunk of {contract_id!r}")
                if key in detected:
                    raise ValueError(f"detection {key!r} appears more than once")
                if probability_of[key] < clean_thresholds[record.category] - THRESHOLD_TOLERANCE:
                    raise ValueError(f"detection {key!r} is below its category threshold")
                detected.add(key)
                chunk_id = chunk_id_for(contract_id, record.chunk_index)
                if grounded:
                    if not record.evidence.validate_against(text):
                        raise ValueError(f"evidence for {key!r} does not match the contract text")
                    evidence_rows.append(dict(
                        evidence_id=evidence_id_for(chunk_id, record.category, model_version), chunk_id=chunk_id,
                        category=record.category, quote=record.evidence.quote, start=record.evidence.start,
                        end=record.evidence.end, model_version=model_version))
                else:
                    unknown_rows.append(dict(chunk_id=chunk_id, category=record.category,
                                             model_version=model_version, reason=record.reason))
            stored_evidence = self._scoped(session, EvidenceRow, contract_id, model_version)
            stored_unknowns = self._scoped(session, UnknownPredictionRow, contract_id, model_version)
            stored_keys = {(e.chunk_id, e.category) for e in [*stored_evidence, *stored_unknowns]}
            if stored_keys and stored_keys != {(chunk_id_for(contract_id, i), c) for i, c in detected}:
                raise ValueError(f"contract {contract_id!r} already has different detections under {model_version!r}")
            _merge(session, EvidenceRow, evidence_rows, ["evidence_id"], stored_evidence, "evidence ID")
            _merge(session, UnknownPredictionRow, unknown_rows, ["chunk_id", "category", "model_version"],
                   stored_unknowns, "unknown prediction")

            # Scores: one per detection, all under weight_set
            clause_scores = [ClauseScore.model_validate(s) for s in result["clause_scores"]]
            if {(s.chunk_index, s.category) for s in clause_scores} != detected or len(clause_scores) != len(detected):
                raise ValueError("clause_scores must hold exactly one score per detection")
            self._write_scores(session, model_version, weight_set, clause_scores,
                               [ContractScore.model_validate(result["contract_score"])])
        return self.get_contract_score(contract_id, model_version, weight_set.weight_version_id)

    @staticmethod
    def _scoped(session: Session, row_class: type, contract_id: str, model_version: str) -> list[Any]:
        """Rows of a (chunk_id, ..., model_version) table that belong to one contract and model version."""
        return session.scalars(select(row_class).join(ChunkRow, row_class.chunk_id == ChunkRow.chunk_id).where(
            ChunkRow.contract_id == contract_id, row_class.model_version == model_version)).all()

    # ----- Rebuilding clause inputs and the Unknown queue -----
    def list_unknowns(self, model_version: str, contract_id: str | None = None) -> list[UnknownItem]:
        """
        Unknown-queue items under a model version: predictions that cleared their threshold but have
        no grounded evidence. All contracts if contract_id is None; ordered by contract, chunk, category.
        """
        with self.Session() as session:
            query = (select(UnknownPredictionRow, ChunkRow, ChunkProbabilityRow)
                     .join(ChunkRow, UnknownPredictionRow.chunk_id == ChunkRow.chunk_id)
                     .join(ChunkProbabilityRow, (ChunkProbabilityRow.chunk_id == UnknownPredictionRow.chunk_id)
                           & (ChunkProbabilityRow.category == UnknownPredictionRow.category)
                           & (ChunkProbabilityRow.model_version == UnknownPredictionRow.model_version))
                     .where(UnknownPredictionRow.model_version == model_version))
            if contract_id is not None:
                query = query.where(ChunkRow.contract_id == contract_id)
            model = session.get(ModelVersionRow, model_version)
            thresholds = json.loads(model.thresholds_json) if model is not None else {}
            items = [UnknownItem(contract_id=chunk.contract_id, chunk_id=chunk.chunk_id, chunk_index=chunk.chunk_index,
                                 chunk_start=chunk.start, chunk_end=chunk.end, category=unknown.category,
                                 model_version=model_version, probability=prob.probability,
                                 threshold=thresholds.get(unknown.category), reason=unknown.reason)
                     for unknown, chunk, prob in session.execute(query)]
            return sorted(items, key=lambda i: (i.contract_id, *_detection_order(i.chunk_index, i.category)))

    def get_clause_inputs(self, contract_id: str, model_version: str) -> list[ClauseInput]:
        """
        Rebuild the ClauseInput rows the pipeline scored for one contract, from stored probabilities,
        evidence and Unknowns, in the pipeline's order (grounded first, then Unknowns, each by chunk
        then category). Raises KeyError if the contract was never run under model_version, and
        ValueError if a stored detection sits below its model version's threshold.
        """
        with self.Session() as session:
            if contract_id not in self._contracts_run_under(session, model_version, [contract_id]):
                raise KeyError(f"contract {contract_id!r} has no stored results under {model_version!r}")
            model = session.get(ModelVersionRow, model_version)
            thresholds = json.loads(model.thresholds_json) if model is not None else {}
            probability_of = {(row.chunk_id, row.category): row.probability
                              for row in self._scoped(session, ChunkProbabilityRow, contract_id, model_version)}
            index_of = {row.chunk_id: row.chunk_index for row in
                        session.scalars(select(ChunkRow).where(ChunkRow.contract_id == contract_id))}

            # Grounded clauses carry their Evidence; Unknowns carry None
            detections = [(ev.chunk_id, ev.category, Evidence(quote=ev.quote, start=ev.start, end=ev.end))
                          for ev in self._scoped(session, EvidenceRow, contract_id, model_version)]
            unknowns = [(u.chunk_id, u.category, None)
                        for u in self._scoped(session, UnknownPredictionRow, contract_id, model_version)]
            clauses = []
            for group in (detections, unknowns):
                for chunk_id, category, evidence in sorted(group, key=lambda d: _detection_order(index_of[d[0]], d[1])):
                    probability = probability_of[chunk_id, category]
                    if category in thresholds and probability < thresholds[category] - THRESHOLD_TOLERANCE:
                        raise ValueError(f"stored detection {(chunk_id, category)!r} is below its threshold")
                    clauses.append(ClauseInput(contract_id=contract_id, chunk_index=index_of[chunk_id],
                                               category=category, probability=probability, evidence=evidence))
            return clauses

    def list_run_contract_ids(self, model_version: str) -> list[str]:
        """Contracts with stored results under a model version (probabilities or a contract score), sorted."""
        with self.Session() as session:
            return sorted(self._contracts_run_under(session, model_version))

    @staticmethod
    def _contracts_run_under(session: Session, model_version: str,
                             contract_ids: Sequence[str] | None = None) -> set[str]:
        """Contract IDs with probabilities or a contract score under model_version (optionally restricted)."""
        with_probs = (select(ChunkRow.contract_id)
                      .join(ChunkProbabilityRow, ChunkProbabilityRow.chunk_id == ChunkRow.chunk_id)
                      .where(ChunkProbabilityRow.model_version == model_version))
        with_scores = select(ContractScoreRow.contract_id).where(ContractScoreRow.model_version == model_version)
        if contract_ids is not None:
            with_probs = with_probs.where(ChunkRow.contract_id.in_(contract_ids))
            with_scores = with_scores.where(ContractScoreRow.contract_id.in_(contract_ids))
        return set(session.scalars(with_probs.distinct())) | set(session.scalars(with_scores))

    # ----- Weight versions -----
    def put_weight_version(self, weight_set: WeightSet) -> None:
        """Store a WeightSet; immutable, so the same id with different content raises ValueError."""
        with self.Session.begin() as session:
            self._put_weight_version(session, weight_set)

    @staticmethod
    def _put_weight_version(session: Session, weight_set: WeightSet) -> None:
        """Insert or verify one weight version inside an open transaction."""
        existing = session.get(WeightVersionRow, weight_set.weight_version_id)
        if existing is not None:
            if existing.content_sha256 != weight_set.content_sha256:
                raise ValueError(f"weight version {weight_set.weight_version_id!r} already exists with "
                                 "different content; edits need a new weight_version_id")
            return
        session.add(WeightVersionRow(
            weight_version_id=weight_set.weight_version_id,
            values_json=_canonical_json(weight_set.model_dump(mode="json")),
            content_sha256=weight_set.content_sha256, created_by=weight_set.created_by,
        ))
        session.flush()

    def get_weight_version(self, weight_version_id: str) -> WeightSet:
        """Load a stored WeightSet; KeyError if missing, ValueError if its content no longer matches its hash."""
        with self.Session() as session:
            row = session.get(WeightVersionRow, weight_version_id)
            if row is None:
                raise KeyError(f"unknown weight version {weight_version_id!r}")
            weight_set = WeightSet.model_validate_json(row.values_json)
            if weight_set.content_sha256 != row.content_sha256:
                raise ValueError(f"weight version {weight_version_id!r} does not match its stored hash")
            return weight_set

    def list_weight_versions(self) -> list[WeightVersionRecord]:
        """Every stored weight version's metadata, oldest first."""
        with self.Session() as session:
            rows = session.scalars(select(WeightVersionRow).order_by(WeightVersionRow.created_at,
                                                                     WeightVersionRow.weight_version_id))
            return [WeightVersionRecord(weight_version_id=row.weight_version_id, created_by=row.created_by,
                                        description=json.loads(row.values_json).get("description", ""),
                                        content_sha256=row.content_sha256, created_at=row.created_at)
                    for row in rows]

    # ----- Clause and contract scores -----
    def save_scores(self, model_version: str, weight_set: WeightSet, clause_scores: Iterable[ClauseScore],
                    contract_scores: Iterable[ContractScore]) -> None:
        """
        Store clause and contract scores produced under weight_set in one transaction (the weight
        version is stored too). Every clause score must be a stored detection under model_version.
        Rewriting identical scores is a no-op; different scores under an existing key raise ValueError.
        """
        with self.Session.begin() as session:
            self._put_weight_version(session, weight_set)
            self._write_scores(session, model_version, weight_set, list(clause_scores), list(contract_scores))

    def _write_scores(self, session: Session, model_version: str, weight_set: WeightSet,
                      clause_scores: list[ClauseScore], contract_scores: list[ContractScore]) -> None:
        """Validate and merge score rows inside an open transaction."""
        version = (weight_set.weight_version_id, weight_set.content_sha256)
        for score in [*clause_scores, *contract_scores]:
            if (score.weight_version_id, score.weight_set_sha256) != version:
                raise ValueError(f"score for {score.contract_id!r} was not produced with weight version "
                                 f"{weight_set.weight_version_id!r}")

        # Each clause score must point at a stored detection with the same probability
        clause_rows = []
        for contract_id in sorted({s.contract_id for s in clause_scores}):
            probability_of = {(r.chunk_id, r.category): r.probability
                              for r in self._scoped(session, ChunkProbabilityRow, contract_id, model_version)}
            detected = {(r.chunk_id, r.category) for r in [
                *self._scoped(session, EvidenceRow, contract_id, model_version),
                *self._scoped(session, UnknownPredictionRow, contract_id, model_version)]}
            for score in (s for s in clause_scores if s.contract_id == contract_id):
                key = (chunk_id_for(contract_id, score.chunk_index), score.category)
                if key not in detected:
                    raise ValueError(f"clause score {key!r} is not a stored detection under {model_version!r}")
                if not _same_value(probability_of[key], score.probability):
                    raise ValueError(f"clause score {key!r} has a different probability than the stored one")
                clause_rows.append(dict(
                    chunk_id=key[0], category=score.category, model_version=model_version,
                    weight_version_id=score.weight_version_id, contract_id=contract_id,
                    signals_json=_canonical_json(score.signals), composite_score=score.composite_score,
                    rating=score.rating, has_evidence=score.has_evidence, weight_set_sha256=score.weight_set_sha256))
        existing = session.scalars(select(ClauseScoreRow).where(
            ClauseScoreRow.model_version == model_version,
            ClauseScoreRow.weight_version_id == weight_set.weight_version_id,
            ClauseScoreRow.contract_id.in_(sorted({row["contract_id"] for row in clause_rows})))).all()
        _merge(session, ClauseScoreRow, clause_rows, ["chunk_id", "category", "model_version", "weight_version_id"],
               existing, "clause score")

        # Contract scores; rank is not stored because it depends on which other contracts are scored
        contract_rows = []
        for score in contract_scores:
            if session.get(ContractRow, score.contract_id) is None:
                raise KeyError(f"unknown contract {score.contract_id!r}")
            for clause in score.top_clauses:
                key = (chunk_id_for(score.contract_id, clause.chunk_index), clause.category, model_version,
                       weight_set.weight_version_id)
                if session.get(ClauseScoreRow, key) is None:
                    raise ValueError(f"top clause {key[:2]!r} of {score.contract_id!r} has no stored clause score")
            contract_rows.append(dict(
                contract_id=score.contract_id, model_version=model_version,
                weight_version_id=score.weight_version_id, triage_score=score.triage_score,
                high_count=score.high_count, medium_count=score.medium_count, low_count=score.low_count,
                unknown_count=score.unknown_count, weight_set_sha256=score.weight_set_sha256,
                top_clauses_json=_canonical_json([[c.chunk_index, c.category] for c in score.top_clauses])))
        existing = session.scalars(select(ContractScoreRow).where(
            ContractScoreRow.model_version == model_version,
            ContractScoreRow.weight_version_id == weight_set.weight_version_id,
            ContractScoreRow.contract_id.in_(sorted({row["contract_id"] for row in contract_rows})))).all()
        _merge(session, ContractScoreRow, contract_rows, ["contract_id", "model_version", "weight_version_id"],
               existing, "contract score")

    def get_clause_scores(self, contract_id: str, model_version: str, weight_version_id: str) -> list[ClauseScore]:
        """A contract's clause scores under one model and weight version, in chunk then category order."""
        with self.Session() as session:
            return self._clause_scores(session, model_version, weight_version_id, contract_id)

    @staticmethod
    def _clause_scores(session: Session, model_version: str, weight_version_id: str,
                       contract_id: str | None = None) -> list[ClauseScore]:
        """Rebuild ClauseScore models from stored rows (one contract, or all if contract_id is None)."""
        query = (select(ClauseScoreRow, ChunkRow.chunk_index, ChunkProbabilityRow.probability)
                 .join(ChunkRow, ClauseScoreRow.chunk_id == ChunkRow.chunk_id)
                 .join(ChunkProbabilityRow, (ChunkProbabilityRow.chunk_id == ClauseScoreRow.chunk_id)
                       & (ChunkProbabilityRow.category == ClauseScoreRow.category)
                       & (ChunkProbabilityRow.model_version == ClauseScoreRow.model_version))
                 .where(ClauseScoreRow.model_version == model_version,
                        ClauseScoreRow.weight_version_id == weight_version_id))
        if contract_id is not None:
            query = query.where(ClauseScoreRow.contract_id == contract_id)
        scores = [ClauseScore(contract_id=row.contract_id, chunk_index=chunk_index, category=row.category,
                              probability=probability, signals=json.loads(row.signals_json),
                              composite_score=row.composite_score, rating=row.rating,
                              has_evidence=row.has_evidence, weight_version_id=row.weight_version_id,
                              weight_set_sha256=row.weight_set_sha256)
                  for row, chunk_index, probability in session.execute(query)]
        return sorted(scores, key=lambda s: (s.contract_id, *_detection_order(s.chunk_index, s.category)))

    def list_contract_scores(self, model_version: str, weight_version_id: str) -> list[ContractScore]:
        """
        Every stored contract score under one model and weight version, ranked like
        risk.contract.aggregate_contracts: highest triage_score first, ties by contract_id and
        sharing the best rank (1, 2, 2, 4).
        """
        with self.Session() as session:
            return self._ranked_contract_scores(session, model_version, weight_version_id)

    def get_contract_score(self, contract_id: str, model_version: str,
                           weight_version_id: str) -> ContractScore | None:
        """One contract's score (ranked among all contracts under the same pair), or None if not scored."""
        return next((s for s in self.list_contract_scores(model_version, weight_version_id)
                     if s.contract_id == contract_id), None)

    def _ranked_contract_scores(self, session: Session, model_version: str,
                                weight_version_id: str) -> list[ContractScore]:
        """Rebuild and rank every ContractScore under one model and weight version."""
        rows = session.scalars(select(ContractScoreRow).where(
            ContractScoreRow.model_version == model_version,
            ContractScoreRow.weight_version_id == weight_version_id)).all()
        clause_of = {(s.contract_id, s.chunk_index, s.category): s
                     for s in self._clause_scores(session, model_version, weight_version_id)}
        rows = sorted(rows, key=lambda r: (-r.triage_score, r.contract_id))
        ranked, rank = [], 0
        for position, row in enumerate(rows, start=1):
            if position == 1 or row.triage_score != rows[position - 2].triage_score:
                rank = position
            ranked.append(ContractScore(
                contract_id=row.contract_id, triage_score=row.triage_score, rank=rank,
                high_count=row.high_count, medium_count=row.medium_count, low_count=row.low_count,
                unknown_count=row.unknown_count, weight_version_id=row.weight_version_id,
                weight_set_sha256=row.weight_set_sha256,
                top_clauses=[clause_of[row.contract_id, index, category]
                             for index, category in json.loads(row.top_clauses_json)]))
        return ranked

    # ----- Contract reads -----
    def list_contracts(self) -> list[ContractSummary]:
        """Every stored contract without its text, sorted by contract_id."""
        with self.Session() as session:
            rows = session.scalars(select(ContractRow).order_by(ContractRow.contract_id))
            return [ContractSummary(**self._contract_fields(row)) for row in rows]

    def get_contract(self, contract_id: str) -> ContractRecord:
        """One contract with its text and source metadata; KeyError if missing."""
        with self.Session() as session:
            row = session.get(ContractRow, contract_id)
            if row is None:
                raise KeyError(f"unknown contract {contract_id!r}")
            starts = None if row.page_starts_json is None else tuple(json.loads(row.page_starts_json))
            return ContractRecord(**self._contract_fields(row), text=row.text, page_starts=starts)

    @staticmethod
    def _contract_fields(row: ContractRow) -> dict[str, Any]:
        """Summary fields of a contract row."""
        starts = None if row.page_starts_json is None else json.loads(row.page_starts_json)
        return dict(contract_id=row.contract_id, text_sha256=row.text_sha256, text_length=len(row.text),
                    source_name=row.source_name, source_type=row.source_type,
                    page_count=None if starts is None else len(starts), ocr_used=row.ocr_used,
                    created_at=row.created_at)

    def get_chunks(self, contract_id: str) -> list[ChunkRecord]:
        """A contract's chunks in chunk_index order."""
        with self.Session() as session:
            rows = session.scalars(select(ChunkRow).where(ChunkRow.contract_id == contract_id)
                                   .order_by(ChunkRow.chunk_index))
            return [ChunkRecord(chunk_id=r.chunk_id, contract_id=r.contract_id, chunk_index=r.chunk_index,
                                start=r.start, end=r.end, text=r.text) for r in rows]

    def get_contract_detail(self, contract_id: str, model_version: str, weight_version_id: str) -> ContractDetail:
        """
        A contract with its source metadata, chunks, evidence, Unknowns, clause scores and ranked
        contract score under one model version and weight version. KeyError if the contract is missing.
        """
        return ContractDetail(
            contract=self.get_contract(contract_id), model_version=model_version,
            weight_version_id=weight_version_id, chunks=self.get_chunks(contract_id),
            evidence=self.get_evidence(contract_id, model_version),
            unknowns=self.list_unknowns(model_version, contract_id),
            clause_scores=self.get_clause_scores(contract_id, model_version, weight_version_id),
            contract_score=self.get_contract_score(contract_id, model_version, weight_version_id),
        )

    # ----- Reviewer feedback -----
    def get_chunk(self, chunk_id: str) -> ChunkRecord:
        """One stored chunk; KeyError if missing."""
        with self.Session() as session:
            row = session.get(ChunkRow, chunk_id)
            if row is None:
                raise KeyError(f"unknown chunk {chunk_id!r}")
            return ChunkRecord(chunk_id=row.chunk_id, contract_id=row.contract_id, chunk_index=row.chunk_index,
                               start=row.start, end=row.end, text=row.text)

    def get_detection(self, chunk_id: str, category: str, model_version: str) -> DetectionKind | None:
        """
        How the model reported (chunk, category) under model_version: "grounded" (stored evidence),
        "unknown" (an Unknown-queue item) or None (no stored detection).
        """
        with self.Session() as session:
            return self._detection(session, chunk_id, category, model_version)

    @staticmethod
    def _detection(session: Session, chunk_id: str, category: str, model_version: str) -> DetectionKind | None:
        """get_detection inside an open session."""
        if session.scalar(select(EvidenceRow.evidence_id).where(
                EvidenceRow.chunk_id == chunk_id, EvidenceRow.category == category,
                EvidenceRow.model_version == model_version)) is not None:
            return "grounded"
        if session.get(UnknownPredictionRow, (chunk_id, category, model_version)) is not None:
            return "unknown"
        return None

    def add_feedback(self, *, chunk_id: str, category: str, model_version: str, verdict: Verdict,
                     reviewer: str, note: str | None = None, timestamp: datetime | None = None) -> FeedbackRecord:
        """
        Append a reviewer verdict on one (chunk, category) under the model version that was reviewed.
        The chunk must have been run under model_version. "accept" needs a stored detection (grounded
        evidence or an Unknown); "missing" needs there to be none; "reject" needs a detection or an
        earlier "missing" verdict on the same key, so a reviewer can contradict a claimed miss.
        The reviewer name is stripped of surrounding whitespace. timestamp defaults to now;
        a naive timestamp is taken as UTC, and it may not be earlier than the same reviewer's last
        verdict on the same (chunk, category), so log order and time order always agree per reviewer.
        KeyError for an unknown chunk; ValueError for anything else invalid. Nothing is written on error.
        """
        if verdict not in get_args(Verdict):
            raise ValueError(f"verdict must be one of {get_args(Verdict)}, got {verdict!r}")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("reviewer must be a non-empty string")
        reviewer = reviewer.strip()
        _check_category(category)
        timestamp = utc_now() if timestamp is None else _as_utc(timestamp)
        with self.Session.begin() as session:
            chunk = session.get(ChunkRow, chunk_id)
            if chunk is None:
                raise KeyError(f"unknown chunk {chunk_id!r}")

            # What the verdict refers to must exist: a detection (or a claimed miss, for reject), or none for missing
            if session.get(ChunkProbabilityRow, (chunk_id, category, model_version)) is None:
                raise ValueError(f"chunk {chunk_id!r} has no stored predictions under model version {model_version!r}")
            detection = self._detection(session, chunk_id, category, model_version)
            claimed_missing = verdict == "reject" and detection is None and session.scalar(
                select(FeedbackRow.id).where(FeedbackRow.chunk_id == chunk_id, FeedbackRow.category == category,
                                             FeedbackRow.model_version == model_version,
                                             FeedbackRow.verdict == "missing").limit(1)) is not None
            if verdict in ("accept", "reject") and detection is None and not claimed_missing:
                raise ValueError(f"{verdict!r} needs a prediction, but {model_version!r} did not predict "
                                 f"{category!r} in chunk {chunk_id!r}; use 'missing' for a clause the model missed")
            if verdict == "missing" and detection is not None:
                raise ValueError(f"'missing' is for clauses the model did not predict, but {model_version!r} "
                                 f"predicted {category!r} in chunk {chunk_id!r} ({detection}); use accept or reject")

            # One reviewer's verdicts on a key never go back in time
            last = session.scalar(select(FeedbackRow.timestamp).where(
                FeedbackRow.chunk_id == chunk_id, FeedbackRow.category == category,
                FeedbackRow.reviewer == reviewer).order_by(FeedbackRow.id.desc()).limit(1))
            if last is not None and timestamp < _as_utc(last):
                raise ValueError(f"timestamp {timestamp.isoformat()} is earlier than {reviewer!r}'s last verdict on "
                                 f"{(chunk_id, category)!r} at {_as_utc(last).isoformat()}")
            row = FeedbackRow(chunk_id=chunk_id, category=category, model_version=model_version, verdict=verdict,
                              reviewer=reviewer, note=note, timestamp=timestamp)
            session.add(row)
            session.flush()
            return self._feedback_record(row, chunk.contract_id)

    def list_feedback(self, *, contract_id: str | None = None, model_version: str | None = None,
                      chunk_id: str | None = None, category: str | None = None) -> list[FeedbackRecord]:
        """Stored feedback, optionally filtered by contract, model version, chunk and category, oldest first."""
        with self.Session() as session:
            query = select(FeedbackRow, ChunkRow.contract_id).join(ChunkRow, FeedbackRow.chunk_id == ChunkRow.chunk_id)
            if contract_id is not None:
                query = query.where(ChunkRow.contract_id == contract_id)
            if model_version is not None:
                query = query.where(FeedbackRow.model_version == model_version)
            if chunk_id is not None:
                query = query.where(FeedbackRow.chunk_id == chunk_id)
            if category is not None:
                query = query.where(FeedbackRow.category == category)
            return [self._feedback_record(row, cid) for row, cid in session.execute(query.order_by(FeedbackRow.id))]

    @staticmethod
    def _feedback_record(row: FeedbackRow, contract_id: str) -> FeedbackRecord:
        """Convert a feedback row to its record; the timestamp is always timezone-aware UTC."""
        return FeedbackRecord(id=row.id, contract_id=contract_id, chunk_id=row.chunk_id, category=row.category,
                              model_version=row.model_version, verdict=row.verdict, reviewer=row.reviewer,
                              timestamp=_as_utc(row.timestamp), note=row.note)
