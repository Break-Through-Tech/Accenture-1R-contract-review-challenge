"""
SQLAlchemy table definitions for the artifact store.

Every table is keyed by caller-supplied stable IDs and versions, so writes can be idempotent:
writing the same row twice is a no-op and writing different content under an existing key fails.

contracts            one row per contract: exact text, its hash, optional upload metadata
chunks               exact slices of a contract (text == contract.text[start:end])
model_versions       per model version: name, seed and the per-category thresholds it was run with
chunk_probs          raw probability per (chunk, category, model_version), for every category
evidence             grounded quotes for detected clauses, at absolute contract offsets
unknown_predictions  detected clauses that could not be grounded (the Unknown queue)
weight_versions      immutable risk WeightSets, stored as canonical JSON plus a content hash
clause_scores        one ClauseScore per detected clause, model version and weight version
contract_scores      one ContractScore per contract, model version and weight version (rank is computed on read)
feedback             reviewer verdicts (accept / reject / missing) on a (chunk, category) prediction

There are no migrations: a schema change means deleting and recreating a local database.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, Float, ForeignKey, ForeignKeyConstraint, Integer, String,
    Text, UniqueConstraint, event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime (column default)."""
    return datetime.now(timezone.utc)


def enable_sqlite_foreign_keys(engine: Engine) -> None:
    """
    Turn on foreign-key enforcement for every new connection of this one engine.
    SQLite ignores foreign keys unless each connection asks; scoped to one engine so other
    SQLAlchemy users in the process (e.g. MLflow's tracking store) are untouched.
    """
    def _on_connect(dbapi_connection, connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _on_connect)


class Base(DeclarativeBase):
    """Declarative base for every artifact-store table."""


# ----- Contracts and chunks -----
class ContractRow(Base):
    """A contract's exact text plus optional source metadata for uploaded documents."""
    __tablename__ = "contracts"
    contract_id: Mapped[str] = mapped_column(String, primary_key=True)
    text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str | None] = mapped_column(String)
    source_type: Mapped[str | None] = mapped_column(String)
    page_starts_json: Mapped[str | None] = mapped_column(Text)
    ocr_used: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ChunkRow(Base):
    """One exact slice of a contract; chunk IDs are stable and supplied by the caller."""
    __tablename__ = "chunks"
    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    contract_id: Mapped[str] = mapped_column(ForeignKey("contracts.contract_id"), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start: Mapped[int] = mapped_column(Integer, nullable=False)
    end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (UniqueConstraint("contract_id", "chunk_index"),)


# ----- Model outputs -----
class ModelVersionRow(Base):
    """What rescoring needs to know about a model version without loading the model."""
    __tablename__ = "model_versions"
    model_version: Mapped[str] = mapped_column(String, primary_key=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    seed: Mapped[int | None] = mapped_column(Integer)
    thresholds_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ChunkProbabilityRow(Base):
    """Raw classifier probability for one (chunk, category) under one model version."""
    __tablename__ = "chunk_probs"
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.chunk_id"), primary_key=True)
    category: Mapped[str] = mapped_column(String, primary_key=True)
    model_version: Mapped[str] = mapped_column(String, primary_key=True)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class EvidenceRow(Base):
    """A grounded quote for a detected (chunk, category); start/end are absolute contract offsets."""
    __tablename__ = "evidence"
    evidence_id: Mapped[str] = mapped_column(String, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.chunk_id"), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String, nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    start: Mapped[int] = mapped_column(Integer, nullable=False)
    end: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    __table_args__ = (
        UniqueConstraint("chunk_id", "category", "model_version"),
        ForeignKeyConstraint(
            ["chunk_id", "category", "model_version"],
            ["chunk_probs.chunk_id", "chunk_probs.category", "chunk_probs.model_version"],
        ),
    )


class UnknownPredictionRow(Base):
    """A detected (chunk, category) whose evidence could not be grounded: one Unknown-queue item."""
    __tablename__ = "unknown_predictions"
    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    category: Mapped[str] = mapped_column(String, primary_key=True)
    model_version: Mapped[str] = mapped_column(String, primary_key=True)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    __table_args__ = (
        CheckConstraint("reason IN ('no_cue_match', 'span_mismatch')", name="ck_unknown_reason"),
        ForeignKeyConstraint(
            ["chunk_id", "category", "model_version"],
            ["chunk_probs.chunk_id", "chunk_probs.category", "chunk_probs.model_version"],
        ),
    )


# ----- Risk scores -----
class WeightVersionRow(Base):
    """An immutable WeightSet: canonical JSON plus its content hash."""
    __tablename__ = "weight_versions"
    weight_version_id: Mapped[str] = mapped_column(String, primary_key=True)
    values_json: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ClauseScoreRow(Base):
    """One scored clause; contract_id is denormalised from the chunk for per-contract queries."""
    __tablename__ = "clause_scores"
    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    category: Mapped[str] = mapped_column(String, primary_key=True)
    model_version: Mapped[str] = mapped_column(String, primary_key=True)
    weight_version_id: Mapped[str] = mapped_column(ForeignKey("weight_versions.weight_version_id"), primary_key=True)
    contract_id: Mapped[str] = mapped_column(ForeignKey("contracts.contract_id"), nullable=False, index=True)
    signals_json: Mapped[str] = mapped_column(Text, nullable=False)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    rating: Mapped[str] = mapped_column(String, nullable=False)
    has_evidence: Mapped[bool] = mapped_column(Boolean, nullable=False)
    weight_set_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    __table_args__ = (
        CheckConstraint("rating IN ('Low', 'Medium', 'High')", name="ck_clause_rating"),
        ForeignKeyConstraint(
            ["chunk_id", "category", "model_version"],
            ["chunk_probs.chunk_id", "chunk_probs.category", "chunk_probs.model_version"],
        ),
    )


class ContractScoreRow(Base):
    """One contract's triage score; top_clauses_json lists [chunk_index, category] keys into clause_scores."""
    __tablename__ = "contract_scores"
    contract_id: Mapped[str] = mapped_column(ForeignKey("contracts.contract_id"), primary_key=True)
    model_version: Mapped[str] = mapped_column(String, primary_key=True)
    weight_version_id: Mapped[str] = mapped_column(ForeignKey("weight_versions.weight_version_id"), primary_key=True)
    triage_score: Mapped[float] = mapped_column(Float, nullable=False)
    high_count: Mapped[int] = mapped_column(Integer, nullable=False)
    medium_count: Mapped[int] = mapped_column(Integer, nullable=False)
    low_count: Mapped[int] = mapped_column(Integer, nullable=False)
    unknown_count: Mapped[int] = mapped_column(Integer, nullable=False)
    top_clauses_json: Mapped[str] = mapped_column(Text, nullable=False)
    weight_set_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


# ----- Human in the loop -----
class FeedbackRow(Base):
    """A reviewer's verdict on one (chunk, category) prediction; an append-only log."""
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.chunk_id"), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    verdict: Mapped[str] = mapped_column(String, nullable=False)
    reviewer: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    note: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        CheckConstraint("verdict IN ('accept', 'reject', 'missing')", name="ck_feedback_verdict"),
    )
