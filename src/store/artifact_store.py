"""SQLite persistence for contracts, exact chunks, raw model scores, and evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    DateTime, Float, ForeignKey, ForeignKeyConstraint, Integer, String, Text,
    UniqueConstraint, create_engine, select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class ContractRow(Base):
    __tablename__ = "contracts"
    contract_id: Mapped[str] = mapped_column(String, primary_key=True)
    text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ChunkRow(Base):
    __tablename__ = "chunks"
    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    contract_id: Mapped[str] = mapped_column(ForeignKey("contracts.contract_id"), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start: Mapped[int] = mapped_column(Integer, nullable=False)
    end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (UniqueConstraint("contract_id", "chunk_index"),)


class ChunkProbabilityRow(Base):
    __tablename__ = "chunk_probs"
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.chunk_id"), primary_key=True)
    category: Mapped[str] = mapped_column(String, primary_key=True)
    model_version: Mapped[str] = mapped_column(String, primary_key=True)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class EvidenceRow(Base):
    __tablename__ = "evidence"
    evidence_id: Mapped[str] = mapped_column(String, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.chunk_id"), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String, nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    start: Mapped[int] = mapped_column(Integer, nullable=False)
    end: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)


class WeightVersionRow(Base):
    __tablename__ = "weight_versions"
    weight_version_id: Mapped[str] = mapped_column(String, primary_key=True)
    values_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ClauseScoreRow(Base):
    __tablename__ = "clause_scores"
    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    category: Mapped[str] = mapped_column(String, primary_key=True)
    model_version: Mapped[str] = mapped_column(String, primary_key=True)
    weight_version_id: Mapped[str] = mapped_column(ForeignKey("weight_versions.weight_version_id"), primary_key=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String, nullable=False)
    __table_args__ = (ForeignKeyConstraint(
        ["chunk_id", "category", "model_version"],
        ["chunk_probs.chunk_id", "chunk_probs.category", "chunk_probs.model_version"],
    ),)


class ContractScoreRow(Base):
    __tablename__ = "contract_scores"
    contract_id: Mapped[str] = mapped_column(ForeignKey("contracts.contract_id"), primary_key=True)
    model_version: Mapped[str] = mapped_column(String, primary_key=True)
    weight_version_id: Mapped[str] = mapped_column(ForeignKey("weight_versions.weight_version_id"), primary_key=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int | None] = mapped_column(Integer)


class ArtifactStore:
    """Small relational store; callers supply explicit stable IDs and versions."""

    def __init__(self, database: str | Path):
        database = str(database)
        if database.startswith("sqlite:"):
            url = database
        else:
            path = Path(database).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{path.as_posix()}"
        self.engine = create_engine(url, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def put_contract(self, contract_id: str, text: str) -> None:
        import hashlib
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self.Session.begin() as session:
            existing = session.get(ContractRow, contract_id)
            if existing and existing.text_sha256 != digest:
                raise ValueError(f"contract {contract_id!r} already exists with different text")
            if not existing:
                session.add(ContractRow(contract_id=contract_id, text_sha256=digest, text=text))

    def put_chunk(self, *, chunk_id: str, contract_id: str, chunk_index: int,
                  start: int, end: int, text: str) -> None:
        if start < 0 or end <= start or end - start != len(text):
            raise ValueError("chunk offsets must be nonempty and match text length")
        with self.Session.begin() as session:
            if not session.get(ContractRow, contract_id):
                raise KeyError(f"unknown contract {contract_id!r}")
            contract = session.get(ContractRow, contract_id)
            if contract.text[start:end] != text:
                raise ValueError("chunk text does not match its source contract offsets")
            row = session.get(ChunkRow, chunk_id)
            values = (contract_id, chunk_index, start, end, text)
            if row and (row.contract_id, row.chunk_index, row.start, row.end, row.text) != values:
                raise ValueError(f"chunk ID {chunk_id!r} already has different content")
            if not row:
                session.add(ChunkRow(chunk_id=chunk_id, contract_id=contract_id,
                                     chunk_index=chunk_index, start=start, end=end, text=text))

    def put_probabilities(self, chunk_id: str, model_version: str,
                          categories: list[str], probabilities: list[float]) -> None:
        if len(categories) != len(probabilities) or len(set(categories)) != len(categories):
            raise ValueError("categories and probabilities must be aligned and unique")
        if any(not 0 <= float(value) <= 1 for value in probabilities):
            raise ValueError("probabilities must be in [0, 1]")
        with self.Session.begin() as session:
            if not session.get(ChunkRow, chunk_id):
                raise KeyError(f"unknown chunk {chunk_id!r}")
            for category, value in zip(categories, probabilities):
                key = (chunk_id, category, model_version)
                row = session.get(ChunkProbabilityRow, key)
                if row and abs(row.probability - float(value)) > 1e-12:
                    raise ValueError(f"prediction already exists with different value for {key}")
                if not row:
                    session.add(ChunkProbabilityRow(chunk_id=chunk_id, category=category,
                                                    model_version=model_version, probability=float(value)))

    def get_probabilities(self, chunk_id: str, model_version: str) -> dict[str, float]:
        with self.Session() as session:
            rows = session.scalars(select(ChunkProbabilityRow).where(
                ChunkProbabilityRow.chunk_id == chunk_id,
                ChunkProbabilityRow.model_version == model_version,
            )).all()
            return {row.category: row.probability for row in rows}

    def add_evidence(self, *, evidence_id: str, chunk_id: str, category: str,
                     quote: str, start: int, end: int, model_version: str) -> None:
        if end <= start or len(quote) != end - start:
            raise ValueError("evidence offsets must be nonempty and match quote length")
        with self.Session.begin() as session:
            chunk = session.get(ChunkRow, chunk_id)
            if not chunk:
                raise KeyError(f"unknown chunk {chunk_id!r}")
            contract = session.get(ContractRow, chunk.contract_id)
            if end > len(contract.text) or contract.text[start:end] != quote:
                raise ValueError("evidence quote does not match its absolute contract offsets")
            row = session.get(EvidenceRow, evidence_id)
            values = (chunk_id, category, quote, start, end, model_version)
            if row and (row.chunk_id, row.category, row.quote, row.start, row.end, row.model_version) != values:
                raise ValueError(f"evidence ID {evidence_id!r} already exists with different content")
            if not row:
                session.add(EvidenceRow(evidence_id=evidence_id, chunk_id=chunk_id,
                                        category=category, quote=quote, start=start, end=end,
                                        model_version=model_version))

    def close(self) -> None:
        self.engine.dispose()
