"""
Scoring-job status for the API, kept in its own table (api_jobs) in the artifact store's SQLite file.

A job is one contract scored under one model version. Its status moves queued -> running ->
complete or failed, and only the service moves it, after the pipeline result is committed to the
artifact store (complete) or after anything at all went wrong (failed). So "complete" always
means the store holds the result.

The table belongs to the API, not to the artifact store: it has its own declarative base, the
store's schema check never looks at it, and it holds no pipeline output. Like the store it has
no migrations; a database whose api_jobs table has other columns is refused.

How everything comes together:
JobRecord: the Pydantic model the service and the endpoints return
JobStore:  queue(), claim(), finish(), get(), list(), and recover() (run at startup: a job left
           queued or running by a stopped server is marked complete if its result is in the store,
           and failed otherwise)
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import CheckConstraint, DateTime, String, Text, inspect, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

JobStatus = Literal["queued", "running", "complete", "failed"]
# Shown for jobs a stopped server never finished
INTERRUPTED = "Scoring was interrupted (the server stopped before it finished). Upload the contract again."


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class JobBase(DeclarativeBase):
    """Declarative base for the API's own tables (separate from the artifact store's)."""


class JobRow(JobBase):
    """One contract's scoring job under one model version."""
    __tablename__ = "api_jobs"
    contract_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_version: Mapped[str] = mapped_column(String, primary_key=True)
    weight_version_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    source_name: Mapped[str | None] = mapped_column(String)
    source_type: Mapped[str | None] = mapped_column(String)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    __table_args__ = (
        CheckConstraint("status IN ('queued', 'running', 'complete', 'failed')", name="ck_api_job_status"),
    )


class JobRecord(BaseModel):
    """
    A scoring job's status. error is a reviewer-readable reason, set only when status is failed.
    weight_version_id is the weight version the contract is scored with on upload.
    """
    model_config = ConfigDict(from_attributes=True)

    contract_id: str
    model_version: str
    weight_version_id: str
    status: JobStatus
    error: str | None = None
    source_name: str | None = None
    source_type: str | None = None
    submitted_at: datetime
    updated_at: datetime

    # SQLite drops the time zone; every stored time is UTC, so say so on the way out
    @field_validator("submitted_at", "updated_at")
    @classmethod
    def as_utc(cls, value: datetime) -> datetime:
        """Mark a naive timestamp read back from SQLite as UTC."""
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class JobStore:
    """Reads and writes api_jobs on an existing SQLAlchemy engine (the artifact store's)."""

    def __init__(self, engine: Engine) -> None:
        """Check the table's columns, create it if missing, and open sessions on engine."""
        table = JobBase.metadata.tables[JobRow.__tablename__]
        inspector = inspect(engine)
        if table.name in inspector.get_table_names():
            stored = {column["name"] for column in inspector.get_columns(table.name)}
            if stored != set(table.columns.keys()):
                raise RuntimeError(f"table {table.name!r} in {engine.url} has an outdated schema; there are no "
                                   "migrations, so delete the database file and rerun to recreate it")
        JobBase.metadata.create_all(engine)
        self.Session = sessionmaker(engine, expire_on_commit=False)

    def get(self, contract_id: str, model_version: str) -> JobRecord | None:
        """The job for one contract and model version, or None if there is none."""
        with self.Session() as session:
            row = session.get(JobRow, (contract_id, model_version))
            return None if row is None else JobRecord.model_validate(row)

    def list(self, model_version: str | None = None) -> list[JobRecord]:
        """Every job (optionally for one model version), most recently submitted first."""
        with self.Session() as session:
            query = select(JobRow).order_by(JobRow.submitted_at.desc(), JobRow.contract_id)
            if model_version is not None:
                query = query.where(JobRow.model_version == model_version)
            return [JobRecord.model_validate(row) for row in session.scalars(query)]

    def queue(self, contract_id: str, model_version: str, weight_version_id: str, *,
              source_name: str | None, source_type: str | None) -> JobRecord:
        """Create the job as queued, or reset an existing one to queued (clearing its error)."""
        now = utc_now()
        with self.Session.begin() as session:
            row = session.get(JobRow, (contract_id, model_version))
            if row is None:
                row = JobRow(contract_id=contract_id, model_version=model_version)
                session.add(row)
            row.weight_version_id, row.status, row.error = weight_version_id, "queued", None
            row.source_name, row.source_type, row.submitted_at, row.updated_at = source_name, source_type, now, now
            session.flush()
            return JobRecord.model_validate(row)

    def claim(self, contract_id: str, model_version: str) -> bool:
        """Move a queued job to running; False (and no change) if it is not queued."""
        with self.Session.begin() as session:
            result = session.execute(update(JobRow).where(
                JobRow.contract_id == contract_id, JobRow.model_version == model_version, JobRow.status == "queued",
            ).values(status="running", updated_at=utc_now()))
            return result.rowcount == 1

    def finish(self, contract_id: str, model_version: str, status: Literal["complete", "failed"],
               error: str | None = None) -> JobRecord:
        """Record a job's outcome; KeyError if the job doesn't exist."""
        with self.Session.begin() as session:
            row = session.get(JobRow, (contract_id, model_version))
            if row is None:
                raise KeyError(f"no job for {contract_id!r} under {model_version!r}")
            row.status, row.error, row.updated_at = status, error if status == "failed" else None, utc_now()
            session.flush()
            return JobRecord.model_validate(row)

    def recover(self, is_stored: Callable[[str, str], bool]) -> int:
        """
        Settle jobs a previous server process left queued or running: complete if is_stored(contract_id,
        model_version) says the store holds the result, failed (INTERRUPTED) otherwise. Returns how many
        jobs were settled. Run once at startup, before any new job is queued.
        """
        with self.Session.begin() as session:
            stale = session.scalars(select(JobRow).where(JobRow.status.in_(("queued", "running")))).all()
            for row in stale:
                stored = is_stored(row.contract_id, row.model_version)
                row.status = "complete" if stored else "failed"
                row.error = None if stored else INTERRUPTED
                row.updated_at = utc_now()
            return len(stale)
