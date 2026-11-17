"""Tests for src/api/jobs.py: job status transitions, restart recovery and the schema check."""
import sqlite3

import pytest
from sqlalchemy import create_engine

from src.api.jobs import INTERRUPTED, JobStore


@pytest.fixture
def jobs(tmp_path):
    """A JobStore on a fresh SQLite file."""
    engine = create_engine(f"sqlite:///{(tmp_path / 'jobs.db').as_posix()}")
    yield JobStore(engine)
    engine.dispose()


def queue(jobs: JobStore, contract_id: str = "c-1") -> None:
    """Queue a job for contract_id under model m:1 and weights w1."""
    jobs.queue(contract_id, "m:1", "w1", source_name="c.pdf", source_type="pdf")


# ----- Transitions -----
def test_queue_claim_finish_moves_through_statuses(jobs):
    """queued -> running -> complete, with the error cleared and times in UTC."""
    queue(jobs)
    assert jobs.get("c-1", "m:1").status == "queued"
    assert jobs.claim("c-1", "m:1")
    assert jobs.get("c-1", "m:1").status == "running"
    done = jobs.finish("c-1", "m:1", "complete", "ignored")
    assert (done.status, done.error, done.source_name) == ("complete", None, "c.pdf")
    assert jobs.get("c-1", "m:1").updated_at.tzinfo is not None


def test_claim_only_takes_queued_jobs(jobs):
    """A running or finished job can't be claimed again, and a missing job can't be claimed."""
    queue(jobs)
    assert jobs.claim("c-1", "m:1")
    assert not jobs.claim("c-1", "m:1")
    assert not jobs.claim("missing", "m:1")


def test_requeue_resets_a_failed_job(jobs):
    """Queueing a failed job again clears its error."""
    queue(jobs)
    jobs.claim("c-1", "m:1")
    jobs.finish("c-1", "m:1", "failed", "boom")
    queue(jobs)
    assert (jobs.get("c-1", "m:1").status, jobs.get("c-1", "m:1").error) == ("queued", None)


def test_finish_missing_job_raises(jobs):
    """Finishing a job that doesn't exist is a KeyError."""
    with pytest.raises(KeyError):
        jobs.finish("missing", "m:1", "complete")


def test_list_filters_by_model_version(jobs):
    """Jobs are listed per model version."""
    queue(jobs, "c-1")
    jobs.queue("c-2", "m:2", "w1", source_name=None, source_type=None)
    assert [j.contract_id for j in jobs.list("m:1")] == ["c-1"]
    assert {j.contract_id for j in jobs.list()} == {"c-1", "c-2"}


# ----- Restart recovery -----
def test_recover_settles_unfinished_jobs(jobs):
    """Queued or running jobs become complete when stored and failed otherwise; finished jobs are untouched."""
    for contract_id in ("queued-lost", "running-saved", "failed-before"):
        queue(jobs, contract_id)
    jobs.claim("running-saved", "m:1")
    jobs.claim("failed-before", "m:1")
    jobs.finish("failed-before", "m:1", "failed", "bad file")
    assert jobs.recover(lambda contract_id, model_version: contract_id == "running-saved") == 2
    assert (jobs.get("queued-lost", "m:1").status, jobs.get("queued-lost", "m:1").error) == ("failed", INTERRUPTED)
    assert jobs.get("running-saved", "m:1").status == "complete"
    assert jobs.get("failed-before", "m:1").error == "bad file"


# ----- Schema -----
def test_outdated_job_table_is_refused(tmp_path):
    """An api_jobs table with other columns raises with a hint to recreate the database."""
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE api_jobs (contract_id TEXT, status TEXT)")
    connection.close()
    engine = create_engine(f"sqlite:///{path.as_posix()}")
    with pytest.raises(RuntimeError, match="recreate"):
        JobStore(engine)
    engine.dispose()
