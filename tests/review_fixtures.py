"""
Shared fixtures for the src/review/ tests: a store holding three small contracts run through the
keyword pipeline of tests/test_artifact_store.py, and a helper that writes timed verdicts.

What the store holds under MODEL_VERSION (thresholds 0.5, chunks of 80 characters every 40):
    c-1  4 chunks; Governing Law grounded in chunk 0 [0:43], Termination For Convenience grounded in
         chunk 1, Insurance Unknown in chunk 3 [120:167] (p 0.7). Chunk 0's Termination (p 0.8) is a
         dropped duplicate: above threshold, but neither evidence nor Unknown.
    c-2  2 chunks; Insurance Unknown in chunk 0 [0:80] (p 0.7), Governing Law grounded in chunk 1 and
         a dropped duplicate in chunk 0 (p 0.9).
    c-3  1 chunk [0:40] on two pages (page_starts [0, 20]); Uncapped Liability Unknown (p 0.95).
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.review.feedback import submit_feedback
from src.store.artifact_store import ArtifactStore
from src.store.records import FeedbackRecord
from tests import test_artifact_store as store_tests
from tests.test_artifact_store import MODEL_VERSION, THRESHOLDS, save_all, weights_a

TEXT_3 = "The vendor shall indemnify the customer."
T0 = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)

# The store tests' fixtures, bound here so review tests can bind them in turn
make_pipeline = store_tests.make_pipeline
store = store_tests.store


@pytest.fixture
def review_store(store, make_pipeline, monkeypatch):
    """The store with c-1, c-2 (and the empty contract) plus c-3, a two-page upload with a 0.95 Unknown."""
    monkeypatch.setitem(store_tests.KEYWORDS, "indemnify", ("Uncapped Liability", 0.95))
    save_all(store, make_pipeline, weights_a())
    result = make_pipeline(weights_a()).predict_contract("c-3", TEXT_3)
    store.save_inference_result(result, text=TEXT_3, model_version=MODEL_VERSION, thresholds=THRESHOLDS,
                                weight_set=weights_a(), source_name="msa.pdf", source_type="pdf",
                                page_starts=[0, 20])
    return store


def verdict(store: ArtifactStore, chunk_id: str, category: str, verdict: str, reviewer: str,
            minute: int = 0, model_version: str = MODEL_VERSION, note: str | None = None) -> FeedbackRecord:
    """Submit one verdict at T0 + minute through the review module."""
    return submit_feedback(store, dict(chunk_id=chunk_id, category=category, model_version=model_version,
                                       verdict=verdict, reviewer=reviewer, note=note,
                                       timestamp=T0 + timedelta(minutes=minute)))
