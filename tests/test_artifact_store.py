"""Tests for src/store/artifact_store.py: idempotent writes, immutability, foreign keys and reads."""
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

import src.inference as inference
from src.chunking.chunker import chunk_contract
from src.inference import ContractInferencePipeline
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.registry.bundle import ClauseModelBundle
from src.risk.schema import WeightSet, load_weight_set
from src.store import tables
from src.store.artifact_store import ArtifactStore, chunk_id_for, evidence_id_for

WEIGHTS_V1_PATH = Path(__file__).resolve().parents[1] / "src" / "risk" / "weights" / "default_weights_v1.json"
MODEL_VERSION = "keyword-model:1"
THRESHOLDS = {category: 0.5 for category in CATEGORIES}

# Three sentences: one grounded by cues, one grounded but repeated across overlapping chunks, one with no cues
TEXT_1 = ("This Agreement is governed by Delaware law. Either party may terminate this Agreement for "
          "convenience on 30 days notice. The supplier shall maintain adequate coverage.")
TEXT_2 = "The supplier shall maintain adequate coverage. This Agreement is governed by Delaware law."
CONTRACTS = {"c-1": TEXT_1, "c-2": TEXT_2, "c-empty": ""}
CUES = {"governinglaw": ["governed", "delaware"], "terminationforconvenience": ["terminate", "convenience"]}
KEYWORDS = {"govern": ("Governing Law", 0.9), "terminate": ("Termination For Convenience", 0.8),
            "coverage": ("Insurance", 0.7)}


class KeywordPredictor:
    """Fake classifier: a fixed float32 probability for each category whose keyword is in the chunk."""

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return a (chunks, 41) float32 matrix with keyword hits set."""
        scores = np.zeros((len(texts), len(CATEGORIES)), dtype=np.float32)
        for row, text in enumerate(texts):
            for keyword, (category, value) in KEYWORDS.items():
                if keyword in text:
                    scores[row, CATEGORIES.index(category)] = value
        return scores


def weights_a() -> WeightSet:
    """The v1 placeholder weights, loaded by explicit path (the default may change)."""
    return load_weight_set(WEIGHTS_V1_PATH)


def weights_b() -> WeightSet:
    """A second weight version that changes severities, thresholds and the triage mix."""
    return weights_a().model_copy(update={
        "weight_version_id": "test-b", "created_by": "tests", "triage_max_weight": 1.0,
        "medium_threshold": 0.1, "high_threshold": 0.25,
        "category_severity": {"Insurance": 1.0, "Governing Law": 0.5},
    })


@pytest.fixture
def make_pipeline(monkeypatch):
    """Factory for a pipeline with the keyword predictor and an 80/40-character chunker."""
    monkeypatch.setattr(inference, "chunk_sentence_windows",
                        lambda contract_id, text, *args: chunk_contract(contract_id, text, 80, 40) if text else [])
    bundle = ClauseModelBundle(predictor=KeywordPredictor(), categories=CATEGORIES, thresholds=THRESHOLDS,
                               model_name="keyword-model", config={}, seed=7)

    def make(weight_set: WeightSet) -> ContractInferencePipeline:
        """Build a pipeline that scores with weight_set."""
        return ContractInferencePipeline(bundle, tokenizer=object(), category_cues=CUES, weight_set=weight_set)
    return make


@pytest.fixture
def store(tmp_path):
    """A fresh store in a temporary SQLite file."""
    artifact_store = ArtifactStore(tmp_path / "artifacts.db")
    yield artifact_store
    artifact_store.close()


def save_all(store: ArtifactStore, make_pipeline, weight_set: WeightSet) -> dict[str, dict]:
    """Run the pipeline on every fixture contract under weight_set, save each result, and return them."""
    results = {}
    for contract_id, text in CONTRACTS.items():
        results[contract_id] = make_pipeline(weight_set).predict_contract(contract_id, text)
        store.save_inference_result(results[contract_id], text=text, model_version=MODEL_VERSION,
                                    thresholds=THRESHOLDS, weight_set=weight_set)
    return results


def row_counts(store: ArtifactStore) -> dict[str, int]:
    """Number of rows in every table."""
    with store.Session() as session:
        return {name: session.scalar(select(func.count()).select_from(table))
                for name, table in tables.Base.metadata.tables.items()}


def by_key(scores: list) -> list:
    """Sort clause scores into (contract, chunk, category) order for comparison."""
    return sorted(scores, key=lambda s: (s.contract_id, s.chunk_index, CATEGORY_TO_INDEX[s.category]))


# ----- Opening a database -----
def test_store_reopens_existing_database_with_its_rows(tmp_path):
    """Closing and reopening the same file keeps what was written."""
    first = ArtifactStore(tmp_path / "artifacts.db")
    first.put_contract("c-1", "abc")
    first.close()
    reopened = ArtifactStore(tmp_path / "artifacts.db")
    assert [c.contract_id for c in reopened.list_contracts()] == ["c-1"]
    reopened.close()


def test_store_refuses_database_with_outdated_schema(tmp_path):
    """A table left over from an older schema raises with a hint to recreate the database."""
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE clause_scores (chunk_id TEXT, category TEXT, model_version TEXT, "
                           "weight_version_id TEXT, score REAL, risk_level TEXT)")
    connection.close()
    with pytest.raises(RuntimeError, match="recreate"):
        ArtifactStore(path)


# ----- Contracts and source metadata -----
def test_put_contract_same_text_is_idempotent_and_different_text_fails(store):
    """Rewriting a contract is a no-op; new text under the same id raises."""
    store.put_contract("c-1", "abc")
    store.put_contract("c-1", "abc")
    with pytest.raises(ValueError, match="different text"):
        store.put_contract("c-1", "abd")
    assert [c.contract_id for c in store.list_contracts()] == ["c-1"]


def test_put_contract_source_metadata_roundtrips(store):
    """Upload metadata is stored and read back; page_count comes from page_starts."""
    store.put_contract("up-1", "page one\n\npage two", source_name="lease.pdf", source_type="pdf",
                       page_starts=[0, 10], ocr_used=True)
    record = store.get_contract("up-1")
    assert (record.source_name, record.source_type, record.page_starts, record.ocr_used) == \
        ("lease.pdf", "pdf", (0, 10), True)
    assert record.text == "page one\n\npage two"
    (summary,) = store.list_contracts()
    assert (summary.page_count, summary.text_length) == (2, 18)


def test_put_contract_metadata_is_fixed_at_first_write(store):
    """A later call without metadata is fine; one with different metadata raises."""
    store.put_contract("up-1", "text", source_name="a.docx", source_type="docx")
    store.put_contract("up-1", "text")
    with pytest.raises(ValueError, match="source metadata"):
        store.put_contract("up-1", "text", source_name="b.docx", source_type="docx")


@pytest.mark.parametrize("page_starts", [[5, 2], [-1], [99]])
def test_put_contract_invalid_page_starts_raise(store, page_starts):
    """Page starts must be non-decreasing offsets inside the text."""
    with pytest.raises(ValueError, match="page_starts"):
        store.put_contract("up-1", "ten chars!", page_starts=page_starts)


def test_get_contract_unknown_id_raises(store):
    """Reading a missing contract raises KeyError."""
    with pytest.raises(KeyError):
        store.get_contract("nope")


# ----- Weight versions -----
def test_weight_version_roundtrips_to_identical_weight_set(store):
    """A stored WeightSet loads back equal, with the same content hash."""
    original = weights_b()
    store.put_weight_version(original)
    loaded = store.get_weight_version("test-b")
    assert loaded == original
    assert loaded.content_sha256 == original.content_sha256
    (record,) = store.list_weight_versions()
    assert (record.weight_version_id, record.created_by, record.content_sha256) == \
        ("test-b", "tests", original.content_sha256)


def test_weight_version_same_id_different_content_fails(store):
    """Weight versions are immutable: reusing an id for new values raises and keeps the original."""
    store.put_weight_version(weights_a())
    store.put_weight_version(weights_a())
    changed = WeightSet.model_validate({**weights_a().model_dump(), "triage_top_k": 5})
    with pytest.raises(ValueError, match="different content"):
        store.put_weight_version(changed)
    assert store.get_weight_version(weights_a().weight_version_id).triage_top_k == weights_a().triage_top_k


def test_get_weight_version_unknown_id_raises(store):
    """Loading a missing weight version raises KeyError."""
    with pytest.raises(KeyError):
        store.get_weight_version("missing")


# ----- Model versions -----
def test_model_version_roundtrips_and_is_immutable(store):
    """Thresholds read back exactly; the same version with other thresholds raises."""
    thresholds = {**THRESHOLDS, "Insurance": 0.25}
    store.put_model_version("m:1", model_name="m", thresholds=thresholds, seed=3)
    store.put_model_version("m:1", model_name="m", thresholds=thresholds, seed=3)
    record = store.get_model_version("m:1")
    assert (record.model_name, record.seed, record.thresholds) == ("m", 3, thresholds)
    with pytest.raises(ValueError, match="thresholds_json"):
        store.put_model_version("m:1", model_name="m", thresholds=THRESHOLDS, seed=3)


@pytest.mark.parametrize("thresholds", [
    {category: 0.5 for category in CATEGORIES[:-1]},
    {**THRESHOLDS, "Insurance": 1.5},
])
def test_model_version_invalid_thresholds_raise(store, thresholds):
    """Thresholds need every CUAD category and values in [0, 1]."""
    with pytest.raises(ValueError):
        store.put_model_version("m:1", model_name="m", thresholds=thresholds)


# ----- Saving a whole pipeline result -----
def test_save_inference_result_writes_every_artifact(store, make_pipeline):
    """c-1 stores 4 chunks, 4 x 41 probabilities, 2 evidence rows, 1 Unknown and 3 clause scores."""
    result = make_pipeline(weights_a()).predict_contract("c-1", TEXT_1)
    stored_score = store.save_inference_result(result, text=TEXT_1, model_version=MODEL_VERSION,
                                               thresholds=THRESHOLDS, weight_set=weights_a())
    counts = row_counts(store)
    assert {name: counts[name] for name in ("contracts", "chunks", "chunk_probs", "evidence",
                                            "unknown_predictions", "clause_scores", "contract_scores",
                                            "weight_versions", "model_versions")} == {
        "contracts": 1, "chunks": 4, "chunk_probs": 4 * 41, "evidence": 2, "unknown_predictions": 1,
        "clause_scores": 3, "contract_scores": 1, "weight_versions": 1, "model_versions": 1,
    }
    assert store.get_clause_scores("c-1", MODEL_VERSION, "v1-placeholder") == by_key(result["clause_scores"])
    assert stored_score == result["contract_score"]
    assert store.get_probabilities("c-1:3", MODEL_VERSION)["Insurance"] == pytest.approx(0.7)


def test_save_inference_result_twice_is_a_noop(store, make_pipeline):
    """Saving the same results again adds no rows and raises nothing."""
    save_all(store, make_pipeline, weights_a())
    before = row_counts(store)
    save_all(store, make_pipeline, weights_a())
    assert row_counts(store) == before


def test_save_inference_result_conflict_writes_nothing(store, make_pipeline):
    """Scores made under one weight set but saved under another fail, leaving no partial contract."""
    result = make_pipeline(weights_a()).predict_contract("c-2", TEXT_2)
    with pytest.raises(ValueError, match="weight version"):
        store.save_inference_result(result, text=TEXT_2, model_version=MODEL_VERSION,
                                    thresholds=THRESHOLDS, weight_set=weights_b())
    assert all(count == 0 for count in row_counts(store).values())


def test_save_inference_result_changed_probability_fails(store, make_pipeline):
    """Resaving a contract with a different probability raises and keeps the stored value."""
    result = make_pipeline(weights_a()).predict_contract("c-1", TEXT_1)
    store.save_inference_result(result, text=TEXT_1, model_version=MODEL_VERSION,
                                thresholds=THRESHOLDS, weight_set=weights_a())
    result["chunks"][0]["probabilities"]["Parties"] = 0.25
    with pytest.raises(ValueError, match="different value"):
        store.save_inference_result(result, text=TEXT_1, model_version=MODEL_VERSION,
                                    thresholds=THRESHOLDS, weight_set=weights_a())
    assert store.get_probabilities("c-1:0", MODEL_VERSION)["Parties"] == 0.0


@pytest.mark.parametrize("text", [TEXT_1.upper(), TEXT_1[:-1] + "!"])
def test_save_inference_result_wrong_text_fails(store, make_pipeline, text):
    """Chunks must sit at their offsets in the text passed in, and nothing is written otherwise."""
    result = make_pipeline(weights_a()).predict_contract("c-1", TEXT_1)
    with pytest.raises(ValueError, match="chunk text does not match"):
        store.save_inference_result(result, text=text, model_version=MODEL_VERSION,
                                    thresholds=THRESHOLDS, weight_set=weights_a())


def test_save_inference_result_detection_below_threshold_fails(store, make_pipeline):
    """Thresholds that contradict the recorded detections are rejected."""
    result = make_pipeline(weights_a()).predict_contract("c-1", TEXT_1)
    with pytest.raises(ValueError, match="below its category threshold"):
        store.save_inference_result(result, text=TEXT_1, model_version=MODEL_VERSION,
                                    thresholds={**THRESHOLDS, "Insurance": 0.95}, weight_set=weights_a())


def test_save_inference_result_empty_contract_scores_zero(store, make_pipeline):
    """A contract with no chunks still gets a stored contract score of 0."""
    result = make_pipeline(weights_a()).predict_contract("c-empty", "")
    score = store.save_inference_result(result, text="", model_version=MODEL_VERSION,
                                        thresholds=THRESHOLDS, weight_set=weights_a())
    assert (score.triage_score, score.rank, score.top_clauses) == (0.0, 1, [])
    assert store.list_run_contract_ids(MODEL_VERSION) == ["c-empty"]


def test_save_inference_result_records_upload_metadata(store, make_pipeline):
    """Upload metadata passed with a result lands on the contract row."""
    result = make_pipeline(weights_a()).predict_contract("c-2", TEXT_2)
    store.save_inference_result(result, text=TEXT_2, model_version=MODEL_VERSION, thresholds=THRESHOLDS,
                                weight_set=weights_a(), source_name="msa.pdf", source_type="pdf",
                                page_starts=[0, 47], ocr_used=False)
    record = store.get_contract("c-2")
    assert (record.source_type, record.page_starts, record.ocr_used) == ("pdf", (0, 47), False)


# ----- Unknown queue -----
def test_list_unknowns_returns_ungrounded_predictions(store, make_pipeline):
    """c-1's only Unknown is Insurance in chunk 3 ([120:167]), probability 0.7 over threshold 0.5."""
    save_all(store, make_pipeline, weights_a())
    (item,) = store.list_unknowns(MODEL_VERSION, "c-1")
    assert (item.chunk_id, item.chunk_index, item.chunk_start, item.chunk_end, item.category, item.reason) == \
        ("c-1:3", 3, 120, 167, "Insurance", "no_cue_match")
    assert item.probability == pytest.approx(0.7)
    assert item.threshold == 0.5


def test_list_unknowns_skips_dropped_duplicates(store, make_pipeline):
    """Chunk 0's partial Termination sentence is neither evidence nor Unknown, so it is not queued."""
    save_all(store, make_pipeline, weights_a())
    evidence = store.get_evidence("c-1", MODEL_VERSION)
    assert [(e.chunk_index, e.category, e.start, e.end) for e in evidence] == [
        (0, "Governing Law", 0, 43), (1, "Termination For Convenience", 44, 120)]
    assert evidence[0].evidence_id == evidence_id_for("c-1:0", "Governing Law", MODEL_VERSION)
    assert all(item.category != "Termination For Convenience" for item in store.list_unknowns(MODEL_VERSION))


def test_list_unknowns_across_contracts_is_ordered(store, make_pipeline):
    """
    Without a contract filter every contract's Unknowns are listed, by contract then chunk.
    In TEXT_2 "coverage" spans [37:45] and chunk 1 starts at 40, so only chunk 0 detects Insurance.
    """
    save_all(store, make_pipeline, weights_a())
    assert [(i.contract_id, i.chunk_index, i.category) for i in store.list_unknowns(MODEL_VERSION)] == [
        ("c-1", 3, "Insurance"), ("c-2", 0, "Insurance")]
    assert store.list_unknowns("other-model:1") == []


# ----- Foreign keys -----
def test_foreign_keys_are_enforced_by_sqlite(store):
    """A raw score row pointing at a missing weight version and probability is rejected by SQLite."""
    with pytest.raises(IntegrityError):
        with store.Session.begin() as session:
            session.add(tables.ContractScoreRow(
                contract_id="missing", model_version="m", weight_version_id="missing", triage_score=0.0,
                high_count=0, medium_count=0, low_count=0, unknown_count=0, top_clauses_json="[]",
                weight_set_sha256="0" * 64))


def test_evidence_without_probability_fails(store):
    """Evidence needs a stored probability for its (chunk, category, model version)."""
    store.put_contract("c-1", "abcdefgh")
    store.put_chunk(chunk_id="c-1:0", contract_id="c-1", chunk_index=0, start=0, end=8, text="abcdefgh")
    with pytest.raises(KeyError, match="no stored probability"):
        store.add_evidence(evidence_id="ev", chunk_id="c-1:0", category="Parties", quote="cd",
                           start=2, end=4, model_version="m")


def test_save_scores_for_undetected_clause_fails(store, make_pipeline):
    """A clause score must belong to a stored detection."""
    result = make_pipeline(weights_a()).predict_contract("c-1", TEXT_1)
    store.save_inference_result(result, text=TEXT_1, model_version=MODEL_VERSION,
                                thresholds=THRESHOLDS, weight_set=weights_a())
    stray = result["clause_scores"][0].model_copy(update={"category": "Parties"})
    with pytest.raises(ValueError, match="not a stored detection"):
        store.save_scores(MODEL_VERSION, weights_a(), [stray], [])


# ----- Reads -----
def test_get_contract_detail_bundles_everything(store, make_pipeline):
    """The detail view holds the contract, chunks, evidence, Unknowns and both kinds of score."""
    results = save_all(store, make_pipeline, weights_a())
    detail = store.get_contract_detail("c-1", MODEL_VERSION, "v1-placeholder")
    assert detail.contract.text == TEXT_1
    assert [c.chunk_id for c in detail.chunks] == [chunk_id_for("c-1", i) for i in range(4)]
    assert len(detail.evidence) == 2 and len(detail.unknowns) == 1
    assert detail.clause_scores == by_key(results["c-1"]["clause_scores"])
    assert detail.contract_score.triage_score == results["c-1"]["contract_score"].triage_score
    assert store.get_contract_detail("c-1", MODEL_VERSION, "test-b").contract_score is None


def test_list_contract_scores_ranks_ties_together(store, make_pipeline):
    """Two contracts with identical text share rank 1; the empty contract ranks 3 at score 0."""
    save_all(store, make_pipeline, weights_a())
    twin = make_pipeline(weights_a()).predict_contract("c-2-twin", TEXT_2)
    store.save_inference_result(twin, text=TEXT_2, model_version=MODEL_VERSION,
                                thresholds=THRESHOLDS, weight_set=weights_a())
    ranked = store.list_contract_scores(MODEL_VERSION, "v1-placeholder")
    assert [(s.contract_id, s.rank) for s in ranked][-3:] == [("c-2", 2), ("c-2-twin", 2), ("c-empty", 4)]
    assert ranked[0].contract_id == "c-1" and ranked[0].rank == 1


# ----- Feedback -----
def test_add_feedback_roundtrips(store, make_pipeline):
    """Feedback is appended with an id, its contract, and the given timestamp."""
    save_all(store, make_pipeline, weights_a())
    when = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    first = store.add_feedback(chunk_id="c-1:3", category="Insurance", model_version=MODEL_VERSION,
                               verdict="accept", reviewer="ana", note="real insurance clause", timestamp=when)
    store.add_feedback(chunk_id="c-2:0", category="Parties", model_version=MODEL_VERSION,
                       verdict="missing", reviewer="ana")
    assert (first.id, first.contract_id, first.verdict, first.note) == (1, "c-1", "accept", "real insurance clause")
    assert [f.chunk_id for f in store.list_feedback()] == ["c-1:3", "c-2:0"]
    (only,) = store.list_feedback(contract_id="c-1")
    assert only.timestamp.replace(tzinfo=timezone.utc) == when


@pytest.mark.parametrize("changes,error", [
    ({"verdict": "maybe"}, ValueError),
    ({"category": "Not A Category"}, ValueError),
    ({"reviewer": " "}, ValueError),
    ({"chunk_id": "nope:0"}, KeyError),
])
def test_add_feedback_invalid_input_raises(store, make_pipeline, changes, error):
    """Bad verdicts, categories, reviewers and unknown chunks are rejected."""
    save_all(store, make_pipeline, weights_a())
    values = dict(chunk_id="c-1:0", category="Governing Law", model_version=MODEL_VERSION,
                  verdict="reject", reviewer="ana")
    with pytest.raises(error):
        store.add_feedback(**{**values, **changes})


@pytest.mark.parametrize("chunk_id,category,verdict,match", [
    ("c-1:2", "Insurance", "accept", "needs a prediction"),
    ("c-1:2", "Insurance", "reject", "needs a prediction"),
    ("c-1:3", "Insurance", "missing", "predicted"),
    ("c-empty:0", "Insurance", "missing", "unknown chunk"),
])
def test_add_feedback_verdict_must_fit_the_prediction(store, make_pipeline, chunk_id, category, verdict, match):
    """Called directly, the store still refuses accept/reject without a detection and missing on one."""
    save_all(store, make_pipeline, weights_a())
    with pytest.raises((ValueError, KeyError), match=match):
        store.add_feedback(chunk_id=chunk_id, category=category, model_version=MODEL_VERSION, verdict=verdict,
                           reviewer="ana")
    assert store.list_feedback() == []


def test_add_feedback_on_chunk_not_run_under_model_raises(store, make_pipeline):
    """A chunk with no probabilities under the given model version can't take any verdict."""
    save_all(store, make_pipeline, weights_a())
    with pytest.raises(ValueError, match="no stored predictions"):
        store.add_feedback(chunk_id="c-1:3", category="Insurance", model_version="other-model:1",
                           verdict="missing", reviewer="ana")


def test_add_feedback_timestamp_is_utc_and_never_goes_back(store, make_pipeline):
    """12:00+01:00 reads back as 11:00 UTC; ana's next verdict on the key can't be earlier, ben's can."""
    save_all(store, make_pipeline, weights_a())
    plus_one = timezone(timedelta(hours=1))
    first = store.add_feedback(chunk_id="c-1:3", category="Insurance", model_version=MODEL_VERSION,
                               verdict="accept", reviewer="ana", timestamp=datetime(2026, 11, 1, 12, tzinfo=plus_one))
    (listed,) = store.list_feedback()
    assert first.timestamp == listed.timestamp == datetime(2026, 11, 1, 11, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="earlier"):
        store.add_feedback(chunk_id="c-1:3", category="Insurance", model_version=MODEL_VERSION, verdict="reject",
                           reviewer="ana", timestamp=datetime(2026, 11, 1, 10, 59, tzinfo=timezone.utc))
    store.add_feedback(chunk_id="c-1:3", category="Insurance", model_version=MODEL_VERSION, verdict="reject",
                       reviewer="ben", timestamp=datetime(2026, 11, 1, 10, 59, tzinfo=timezone.utc))


def test_get_detection_and_get_chunk(store, make_pipeline):
    """c-1:0 Governing Law is grounded, c-1:3 Insurance unknown, c-1:0's dropped Termination none."""
    save_all(store, make_pipeline, weights_a())
    assert [store.get_detection(chunk, category, MODEL_VERSION) for chunk, category in [
        ("c-1:0", "Governing Law"), ("c-1:3", "Insurance"), ("c-1:0", "Termination For Convenience")]] == \
        ["grounded", "unknown", None]
    chunk = store.get_chunk("c-1:3")
    assert (chunk.contract_id, chunk.chunk_index, chunk.start, chunk.end) == ("c-1", 3, 120, 167)
    with pytest.raises(KeyError):
        store.get_chunk("c-1:9")


def test_list_feedback_filters_by_chunk_and_category(store, make_pipeline):
    """chunk_id and category narrow the log to one key."""
    save_all(store, make_pipeline, weights_a())
    store.add_feedback(chunk_id="c-1:3", category="Insurance", model_version=MODEL_VERSION, verdict="accept",
                       reviewer="ana")
    store.add_feedback(chunk_id="c-1:0", category="Governing Law", model_version=MODEL_VERSION, verdict="accept",
                       reviewer="ana")
    assert [f.category for f in store.list_feedback(chunk_id="c-1:0")] == ["Governing Law"]
    assert [f.chunk_id for f in store.list_feedback(category="Insurance")] == ["c-1:3"]


# ----- Advisor rankings -----
def advisor_row(**changes) -> dict:
    """One valid advisor ranking for c-1's Governing Law sentence."""
    row = dict(contract_id="c-1", category="Governing Law", start=0, end=43,
               quote="This Agreement is governed by Delaware law.", advisor_rank=2, advisor_bucket="Low",
               reviewer="advisor", ranking_set="pilot-2026-11")
    return {**row, **changes}


def test_add_advisor_rankings_is_idempotent(store):
    """The first write stores two rows; repeating it stores none; nullable rank and bucket are allowed."""
    store.put_contract("c-1", TEXT_1)
    rows = [advisor_row(), advisor_row(category="Termination For Convenience", start=44, end=120,
                                       quote=TEXT_1[44:120], advisor_rank=None, advisor_bucket=None)]
    assert store.add_advisor_rankings(rows) == 2
    assert store.add_advisor_rankings(rows) == 0
    stored = store.list_advisor_rankings(ranking_set="pilot-2026-11")
    assert [(r.category, r.advisor_rank, r.advisor_bucket) for r in stored] == [
        ("Governing Law", 2, "Low"), ("Termination For Convenience", None, None)]


@pytest.mark.parametrize("changes,error,match", [
    ({"advisor_rank": 1}, ValueError, "different value"),
    ({"quote": "This Agreement is governed by Delaware law!"}, ValueError, "does not match"),
    ({"advisor_bucket": "Severe"}, ValueError, "advisor_bucket"),
    ({"advisor_rank": 0}, ValueError, "advisor_rank"),
    ({"contract_id": "missing"}, KeyError, "unknown contract"),
])
def test_add_advisor_rankings_invalid_or_conflicting_rows_raise(store, changes, error, match):
    """Conflicts, ungrounded quotes, bad buckets/ranks and unknown contracts all fail atomically."""
    store.put_contract("c-1", TEXT_1)
    store.add_advisor_rankings([advisor_row()])
    with pytest.raises(error, match=match):
        store.add_advisor_rankings([advisor_row(category="Insurance"), advisor_row(**changes)])
    assert len(store.list_advisor_rankings()) == 1
