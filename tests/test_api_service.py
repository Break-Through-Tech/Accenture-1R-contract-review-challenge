"""Tests for clausetriage/api/service.py: scoring jobs, reads, registers, weight versions and preview over a real store."""
import pytest

from clausetriage.api.jobs import INTERRUPTED
from clausetriage.api.service import (INTERNAL_ERROR, ClauseTriageService, ConflictError, InvalidInputError, NotFoundError,
                             TooLargeError, UnsupportedTypeError, safe_filename, upload_contract_id)
from clausetriage.ingest.documents import Document
from clausetriage.register.export import build_register
from clausetriage.risk.contract import aggregate_contracts
from clausetriage.store.artifact_store import ArtifactStore
from clausetriage.store.rescoring import rescore_contracts
from tests.api_fixtures import (MODEL_VERSION, TEXT_1, TEXT_2, FailingPredictor, make_pipeline, make_service,
                                weights_a, weights_b)
from tests.pdf_fixtures import pdf_bytes
from tests.test_artifact_store import row_counts


@pytest.fixture
def service(tmp_path, monkeypatch):
    """A service over a fresh store, scoring with the keyword pipeline under the v1 weights."""
    built = make_service(tmp_path / "api.db", monkeypatch)
    yield built
    built.store.close()


def score_text(service: ClauseTriageService, text: str, name: str | None = None) -> str:
    """Submit pasted text, run its job, and return the contract id."""
    job, queued = service.submit_text(text, name)
    assert queued
    assert service.run_job(job.contract_id).status == "complete"
    return job.contract_id


def fresh_result(monkeypatch, contract_id: str, text: str, weight_set=None) -> dict:
    """What the pipeline returns for text under weight_set, without any store."""
    return make_pipeline(monkeypatch, weight_set).predict_contract(contract_id, text)


# ----- Names and ids from untrusted input -----
@pytest.mark.parametrize(("filename", "expected"), [
    ("../../etc/passwd.pdf", "passwd.pdf"),
    ("C:\\Users\\me\\Lease.PDF", "Lease.PDF"),
    ("bad\x00name\u202e.txt", "badname.txt"),
    ("  two   spaces .docx", "two spaces.docx"),
    ("a" * 200 + ".pdf", "a" * 120 + ".pdf"),
    (None, ""),
])
def test_safe_filename_strips_directories_and_invisible_characters(filename, expected):
    """Directory parts, control and format characters go; whitespace collapses; the stem is capped."""
    assert safe_filename(filename) == expected


@pytest.mark.parametrize(("name", "expected"), [
    ("Lease Agreement (final)", "Lease-Agreement-final-0123456789ab"),
    ("Café Société", "Cafe-Societe-0123456789ab"),
    ("***", "contract-0123456789ab"),
    ("a" * 70, "a" * 60 + "-0123456789ab"),
])
def test_upload_contract_id_is_an_ascii_slug_plus_text_hash(name, expected):
    """Hand-computed ids: letters and digits kept, accents folded, other runs become '-', hash suffix."""
    assert upload_contract_id(name, "0123456789abcdef" * 4) == expected


# ----- Submitting and scoring -----
def test_submit_text_then_run_job_stores_the_pipeline_result(service, monkeypatch):
    """A pasted contract is queued, then scored into the store exactly as a fresh pipeline run scores it."""
    job, queued = service.submit_text(TEXT_1, "Alpha")
    assert queued and job.status == "queued" and job.contract_id.startswith("Alpha-")
    assert service.run_job(job.contract_id).status == "complete"
    expected = fresh_result(monkeypatch, job.contract_id, TEXT_1)
    assert service.store.get_clause_scores(job.contract_id, MODEL_VERSION, "v1-placeholder") == \
        expected["clause_scores"]
    record = service.store.get_contract(job.contract_id)
    assert (record.text, record.source_name, record.source_type) == (TEXT_1, "Alpha", "pasted")


def test_submit_same_text_twice_does_not_rescore(service):
    """Resubmitting a scored contract returns its complete job without queueing anything."""
    contract_id = score_text(service, TEXT_1)
    before = row_counts(service.store)
    job, queued = service.submit_text(TEXT_1)
    assert (job.contract_id, job.status, queued) == (contract_id, "complete", False)
    assert row_counts(service.store) == before


def test_submit_while_queued_returns_the_same_job(service):
    """A second submission of a queued contract is not queued again."""
    first, _ = service.submit_text(TEXT_1)
    second, queued = service.submit_text(TEXT_1)
    assert (second.contract_id, second.status, queued) == (first.contract_id, "queued", False)


def test_submit_upload_pdf_stores_pages_and_source_metadata(service):
    """An uploaded PDF keeps its file name, type and page starts in the store."""
    job, _ = service.submit_upload(pdf_bytes([TEXT_1, TEXT_2]), "folder/Lease.pdf")
    service.run_job(job.contract_id)
    record = service.store.get_contract(job.contract_id)
    assert (record.source_name, record.source_type, record.ocr_used) == ("Lease.pdf", "pdf", False)
    assert record.page_starts[0] == 0 and len(record.page_starts) == 2


@pytest.mark.parametrize(("data", "filename", "error", "message"), [
    (b"MZ\x90\x00", "tool.exe", UnsupportedTypeError, "type '.exe'"),
    (b"text", "no_extension", UnsupportedTypeError, "without an extension"),
    (b"plain words", "fake.pdf", InvalidInputError, "isn't a PDF"),
    (b"%PDF-1.4 truncated", "broken.pdf", InvalidInputError, "damaged"),
    (b"", "empty.txt", InvalidInputError, "empty"),
])
def test_submit_upload_bad_files_raise_reviewer_messages(service, data, filename, error, message):
    """Unsupported, mislabelled, damaged and empty files are refused with a reason and nothing is queued."""
    with pytest.raises(error, match=message):
        service.submit_upload(data, filename)
    assert service.list_jobs() == []


def test_submit_upload_over_the_size_limit_raises(service, monkeypatch):
    """A file over MAX_FILE_BYTES is refused before it is parsed."""
    monkeypatch.setattr("clausetriage.api.service.MAX_FILE_BYTES", 10)
    with pytest.raises(TooLargeError):
        service.submit_upload(b"x" * 11, "big.txt")


@pytest.mark.parametrize(("text", "message"), [("   ", "empty"), ("bad \ud800 surrogate", "valid Unicode")])
def test_submit_text_invalid_text_raises(service, text, message):
    """Blank pasted text and text with an unpaired surrogate can't be stored."""
    with pytest.raises(InvalidInputError, match=message):
        service.submit_text(text)


def test_submit_document_existing_id_with_other_text_conflicts(service):
    """A dataset contract id already stored with different text is refused, not overwritten."""
    service.submit_document(Document(name="c", text=TEXT_1), contract_id="CUAD-1", source_type="cuad")
    service.run_job("CUAD-1")
    with pytest.raises(ConflictError, match="different text"):
        service.submit_document(Document(name="c", text=TEXT_2), contract_id="CUAD-1", source_type="cuad")


# ----- Failures never count as complete -----
def test_run_job_model_failure_is_recorded_as_failed(tmp_path, monkeypatch):
    """A crashing model leaves the job failed with a generic reason and nothing in the store."""
    service = make_service(tmp_path / "api.db", monkeypatch, predictor=FailingPredictor())
    job, _ = service.submit_text(TEXT_1)
    failed = service.run_job(job.contract_id)
    assert (failed.status, failed.error) == ("failed", INTERNAL_ERROR)
    assert service.store.list_contracts() == []
    assert service.list_ranked().contracts == []
    with pytest.raises(NotFoundError):
        service.get_contract_detail(job.contract_id)


def test_run_job_save_failure_is_recorded_as_failed(service, monkeypatch):
    """A store write that fails after scoring leaves the job failed, never complete."""
    def broken_save(*args, **kwargs):
        """Fail like a full disk would."""
        raise OSError("disk full")
    monkeypatch.setattr(service.store, "save_inference_result", broken_save)
    job, _ = service.submit_text(TEXT_1)
    assert service.run_job(job.contract_id).status == "failed"
    assert service.get_job(job.contract_id).error == INTERNAL_ERROR


def test_failed_job_can_be_resubmitted(tmp_path, monkeypatch):
    """After a failure, submitting the same text queues it again and it can complete."""
    database = tmp_path / "api.db"
    broken = make_service(database, monkeypatch, predictor=FailingPredictor())
    job, _ = broken.submit_text(TEXT_1)
    broken.run_job(job.contract_id)
    broken.store.close()
    working = make_service(database, monkeypatch)
    again, queued = working.submit_text(TEXT_1)
    assert (again.status, queued) == ("queued", True)
    assert working.run_job(job.contract_id).status == "complete"


def test_restart_marks_unfinished_jobs_failed_or_complete(tmp_path, monkeypatch):
    """A new service settles a queued job without a result as failed and a running one with a result as complete."""
    database = tmp_path / "api.db"
    first = make_service(database, monkeypatch)
    lost, _ = first.submit_text(TEXT_1)
    saved, _ = first.submit_text(TEXT_2)
    first.run_job(saved.contract_id)
    first.jobs.queue(saved.contract_id, MODEL_VERSION, "v1-placeholder", source_name=None, source_type="pasted")
    first.jobs.claim(saved.contract_id, MODEL_VERSION)
    first.store.close()

    second = make_service(database, monkeypatch)
    assert (second.get_job(lost.contract_id).status, second.get_job(lost.contract_id).error) == \
        ("failed", INTERRUPTED)
    assert second.get_job(saved.contract_id).status == "complete"
    assert second.run_job(lost.contract_id).status == "failed"


def test_get_job_unknown_contract_raises(service):
    """Status of a contract never submitted or stored is NotFoundError."""
    with pytest.raises(NotFoundError):
        service.get_job("nothing-here")


def test_get_job_for_contract_stored_without_a_job_is_complete(service, monkeypatch):
    """A contract another tool saved straight into the store reports complete."""
    result = fresh_result(monkeypatch, "direct", TEXT_1)
    service.store.save_inference_result(result, text=TEXT_1, model_version=MODEL_VERSION,
                                        thresholds=service.pipeline.bundle.thresholds, weight_set=weights_a())
    assert service.get_job("direct").status == "complete"


# ----- Reads -----
def test_list_ranked_matches_ranking_the_fresh_results(service, monkeypatch):
    """The ranking equals aggregate_contracts over fresh pipeline results for the same contracts."""
    ids = [score_text(service, TEXT_1, "alpha"), score_text(service, TEXT_2, "beta")]
    fresh = [fresh_result(monkeypatch, cid, text) for cid, text in zip(ids, [TEXT_1, TEXT_2])]
    expected = aggregate_contracts([s for r in fresh for s in r["clause_scores"]], weights_a(), ids)
    ranking = service.list_ranked()
    assert [(c.contract_id, c.rank, c.triage_score) for c in ranking.contracts] == \
        [(s.contract_id, s.rank, s.triage_score) for s in expected]
    assert (ranking.ranked_out_of, ranking.saved, ranking.weight_version_id) == (2, True, "v1-placeholder")


def test_list_ranked_empty_store_is_an_empty_ranking(service):
    """Before anything is scored the server's own model version ranks nothing (not a 404)."""
    assert service.list_ranked().contracts == []


@pytest.mark.parametrize(("model_version", "weight_version_id"), [("other-model:9", None), (None, "no-such-weights")])
def test_list_ranked_unknown_versions_raise(service, model_version, weight_version_id):
    """An unknown model version or weight version is NotFoundError."""
    with pytest.raises(NotFoundError):
        service.list_ranked(model_version, weight_version_id)


def test_get_contract_detail_has_probabilities_detections_and_rank(service, monkeypatch):
    """The detail carries every chunk's 41 probabilities, one prediction per clause score, and the rank."""
    contract_id = score_text(service, TEXT_1)
    score_text(service, TEXT_2)
    detail = service.get_contract_detail(contract_id)
    expected = fresh_result(monkeypatch, contract_id, TEXT_1)
    assert [chunk.probabilities for chunk in detail.chunks] == \
        [pytest.approx(chunk["probabilities"]) for chunk in expected["chunks"]]
    assert [(p.chunk_index, p.category, p.has_evidence) for p in detail.predictions] == \
        [(s.chunk_index, s.category, s.has_evidence) for s in detail.clause_scores]
    assert all(p.probability >= p.threshold for p in detail.predictions)
    assert (detail.rank, detail.ranked_out_of, detail.contract.text) == (1, 2, TEXT_1)
    assert len(detail.unknowns) == len([p for p in detail.predictions if not p.has_evidence])


def test_get_contract_detail_unknown_contract_raises(service):
    """Detail of an unknown contract is NotFoundError."""
    with pytest.raises(NotFoundError):
        service.get_contract_detail("missing")


# ----- Registers -----
def test_get_register_equals_build_register_on_the_pipeline_result(service, monkeypatch):
    """The register from stored data is identical to build_register on the fresh result with the same rank."""
    contract_id = score_text(service, TEXT_1)
    other = score_text(service, TEXT_2)
    fresh = {cid: fresh_result(monkeypatch, cid, text) for cid, text in [(contract_id, TEXT_1), (other, TEXT_2)]}
    ranked = aggregate_contracts([s for r in fresh.values() for s in r["clause_scores"]], weights_a(), list(fresh))
    score = next(s for s in ranked if s.contract_id == contract_id)
    expected = build_register(fresh[contract_id], score, text=TEXT_1, ranked_out_of=2)
    assert service.get_register(contract_id) == expected


def test_get_register_pdf_has_page_numbers(service):
    """Quoted rows of an uploaded PDF carry the page their quote starts on."""
    job, _ = service.submit_upload(pdf_bytes(["Nothing here.", TEXT_1]), "two-pages.pdf")
    service.run_job(job.contract_id)
    quoted = [row for row in service.get_register(job.contract_id).rows if row.quote]
    assert quoted and {row.page for row in quoted} == {2}


# ----- Weight versions -----
def test_list_weights_has_the_default_version(service):
    """The pipeline's weight set is stored at startup and named as the default."""
    weights = service.list_weights()
    assert weights.default_weight_version_id == "v1-placeholder"
    assert [v.weight_version_id for v in weights.versions] == ["v1-placeholder"]
    assert service.get_weights("v1-placeholder").is_default


def test_create_weights_stores_version_and_rescores_stored_contracts(service, monkeypatch):
    """A new version is stored with created_by and every stored contract gets scores equal to a fresh run."""
    contract_id = score_text(service, TEXT_1)
    created = service.create_weights(weights_b())
    assert (created.created, created.rescored_contracts, created.version.created_by) == (True, 1, "tests")
    expected = fresh_result(monkeypatch, contract_id, TEXT_1, weights_b())
    assert service.store.get_clause_scores(contract_id, MODEL_VERSION, "test-b") == expected["clause_scores"]
    detail = service.get_weights("test-b")
    assert detail.weight_set == weights_b() and not detail.is_default


def test_create_weights_identical_repost_changes_nothing(service):
    """Posting the same version again reports created False and writes no rows."""
    score_text(service, TEXT_1)
    service.create_weights(weights_b())
    before = row_counts(service.store)
    again = service.create_weights(weights_b())
    assert (again.created, again.rescored_contracts) == (False, 0)
    assert row_counts(service.store) == before


def test_create_weights_same_id_other_values_conflicts_and_keeps_original(service):
    """An existing id with different values is refused and the stored version and scores stay as they were."""
    contract_id = score_text(service, TEXT_1)
    service.create_weights(weights_b())
    before = (row_counts(service.store), service.store.get_contract_score(contract_id, MODEL_VERSION, "test-b"))
    edited = type(weights_b()).model_validate({**weights_b().model_dump(), "triage_top_k": 5})
    with pytest.raises(ConflictError, match="new weight_version_id"):
        service.create_weights(edited)
    assert service.get_weights("test-b").weight_set == weights_b()
    assert (row_counts(service.store), service.store.get_contract_score(contract_id, MODEL_VERSION, "test-b")) == \
        before


def test_create_weights_rejects_unsafe_version_id(service):
    """A weight_version_id with a slash or spaces is refused before anything is stored."""
    unsafe = type(weights_b()).model_validate({**weights_b().model_dump(), "weight_version_id": "../v 3"})
    with pytest.raises(InvalidInputError, match="weight_version_id"):
        service.create_weights(unsafe)
    assert [v.weight_version_id for v in service.list_weights().versions] == ["v1-placeholder"]


def test_contract_scored_after_a_weight_post_is_scored_under_it_on_read(service, monkeypatch):
    """A contract uploaded after a version was posted is filled in, from cached probabilities, when read."""
    service.create_weights(weights_b())
    contract_id = score_text(service, TEXT_1)
    ranking = service.list_ranked(weight_version_id="test-b")
    expected = fresh_result(monkeypatch, contract_id, TEXT_1, weights_b())
    assert [c.contract_id for c in ranking.contracts] == [contract_id]
    assert ranking.contracts[0].triage_score == expected["contract_score"].triage_score


def test_get_weights_unknown_raises(service):
    """An unknown weight version is NotFoundError."""
    with pytest.raises(NotFoundError):
        service.get_weights("nope")


# ----- Preview -----
def test_preview_ranks_without_saving(service):
    """Preview equals rescore_contracts(save=False) and leaves every table unchanged."""
    ids = [score_text(service, TEXT_1), score_text(service, TEXT_2)]
    before = row_counts(service.store)
    preview = service.preview(weights_b())
    expected = rescore_contracts(service.store, MODEL_VERSION, weights_b(), save=False)
    assert [(c.contract_id, c.rank, c.triage_score) for c in preview.contracts] == \
        [(s.contract_id, s.rank, s.triage_score) for s in expected]
    assert sorted(c.contract_id for c in preview.contracts) == sorted(ids)
    assert preview.saved is False
    assert row_counts(service.store) == before
    with pytest.raises(NotFoundError):
        service.get_weights("test-b")


def test_preview_subset_and_unknown_ids(service):
    """A subset ranks among itself; an id with no stored results is NotFoundError."""
    first = score_text(service, TEXT_1)
    score_text(service, TEXT_2)
    assert [c.contract_id for c in service.preview(weights_b(), [first]).contracts] == [first]
    with pytest.raises(NotFoundError, match="never-run"):
        service.preview(weights_b(), [first, "never-run"])


def test_service_requires_a_model_version(tmp_path, monkeypatch):
    """An empty model version key is refused."""
    with pytest.raises(ValueError, match="model_version"):
        ClauseTriageService(ArtifactStore(tmp_path / "api.db"), make_pipeline(monkeypatch), " ")
