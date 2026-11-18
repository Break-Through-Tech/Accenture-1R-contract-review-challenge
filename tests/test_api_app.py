"""Tests for clausetriage/api/app.py and clausetriage/api/routers/: every endpoint over FastAPI's test client, with a keyword model."""
import io
import json
import zipfile

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from clausetriage.api import limits
from clausetriage.api.app import create_app
from clausetriage.api.routers import ServiceDep
from clausetriage.register.export import build_register, read_json, register_file_stem, register_frame, to_json
from clausetriage.risk.contract import aggregate_contracts
from tests.api_fixtures import MODEL_VERSION, TEXT_1, TEXT_2, FailingPredictor, make_pipeline, make_service, weights_b
from tests.pdf_fixtures import pdf_bytes
from tests.test_artifact_store import row_counts, weights_a


@pytest.fixture
def service(tmp_path, monkeypatch):
    """A service over a fresh store, scoring with the keyword pipeline under the v1 weights."""
    built = make_service(tmp_path / "api.db", monkeypatch)
    yield built
    built.store.close()


@pytest.fixture
def client(service):
    """A test client for the API without the Gradio page; server errors come back as responses."""
    with TestClient(create_app(service, ui_path=None), raise_server_exceptions=False) as test_client:
        yield test_client


def upload(client: TestClient, name: str, data: bytes, content_type: str = "application/octet-stream"):
    """POST one file to /contracts/upload."""
    return client.post("/contracts/upload", files={"file": (name, data, content_type)})


def uploaded_id(client: TestClient, name: str, text: str) -> str:
    """Upload text as a .txt file, check it was accepted, and return its contract id."""
    response = upload(client, name, text.encode("utf-8"), "text/plain")
    assert response.status_code == 202
    return response.json()["contract_id"]


def weights_body(weight_set) -> dict:
    """A WeightSet as the JSON a client posts."""
    return weight_set.model_dump(mode="json")


# ----- Server endpoints -----
def test_index_health_models_and_openapi(client):
    """The index links the docs, health names the model and default weights, and the schema renders."""
    assert client.get("/").json()["docs"] == "/docs"
    assert client.get("/health").json() == {"status": "ok", "model_version": MODEL_VERSION,
                                            "default_weight_version_id": "v1-placeholder"}
    assert client.get("/models").json() == {"server_model_version": MODEL_VERSION, "versions": []}
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/contracts/upload", "/contracts/text", "/contracts/{contract_id}/register.csv", "/weights",
            "/weights/preview", "/weights/{weight_version_id}"} <= set(paths)


# ----- Upload, status and reads -----
def test_upload_text_file_is_scored_and_readable(client):
    """An upload returns 202 and a queued job; after the background task it is complete, ranked and readable."""
    response = upload(client, "../../secret/Alpha Lease.txt", TEXT_1.encode(), "text/plain")
    assert response.status_code == 202
    job = response.json()
    assert (job["status"], job["source_name"], job["source_type"]) == ("queued", "Alpha Lease.txt", "txt")
    assert job["contract_id"].startswith("Alpha-Lease-")
    status = client.get(f"/contracts/{job['contract_id']}/status").json()
    assert (status["status"], status["error"]) == ("complete", None)
    ranking = client.get("/contracts").json()
    assert [c["contract_id"] for c in ranking["contracts"]] == [job["contract_id"]]
    detail = client.get(f"/contracts/{job['contract_id']}").json()
    assert (detail["rank"], detail["ranked_out_of"], detail["contract"]["text"]) == (1, 1, TEXT_1)
    assert len(detail["chunks"][0]["probabilities"]) == 41
    assert {"evidence", "unknowns", "clause_scores", "predictions", "contract_score"} <= set(detail)
    assert [j["contract_id"] for j in client.get("/jobs").json()] == [job["contract_id"]]


def test_upload_same_file_again_returns_existing_job(client):
    """A repeated upload answers 200 with the complete job and doesn't queue a rerun."""
    contract_id = uploaded_id(client, "a.txt", TEXT_1)
    again = upload(client, "a.txt", TEXT_1.encode())
    assert (again.status_code, again.json()["contract_id"], again.json()["status"]) == (200, contract_id, "complete")


def test_upload_pdf_reports_pages(client):
    """A PDF upload is read with page starts and its ranking entry counts the pages."""
    response = upload(client, "contract.pdf", pdf_bytes([TEXT_1, TEXT_2]), "application/pdf")
    assert response.status_code == 202
    (entry,) = client.get("/contracts").json()["contracts"]
    assert (entry["source_type"], entry["page_count"]) == ("pdf", 2)


def test_paste_text_is_scored(client):
    """Pasted text is accepted as JSON and scored under a 'Pasted text' id."""
    response = client.post("/contracts/text", json={"text": TEXT_2})
    assert response.status_code == 202
    assert response.json()["contract_id"].startswith("Pasted-text-")
    assert client.get(f"/contracts/{response.json()['contract_id']}/status").json()["status"] == "complete"


def test_ranking_matches_fresh_pipeline_and_query_parameters_select_versions(client, monkeypatch):
    """GET /contracts ranks like aggregate_contracts and accepts explicit model and weight versions."""
    ids = [uploaded_id(client, "a.txt", TEXT_1), uploaded_id(client, "b.txt", TEXT_2)]
    pipeline = make_pipeline(monkeypatch)
    fresh = [pipeline.predict_contract(cid, text)["clause_scores"] for cid, text in zip(ids, [TEXT_1, TEXT_2])]
    expected = aggregate_contracts([s for scores in fresh for s in scores], weights_a(), ids)
    ranking = client.get("/contracts", params={"model_version": MODEL_VERSION,
                                               "weight_version_id": "v1-placeholder"}).json()
    assert [(c["contract_id"], c["rank"]) for c in ranking["contracts"]] == [(s.contract_id, s.rank) for s in expected]


# ----- Failures and hostile input -----
def test_failed_scoring_is_reported_failed_and_not_listed(tmp_path, monkeypatch):
    """A model crash shows failed with a generic reason (no path or trace) and the contract stays unlisted."""
    service = make_service(tmp_path / "api.db", monkeypatch, predictor=FailingPredictor())
    with TestClient(create_app(service, ui_path=None)) as client:
        contract_id = upload(client, "a.txt", TEXT_1.encode()).json()["contract_id"]
        status = client.get(f"/contracts/{contract_id}/status").json()
        assert status["status"] == "failed" and "secret" not in status["error"]
        assert client.get("/contracts").json()["contracts"] == []
        assert client.get(f"/contracts/{contract_id}").status_code == 404


@pytest.mark.parametrize(("name", "data", "status", "message"), [
    ("tool.exe", b"MZ\x90\x00", 415, "type '.exe'"),
    ("old.doc", b"\xd0\xcf\x11\xe0", 415, "type '.doc'"),
    ("fake.pdf", b"just text", 400, "isn't a PDF"),
    ("fake.docx", b"%PDF-1.4\n", 400, "looks like a PDF"),
    ("binary.txt", b"\x00\x01\x02\x03" * 10, 400, "isn't plain text"),
    ("empty.txt", b"", 400, "empty"),
])
def test_upload_bad_files_are_refused_with_reasons(client, name, data, status, message):
    """Unsupported, mislabelled and empty files get a 4xx with the reader's message and nothing is queued."""
    response = upload(client, name, data)
    assert response.status_code == status
    assert message in response.json()["detail"]
    assert client.get("/jobs").json() == []


def test_upload_without_file_field_is_422_without_echo(client):
    """A form missing the file part is a 422 that names the field and doesn't echo the body."""
    response = client.post("/contracts/upload", data={"other": "secret-value"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "file"]
    assert "secret-value" not in response.text


def test_upload_over_declared_length_is_413_before_reading(client, monkeypatch):
    """A body whose Content-Length is over the upload limit is refused by the middleware."""
    monkeypatch.setattr(limits, "UPLOAD_BODY_LIMIT", 1_000)
    response = upload(client, "big.txt", b"a" * 5_000)
    assert response.status_code == 413
    assert client.get("/jobs").json() == []


def test_upload_file_over_size_limit_is_413(client, monkeypatch):
    """A file part over MAX_FILE_BYTES is refused by the handler even when the body limit allows it."""
    monkeypatch.setattr("clausetriage.api.routers.contracts.MAX_FILE_BYTES", 100)
    monkeypatch.setattr("clausetriage.api.service.MAX_FILE_BYTES", 100)
    response = upload(client, "big.txt", b"a" * 500)
    assert response.status_code == 413


def test_pasted_text_over_character_limit_is_422_without_echo(client, monkeypatch):
    """Pasted text longer than MAX_TEXT_CHARS is rejected by validation, without the text in the reply."""
    response = client.post("/contracts/text", json={"text": "x" * 2_000_001})
    assert response.status_code == 422
    assert len(response.content) < 1_000


def test_invalid_json_body_is_422(client):
    """A malformed JSON body is a 422, not a 500."""
    response = client.post("/contracts/text", content=b"{not json", headers={"content-type": "application/json"})
    assert response.status_code == 422


@pytest.mark.parametrize("path", ["/contracts/missing", "/contracts/missing/status",
                                  "/contracts/missing/register.csv", "/contracts/missing/register.json",
                                  "/weights/missing", "/contracts?weight_version_id=missing",
                                  "/contracts?model_version=missing"])
def test_unknown_ids_are_404(client, path):
    """Unknown contracts, weight versions and model versions are 404 with a message."""
    response = client.get(path)
    assert response.status_code == 404
    assert response.json()["detail"]


def test_unexpected_error_is_500_without_details(client, service, monkeypatch):
    """An unexpected exception becomes a plain 500 that leaks neither its message nor a trace."""
    def explode(*args, **kwargs):
        """Fail with a message that must not reach the client."""
        raise RuntimeError("C:\\secret\\database.sqlite is corrupt")
    monkeypatch.setattr(service, "list_ranked", explode)
    response = client.get("/contracts")
    assert (response.status_code, response.json()) == (500, {"detail": "Internal server error."})
    assert "secret" not in response.text and "Traceback" not in response.text


# ----- Registers -----
def test_register_downloads_match_build_register(client, monkeypatch):
    """register.csv and register.json are exactly what build_register gives for the same result and rank."""
    contract_id = uploaded_id(client, "alpha.txt", TEXT_1)
    other = uploaded_id(client, "beta.txt", TEXT_2)
    pipeline = make_pipeline(monkeypatch)
    fresh = {cid: pipeline.predict_contract(cid, text) for cid, text in [(contract_id, TEXT_1), (other, TEXT_2)]}
    ranked = aggregate_contracts([s for r in fresh.values() for s in r["clause_scores"]], weights_a(), list(fresh))
    score = next(s for s in ranked if s.contract_id == contract_id)
    expected = build_register(fresh[contract_id], score, text=TEXT_1, ranked_out_of=2)

    csv = client.get(f"/contracts/{contract_id}/register.csv")
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    assert csv.headers["content-disposition"] == f'attachment; filename="{register_file_stem(contract_id)}.csv"'
    assert csv.text == register_frame([expected]).to_csv(index=False, lineterminator="\n")

    body = client.get(f"/contracts/{contract_id}/register.json")
    assert body.text == to_json(expected)


def test_register_json_reads_back_with_read_json(client, tmp_path):
    """The downloaded JSON validates with register.export.read_json."""
    contract_id = uploaded_id(client, "alpha.txt", TEXT_1)
    path = tmp_path / "register.json"
    path.write_bytes(client.get(f"/contracts/{contract_id}/register.json").content)
    assert read_json(path).header.contract_id == contract_id


def test_register_under_another_weight_version(client):
    """A register can be read under a posted weight version; its rows carry that version."""
    contract_id = uploaded_id(client, "alpha.txt", TEXT_1)
    client.post("/weights", json=weights_body(weights_b()))
    rows = client.get(f"/contracts/{contract_id}/register.json", params={"weight_version_id": "test-b"}).json()["rows"]
    assert rows and {row["weight_version_id"] for row in rows} == {"test-b"}


# ----- Weights -----
def test_get_weights_lists_default_and_reads_one(client):
    """GET /weights names the default; GET /weights/{id} returns created_by, created_at and the full set."""
    listing = client.get("/weights").json()
    assert listing["default_weight_version_id"] == "v1-placeholder"
    one = client.get("/weights/v1-placeholder").json()
    assert one["is_default"] and one["version"]["created_by"] and one["version"]["created_at"]
    assert one["weight_set"]["weight_version_id"] == "v1-placeholder"


def test_post_weights_creates_version_and_rescores(client):
    """POST /weights answers 201, rescoring stored contracts; the ranking under it is readable at once."""
    uploaded_id(client, "alpha.txt", TEXT_1)
    response = client.post("/weights", json=weights_body(weights_b()))
    assert response.status_code == 201
    assert (response.json()["created"], response.json()["rescored_contracts"]) == (True, 1)
    assert [v["weight_version_id"] for v in client.get("/weights").json()["versions"]] == ["v1-placeholder", "test-b"]
    assert client.get("/contracts", params={"weight_version_id": "test-b"}).json()["contracts"]


def test_post_weights_never_changes_an_existing_version(client, service):
    """An identical repost is 200 with nothing written; changed values under an existing id are 409."""
    uploaded_id(client, "alpha.txt", TEXT_1)
    client.post("/weights", json=weights_body(weights_b()))
    before = (row_counts(service.store), client.get("/weights/test-b").json())
    again = client.post("/weights", json=weights_body(weights_b()))
    assert (again.status_code, again.json()["created"]) == (200, False)
    conflicting = client.post("/weights", json={**weights_body(weights_b()), "triage_top_k": 5})
    assert conflicting.status_code == 409
    original_default = client.post("/weights", json={**weights_body(weights_a()), "triage_max_weight": 0.9})
    assert original_default.status_code == 409
    assert (row_counts(service.store), client.get("/weights/test-b").json()) == before


@pytest.mark.parametrize(("change", "status"), [
    ({"signal_weights": {"category_severity": 0.9, "one_sidedness": 0.9, "magnitude": 0.0,
                         "contract_context": 0.0}}, 422),
    ({"medium_threshold": 0.9, "high_threshold": 0.1}, 422),
    ({"category_severity": {"Not A Category": 0.5}}, 422),
    ({"unexpected_field": 1}, 422),
    ({"weight_version_id": "has spaces/and slash"}, 400),
])
def test_post_invalid_weights_is_refused(client, service, change, status):
    """WeightSet validation failures are 422 and unsafe ids 400; nothing is stored."""
    before = row_counts(service.store)
    response = client.post("/weights", json={**weights_body(weights_b()), **change})
    assert response.status_code == status
    assert row_counts(service.store) == before


def test_preview_returns_ranking_and_saves_nothing(client, service):
    """POST /weights/preview ranks under the candidate with saved False and every table unchanged."""
    uploaded_id(client, "alpha.txt", TEXT_1)
    uploaded_id(client, "beta.txt", TEXT_2)
    before = row_counts(service.store)
    response = client.post("/weights/preview", json={"weight_set": weights_body(weights_b())})
    assert response.status_code == 200
    preview = response.json()
    assert (preview["saved"], preview["weight_version_id"], preview["ranked_out_of"]) == (False, "test-b", 2)
    assert row_counts(service.store) == before
    assert client.get("/weights/test-b").status_code == 404


def test_preview_unknown_contract_is_404(client):
    """A preview naming a contract with no stored results is 404."""
    uploaded_id(client, "alpha.txt", TEXT_1)
    response = client.post("/weights/preview", json={"weight_set": weights_body(weights_b()),
                                                     "contract_ids": ["never-scored"]})
    assert response.status_code == 404


def test_json_body_over_limit_is_413(client, monkeypatch):
    """A weight body over the JSON limit is refused before parsing."""
    monkeypatch.setattr(limits, "JSON_BODY_LIMIT", 100)
    response = client.post("/weights", content=json.dumps(weights_body(weights_b())).encode(),
                           headers={"content-type": "application/json"})
    assert response.status_code == 413


# ----- Extending the app -----
def test_extra_router_is_included_and_reaches_the_service(service):
    """A router passed to create_app is served and its handlers get the service through ServiceDep."""
    router = APIRouter()

    @router.get("/extra/model")
    def model(service: ServiceDep) -> dict:
        """Echo the service's model version."""
        return {"model_version": service.model_version}

    with TestClient(create_app(service, routers=[router], ui_path=None)) as client:
        assert client.get("/extra/model").json() == {"model_version": MODEL_VERSION}


# ----- Gradio page -----
def test_gradio_page_is_mounted_in_the_same_app(service, monkeypatch):
    """With ui_path set, the store-backed reviewer page is served under /ui/ next to the API."""
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    with TestClient(create_app(service)) as client:
        page = client.get("/ui/")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert "ClauseTriage reviewer" in page.text
        assert client.get("/health").status_code == 200
        # The page's own upload route is outside the API's body limits (Gradio applies its own)
        assert limits.default_limit_for("/ui")("/ui/gradio_api/upload") is None


def test_upload_zip_named_docx_is_refused_as_damaged(client):
    """A ZIP that isn't a Word document, uploaded as .docx, is a 400 with the Word reader's message."""
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("hello.txt", "not word")
    response = upload(client, "contract.docx", archive_bytes.getvalue())
    assert response.status_code == 400 and "Word" in response.json()["detail"]
