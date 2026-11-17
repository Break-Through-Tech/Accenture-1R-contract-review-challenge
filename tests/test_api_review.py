"""
Tests for src/api/routers/review.py and the review methods of src/api/service.py: the review queue,
verdicts and their resolution over FastAPI's test client, with the keyword pipeline.

Stored contracts (tests/api_fixtures.py texts): 'alpha' (TEXT_1) has Governing Law grounded in chunk 0,
Termination For Convenience grounded in chunk 1 and an Insurance Unknown in chunk 3; 'beta' (TEXT_2)
has an Insurance Unknown in chunk 0 and Governing Law grounded in chunk 1.
"""
import pytest
from fastapi.testclient import TestClient

import src.api.__main__ as server
from src.api.app import create_app
from src.api.routers import review
from src.api.service import InvalidInputError, NotFoundError
from tests.api_fixtures import MODEL_VERSION, TEXT_1, TEXT_2, add_text, make_service
from tests.test_artifact_store import row_counts


@pytest.fixture
def service(tmp_path, monkeypatch):
    """A service over a fresh store with the keyword pipeline."""
    built = make_service(tmp_path / "review.db", monkeypatch)
    yield built
    built.store.close()


@pytest.fixture
def ids(service):
    """alpha and beta stored; returns their contract ids."""
    return {"alpha": add_text(service, TEXT_1, "alpha"), "beta": add_text(service, TEXT_2, "beta")}


@pytest.fixture
def client(service):
    """A test client for the API with the review router, as python -m src.api builds it (no page)."""
    with TestClient(create_app(service, routers=[review.router], ui_path=None),
                    raise_server_exceptions=False) as test_client:
        yield test_client


def verdict(chunk_id: str, category: str, verdict: str, reviewer: str = "Ana", **extra) -> dict:
    """A POST /feedback body."""
    return {"chunk_id": chunk_id, "category": category, "verdict": verdict, "reviewer": reviewer, **extra}


# ----- The queue -----
def test_queue_lists_unknowns_with_their_context(client, ids):
    """GET /review/queue returns both Unknowns, pending, with chunk text, reason and allowed verdicts."""
    page = client.get("/review/queue").json()
    assert (page["model_version"], page["total"], page["status_counts"]) == (
        MODEL_VERSION, 2, {"conflict": 0, "pending": 2, "reviewed": 0})
    first = page["items"][0]
    assert (first["category"], first["reason"], first["allowed_verdicts"]) == (
        "Insurance", "no_cue_match", ["accept", "reject"])
    assert first["chunk_text"] and first["reason_text"] and first["item_id"].endswith(f"|Insurance|{MODEL_VERSION}")


def test_queue_filters_by_contract_status_and_page(client, ids):
    """contract_id, status, offset and limit narrow the queue."""
    one = client.get("/review/queue", params={"contract_id": ids["beta"]}).json()
    assert [item["contract_id"] for item in one["items"]] == [ids["beta"]]
    assert client.get("/review/queue", params={"status": "reviewed"}).json()["total"] == 0
    paged = client.get("/review/queue", params={"offset": 1, "limit": 1}).json()
    assert (paged["total"], len(paged["items"])) == (2, 1)


def test_queue_before_anything_is_scored_is_empty(client):
    """With nothing stored, the queue is an empty page, not an error."""
    page = client.get("/review/queue").json()
    assert (page["total"], page["items"], page["status_counts"]) == (0, [], {"conflict": 0, "pending": 0, "reviewed": 0})


@pytest.mark.parametrize(("params", "status"), [
    ({"contract_id": "nope"}, 404), ({"model_version": "other:1"}, 404), ({"status": "done"}, 422),
    ({"limit": 0}, 422), ({"limit": 501}, 422), ({"offset": -1}, 422),
])
def test_queue_with_bad_parameters_is_refused(client, ids, params, status):
    """Unknown contracts and model versions are 404; out-of-range parameters are 422."""
    assert client.get("/review/queue", params=params).status_code == status


# ----- Recording verdicts -----
def test_accept_moves_the_item_to_reviewed(client, ids, service):
    """A 201 accept on an Unknown is stored with the reviewer and server time, and the queue reflects it."""
    response = client.post("/feedback", json=verdict(f"{ids['alpha']}:3", "Insurance", "accept", " Ana ",
                                                     note="checked"))
    assert response.status_code == 201
    record = response.json()
    assert (record["contract_id"], record["reviewer"], record["verdict"], record["note"], record["model_version"]) == (
        ids["alpha"], "Ana", "accept", "checked", MODEL_VERSION)
    counts = client.get("/review/queue").json()["status_counts"]
    assert counts == {"conflict": 0, "pending": 1, "reviewed": 1}
    reviewed = client.get("/review/queue", params={"status": "reviewed"}).json()["items"][0]
    assert (reviewed["resolved_label"], [v["reviewer"] for v in reviewed["current_verdicts"]]) == (1, ["Ana"])


def test_grounded_clause_and_missing_clause_verdicts_are_accepted(client, ids):
    """Reject on a grounded clause and missing on a category the model did not predict are both stored."""
    assert client.post("/feedback", json=verdict(f"{ids['alpha']}:0", "Governing Law", "reject")).status_code == 201
    assert client.post("/feedback", json=verdict(f"{ids['alpha']}:1", "Audit Rights", "missing")).status_code == 201
    listed = client.get("/feedback", params={"contract_id": ids["alpha"]}).json()
    assert [(f["category"], f["verdict"]) for f in listed] == [("Governing Law", "reject"), ("Audit Rights", "missing")]


def test_disagreement_becomes_a_conflict_in_queue_and_resolutions(client, ids):
    """A missing claim contradicted by a reject is listed as a conflict that offers missing/reject."""
    client.post("/feedback", json=verdict(f"{ids['alpha']}:1", "Audit Rights", "missing", "Ana"))
    client.post("/feedback", json=verdict(f"{ids['alpha']}:1", "Audit Rights", "reject", "Ben"))
    (conflict,) = client.get("/review/queue", params={"status": "conflict"}).json()["items"]
    assert (conflict["category"], conflict["detection"], conflict["allowed_verdicts"]) == (
        "Audit Rights", None, ["missing", "reject"])
    (resolution,) = client.get("/feedback/resolutions", params={"status": "conflict"}).json()
    assert (resolution["label"], resolution["chunk_id"]) == (None, f"{ids['alpha']}:1")
    assert client.get("/feedback/resolutions", params={"status": "resolved"}).json() == []


@pytest.mark.parametrize(("body", "status", "fragment"), [
    (("alpha:3", "Insurance", "missing"), 400, "'missing' is for clauses the model did not predict"),
    (("alpha:0", "Insurance", "accept"), 400, "use 'missing' for a clause the model missed"),
    (("alpha:0", "Insurance", "reject"), 400, "'reject' needs a prediction"),
    (("alpha:99", "Insurance", "accept"), 404, "No stored chunk"),
    (("alpha:3", "insurance", "accept"), 400, "unknown CUAD category 'insurance'"),
    (("alpha:3", "Insurance", "accept", "   "), 400, "reviewer must be a non-empty string"),
    (("alpha:3", "Insurance", "maybe"), 422, None),
    (("alpha:3", "Insurance", "accept", "x" * 121), 422, None),
])
def test_refused_verdicts_give_the_reason_and_write_nothing(client, ids, service, body, status, fragment):
    """Verdicts that don't fit the prediction, unknown chunks, bad categories, names or verdicts are refused."""
    chunk, category, choice, *reviewer = body
    contract, index = chunk.split(":")
    before = row_counts(service.store)
    response = client.post("/feedback", json=verdict(f"{ids[contract]}:{index}", category, choice,
                                                     *(reviewer or ["Ana"])))
    assert response.status_code == status
    if fragment:
        assert fragment in response.json()["detail"]
    assert row_counts(service.store) == before


def test_verdict_under_an_unknown_model_version_is_404(client, ids):
    """A verdict naming a model version the store doesn't know is refused."""
    response = client.post("/feedback", json=verdict(f"{ids['alpha']}:3", "Insurance", "accept",
                                                     model_version="other:1"))
    assert response.status_code == 404


def test_feedback_list_filters_and_unknown_contract(client, ids):
    """GET /feedback filters by chunk and category; an unknown contract is 404."""
    client.post("/feedback", json=verdict(f"{ids['alpha']}:3", "Insurance", "accept"))
    client.post("/feedback", json=verdict(f"{ids['beta']}:0", "Insurance", "reject"))
    assert len(client.get("/feedback").json()) == 2
    only = client.get("/feedback", params={"chunk_id": f"{ids['beta']}:0", "category": "Insurance"}).json()
    assert [f["verdict"] for f in only] == ["reject"]
    assert client.get("/feedback", params={"contract_id": "nope"}).status_code == 404


# ----- Service errors and wiring -----
def test_service_review_methods_raise_service_errors(service, ids):
    """The service maps the review layer's KeyError and ValueError to NotFoundError and InvalidInputError."""
    with pytest.raises(NotFoundError):
        service.review_queue(contract_id="nope")
    with pytest.raises(InvalidInputError, match="status must be one of"):
        service.review_queue(status="done")
    with pytest.raises(InvalidInputError, match="limit between 1 and 500"):
        service.review_queue(limit=0)
    with pytest.raises(InvalidInputError, match="status must be one of resolved, conflict"):
        service.list_resolutions(status="pending")
    with pytest.raises(InvalidInputError, match="'missing' is for clauses"):
        service.submit_feedback(verdict(f"{ids['alpha']}:3", "Insurance", "missing"))


def test_review_routes_come_only_from_the_extra_router(service):
    """create_app without routers has no review endpoints; the review router adds them."""
    with TestClient(create_app(service, ui_path=None)) as plain:
        assert plain.get("/review/queue").status_code == 404
    with TestClient(create_app(service, routers=[review.router], ui_path=None)) as extended:
        paths = extended.get("/openapi.json").json()["paths"]
        assert {"/review/queue", "/feedback", "/feedback/resolutions"} <= set(paths)


def test_server_entry_point_includes_the_review_router(service, monkeypatch):
    """python -m src.api serves the review endpoints."""
    served = {}
    monkeypatch.setattr("src.api.runtime.build_service", lambda args: service)
    monkeypatch.setattr("uvicorn.run", lambda app, **options: served.update(app=app))
    server.main(["--no-ui"])
    with TestClient(served["app"]) as client:
        assert client.get("/review/queue").status_code == 200
