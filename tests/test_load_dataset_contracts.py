"""Tests for scripts/load_dataset_contracts.py: CUAD contracts scored into the store under their titles."""
import pytest

from scripts import evaluate_triage
from scripts.load_dataset_contracts import SOURCE_TYPE, load_contracts, parse_args
from clausetriage.labels.loader import Contract
from tests.api_fixtures import MODEL_VERSION, TEXT_1, TEXT_2, make_service

CONTRACTS = [Contract(contract_id="ALPHA CORP - Supply Agreement", text=TEXT_1, spans=[]),
             Contract(contract_id="Beta Inc", text=TEXT_2, spans=[])]


@pytest.fixture
def service(tmp_path, monkeypatch):
    """A service over a fresh store with the keyword pipeline."""
    built = make_service(tmp_path / "api.db", monkeypatch)
    yield built
    built.store.close()


# ----- Loading -----
def test_load_contracts_stores_each_under_its_title(service):
    """Each contract is scored under its CUAD title, with source_type 'cuad', and ranked by the service."""
    lines = []
    outcomes = load_contracts(service, CONTRACTS, progress=lines.append)
    assert outcomes == {"complete": 2}
    assert lines == ["[1/2] complete ALPHA CORP - Supply Agreement", "[2/2] complete Beta Inc"]
    record = service.store.get_contract("ALPHA CORP - Supply Agreement")
    assert (record.source_type, record.source_name, record.text) == (SOURCE_TYPE, "ALPHA CORP - Supply Agreement",
                                                                     TEXT_1)
    assert sorted(c.contract_id for c in service.list_ranked().contracts) == sorted(c.contract_id for c in CONTRACTS)
    assert service.store.list_run_contract_ids(MODEL_VERSION) == sorted(c.contract_id for c in CONTRACTS)


def test_load_contracts_rerun_skips_stored_contracts(service):
    """A second run scores nothing again."""
    load_contracts(service, CONTRACTS, progress=lambda line: None)
    assert load_contracts(service, CONTRACTS, progress=lambda line: None) == {"skipped": 2}


# ----- Data source -----
def test_dataset_source_is_the_training_file_never_test():
    """The contracts come from train_separate_questions.json; test.json is never the source."""
    assert evaluate_triage.TRAIN_PATH.name == "train_separate_questions.json"
    assert "test.json" not in evaluate_triage.TRAIN_PATH.as_posix()


def test_parse_args_limit_and_service_options():
    """--limit and the shared service options parse."""
    args = parse_args(["--limit", "3", "--model-version", "m:1"])
    assert (args.limit, args.model_version) == (3, "m:1")
