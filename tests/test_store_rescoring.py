"""Tests for src/store/rescoring.py: rescoring from the store equals running the pipeline fresh."""
import pytest

from src.risk.contract import aggregate_contracts
from src.risk.scoring import score_contract_clauses
from src.store.rescoring import rescore_contracts
from tests import test_artifact_store as store_tests
from tests.test_artifact_store import CONTRACTS, MODEL_VERSION, by_key, row_counts, save_all, weights_a, weights_b

# Reuse the store test fixtures under their own names; binding them here (rather than importing the
# names) keeps the linter from reading each test's fixture argument as a redefined import
make_pipeline = store_tests.make_pipeline
store = store_tests.store


def fresh_results(make_pipeline, weight_set) -> dict[str, dict]:
    """Run the pipeline on every fixture contract under weight_set without touching the store."""
    pipeline = make_pipeline(weight_set)
    return {contract_id: pipeline.predict_contract(contract_id, text) for contract_id, text in CONTRACTS.items()}


# ----- Rebuilding clause inputs -----
def test_get_clause_inputs_rescore_to_the_stored_scores_in_order(store, make_pipeline):
    """Scoring the rebuilt inputs under the original weights reproduces the pipeline's clause scores exactly."""
    results = save_all(store, make_pipeline, weights_a())
    for contract_id, result in results.items():
        rebuilt = store.get_clause_inputs(contract_id, MODEL_VERSION)
        assert score_contract_clauses(rebuilt, weights_a()) == result["clause_scores"]


def test_get_clause_inputs_grounded_first_then_unknowns(store, make_pipeline):
    """c-1 rebuilds Governing Law (chunk 0) and Termination (chunk 1) with evidence, then Insurance without."""
    save_all(store, make_pipeline, weights_a())
    clauses = store.get_clause_inputs("c-1", MODEL_VERSION)
    assert [(c.chunk_index, c.category, c.evidence is not None) for c in clauses] == [
        (0, "Governing Law", True), (1, "Termination For Convenience", True), (3, "Insurance", False)]
    assert clauses[1].probability == pytest.approx(0.8)


def test_get_clause_inputs_unknown_contract_raises(store, make_pipeline):
    """A contract never run under the model version can't be rebuilt."""
    save_all(store, make_pipeline, weights_a())
    with pytest.raises(KeyError):
        store.get_clause_inputs("c-1", "other-model:1")


# ----- Rescoring equivalence -----
def test_rescore_matches_fresh_pipeline_under_new_weights(store, make_pipeline):
    """Rescoring stored results under weights B gives exactly the scores and ranks of a fresh run under B."""
    save_all(store, make_pipeline, weights_a())
    returned = rescore_contracts(store, MODEL_VERSION, weights_b())

    fresh = fresh_results(make_pipeline, weights_b())
    fresh_clauses = [score for result in fresh.values() for score in result["clause_scores"]]
    expected = aggregate_contracts(fresh_clauses, weights_b(), list(CONTRACTS))
    assert returned == expected
    assert store.list_contract_scores(MODEL_VERSION, "test-b") == expected
    for contract_id, result in fresh.items():
        assert store.get_clause_scores(contract_id, MODEL_VERSION, "test-b") == by_key(result["clause_scores"])


def test_rescore_changes_scores_and_keeps_old_version(store, make_pipeline):
    """Weights B score c-1 differently from A, and the A rows stay untouched next to the B rows."""
    results = save_all(store, make_pipeline, weights_a())
    rescore_contracts(store, MODEL_VERSION, weights_b())
    old = store.get_contract_score("c-1", MODEL_VERSION, "v1-placeholder")
    new = store.get_contract_score("c-1", MODEL_VERSION, "test-b")
    assert old.triage_score == results["c-1"]["contract_score"].triage_score
    assert new.triage_score != old.triage_score
    assert new.weight_set_sha256 == weights_b().content_sha256
    assert store.get_weight_version("test-b") == weights_b()


def test_rescore_twice_is_a_noop(store, make_pipeline):
    """Rescoring under the same weights again adds no rows."""
    save_all(store, make_pipeline, weights_a())
    rescore_contracts(store, MODEL_VERSION, weights_b())
    before = row_counts(store)
    rescore_contracts(store, MODEL_VERSION, weights_b())
    assert row_counts(store) == before


def test_rescore_under_original_weights_matches_stored_rows(store, make_pipeline):
    """Rescoring under the weights the pipeline used is accepted as identical to what is stored."""
    save_all(store, make_pipeline, weights_a())
    before = row_counts(store)
    rescore_contracts(store, MODEL_VERSION, weights_a())
    assert row_counts(store) == before


def test_rescore_preview_writes_nothing(store, make_pipeline):
    """save=False returns the ranking without storing scores or the weight version."""
    save_all(store, make_pipeline, weights_a())
    before = row_counts(store)
    preview = rescore_contracts(store, MODEL_VERSION, weights_b(), save=False)
    assert [s.contract_id for s in preview][-1] == "c-empty"
    assert row_counts(store) == before
    assert store.list_contract_scores(MODEL_VERSION, "test-b") == []


def test_rescore_subset_ranks_only_those_contracts(store, make_pipeline):
    """Rescoring a subset ranks within that subset."""
    save_all(store, make_pipeline, weights_a())
    scores = rescore_contracts(store, MODEL_VERSION, weights_b(), ["c-2", "c-empty"])
    assert [(s.contract_id, s.rank) for s in scores] == [("c-2", 1), ("c-empty", 2)]


def test_rescore_unknown_contract_raises(store, make_pipeline):
    """Asking to rescore a contract with no stored results raises KeyError and writes nothing."""
    save_all(store, make_pipeline, weights_a())
    before = row_counts(store)
    with pytest.raises(KeyError):
        rescore_contracts(store, MODEL_VERSION, weights_b(), ["c-1", "never-run"])
    assert row_counts(store) == before
