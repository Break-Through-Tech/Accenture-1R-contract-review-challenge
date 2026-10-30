import numpy as np
import pytest

from src.labels.categories import CATEGORIES
from src.registry.bundle import ClauseModelBundle
from src.registry.mlflow_registry import load_bundle, register_bundle
from src.store.artifact_store import ArtifactStore


class ConstantPredictor:
    def predict_proba(self, texts):
        return np.full((len(texts), len(CATEGORIES)), 0.75)


def _bundle():
    return ClauseModelBundle(
        predictor=ConstantPredictor(), categories=CATEGORIES,
        thresholds={category: 0.5 for category in CATEGORIES},
        model_name="test-model", config={"feature": "constant"}, seed=42,
    )


def test_bundle_checks_schema_and_predicts():
    bundle = _bundle()
    assert bundle.predict_proba(["contract"]).shape == (1, len(CATEGORIES))
    assert bundle.predict(["contract"]).all()
    with pytest.raises(ValueError, match="canonical CUAD"):
        ClauseModelBundle(ConstantPredictor(), list(reversed(CATEGORIES)), {}, "bad", {}, 1)


def test_mlflow_registry_roundtrip(tmp_path):
    registered = register_bundle(
        _bundle(), experiment_name="registry-test", registry_name="contract-review-test",
        metrics={"macro_f1": 0.5}, metadata={"split_seed": 42},
        tracking_uri=tmp_path / "mlflow.db",
    )
    assert registered.model_version == "1"
    loaded = load_bundle("contract-review-test", "1", tracking_uri=tmp_path / "mlflow.db")
    assert loaded.predict(["a", "b"]).shape == (2, len(CATEGORIES))


def test_store_persists_predictions_idempotently(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts.db")
    store.put_contract("contract-1", "abcdefgh")
    store.put_chunk(chunk_id="contract-1:0", contract_id="contract-1", chunk_index=0,
                    start=0, end=8, text="abcdefgh")
    store.put_probabilities("contract-1:0", "model-v1", ["Parties", "Insurance"], [0.8, 0.2])
    store.put_probabilities("contract-1:0", "model-v1", ["Parties", "Insurance"], [0.8, 0.2])
    store.add_evidence(evidence_id="ev-1", chunk_id="contract-1:0", category="Parties",
                       quote="cdef", start=2, end=6, model_version="model-v1")
    with pytest.raises(ValueError, match="absolute contract offsets"):
        store.add_evidence(evidence_id="ev-bad", chunk_id="contract-1:0", category="Parties",
                           quote="wrong", start=2, end=7, model_version="model-v1")
    assert store.get_probabilities("contract-1:0", "model-v1") == {"Parties": 0.8, "Insurance": 0.2}
    with pytest.raises(ValueError, match="different value"):
        store.put_probabilities("contract-1:0", "model-v1", ["Parties"], [0.9])
    with pytest.raises(ValueError, match="different text"):
        store.put_contract("contract-1", "different")
    store.close()
