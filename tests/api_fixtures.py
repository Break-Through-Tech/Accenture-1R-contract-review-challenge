"""
Shared builders for the API tests: a keyword pipeline (no model, no tokenizer) over a temporary
artifact store, wrapped in a ClauseTriageService. Reuses the store tests' fake predictor and texts.
"""
from pathlib import Path

import src.inference as inference
from src.api.service import ClauseTriageService
from src.chunking.chunker import chunk_contract
from src.inference import ContractInferencePipeline
from src.labels.categories import CATEGORIES
from src.registry.bundle import ClauseModelBundle
from src.risk.schema import WeightSet
from src.store.artifact_store import ArtifactStore
from tests.test_artifact_store import CUES, TEXT_1, TEXT_2, THRESHOLDS, KeywordPredictor, weights_a, weights_b

MODEL_VERSION = "keyword-model:1"
# Texts with keyword hits: c-1's grounded and Unknown clauses, and a second contract to rank against it
TEXTS = {"alpha": TEXT_1, "beta": TEXT_2}

__all__ = ["MODEL_VERSION", "TEXTS", "TEXT_1", "TEXT_2", "THRESHOLDS", "make_pipeline", "make_service",
           "weights_a", "weights_b"]


def patch_chunker(monkeypatch) -> None:
    """Swap the token chunker for 80/40-character windows, so no tokenizer is needed."""
    monkeypatch.setattr(inference, "chunk_sentence_windows",
                        lambda contract_id, text, *args: chunk_contract(contract_id, text, 80, 40) if text else [])


def make_pipeline(monkeypatch, weight_set: WeightSet | None = None, predictor=None) -> ContractInferencePipeline:
    """A pipeline with the keyword predictor (or the one given) that scores with weight_set (default: v1)."""
    patch_chunker(monkeypatch)
    bundle = ClauseModelBundle(predictor=predictor or KeywordPredictor(), categories=CATEGORIES,
                               thresholds=THRESHOLDS, model_name="keyword-model", config={}, seed=7)
    return ContractInferencePipeline(bundle, tokenizer=object(), category_cues=CUES,
                                     weight_set=weight_set or weights_a())


def make_service(database: Path, monkeypatch, weight_set: WeightSet | None = None,
                 predictor=None) -> ClauseTriageService:
    """A service over a store at database, scoring with the keyword pipeline under MODEL_VERSION."""
    return ClauseTriageService(ArtifactStore(database), make_pipeline(monkeypatch, weight_set, predictor),
                               MODEL_VERSION)


class FailingPredictor:
    """A classifier that always raises, standing in for a model crash mid-scoring."""

    def predict_proba(self, texts: list[str]):
        """Raise instead of predicting."""
        raise RuntimeError("model crashed at C:\\secret\\path")
