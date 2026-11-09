import numpy as np
import pytest

from src.chunking.chunk import Chunk
from src.evidence.cues import load_category_cues
from src.inference import ContractInferencePipeline
from src.labels.categories import CATEGORIES
from src.registry.bundle import ClauseModelBundle


class FixedPredictor:
    def __init__(self, value=0.9):
        self.value = value

    def predict_proba(self, texts):
        scores = np.zeros((len(texts), len(CATEGORIES)), dtype=np.float32)
        scores[:, CATEGORIES.index("Governing Law")] = self.value
        return scores


def make_bundle(predictor=None):
    return ClauseModelBundle(
        predictor=predictor or FixedPredictor(), categories=CATEGORIES,
        thresholds={category: 0.5 for category in CATEGORIES},
        model_name="test", config={}, seed=42,
    )


def test_contract_inference_preserves_offsets_and_grounds_predictions(monkeypatch):
    import src.inference as inference

    text = "This Agreement is governed by Delaware law."
    chunks = [Chunk(contract_id="c-1", start=0, end=len(text), text=text, chunk_index=0)]
    monkeypatch.setattr(inference, "chunk_sentence_windows", lambda *args: chunks)
    cues = load_category_cues()
    cues["governinglaw"] = ["governed", "delaware"]
    pipeline = ContractInferencePipeline(make_bundle(), tokenizer=object(),
                                         category_cues=cues)

    result = pipeline.predict_contract("c-1", text)

    assert result["contract_id"] == "c-1"
    assert result["model_name"] == "test"
    assert len(result["chunks"]) == 1
    evidence = result["grounded"][0].evidence
    assert text[evidence.start:evidence.end] == evidence.quote
    assert result["grounded"][0].category == "Governing Law"
    assert result["unknown"] == []


def test_contract_inference_routes_unmatched_prediction_to_unknown(monkeypatch):
    import src.inference as inference

    text = "A clause with no helpful cue."
    chunks = [Chunk(contract_id="c-1", start=0, end=len(text), text=text, chunk_index=0)]
    monkeypatch.setattr(inference, "chunk_sentence_windows", lambda *args: chunks)
    pipeline = ContractInferencePipeline(make_bundle(), tokenizer=object(), category_cues={})

    result = pipeline.predict_contract("c-1", text)

    assert result["grounded"] == []
    assert len(result["unknown"]) == 1
    assert result["unknown"][0].reason == "no_cue_match"


def test_contract_inference_handles_empty_contract(monkeypatch):
    import src.inference as inference

    monkeypatch.setattr(inference, "chunk_sentence_windows", lambda *args: [])
    pipeline = ContractInferencePipeline(make_bundle(), tokenizer=object(), category_cues={})

    result = pipeline.predict_contract("c-empty", "")

    assert result["chunks"] == []
    assert result["grounded"] == []
    assert result["unknown"] == []


@pytest.mark.parametrize("contract_id,text,error", [
    ("", "text", ValueError), ("c-1", None, TypeError),
])
def test_contract_inference_validates_input(monkeypatch, contract_id, text, error):
    pipeline = ContractInferencePipeline(make_bundle(), tokenizer=object(), category_cues={})
    with pytest.raises(error):
        pipeline.predict_contract(contract_id, text)


def test_pipeline_rejects_invalid_chunk_configuration():
    with pytest.raises(ValueError, match="max_tokens"):
        ContractInferencePipeline(make_bundle(), tokenizer=object(), category_cues={}, max_tokens=513)
    with pytest.raises(ValueError, match="overlap_tokens"):
        ContractInferencePipeline(make_bundle(), tokenizer=object(), category_cues={},
                                  max_tokens=10, overlap_tokens=10)
