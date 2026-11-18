import numpy as np
import pytest
from transformers import AutoTokenizer

from clausetriage.chunking.chunk import Chunk
from clausetriage.evidence.cues import load_category_cues
from clausetriage.inference import ContractInferencePipeline
from clausetriage.labels.categories import CATEGORIES
from clausetriage.registry.bundle import ClauseModelBundle


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
    import clausetriage.inference as inference

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


def test_contract_inference_runs_real_chunker_and_preserves_source_offsets():
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert/distilroberta-base", local_files_only=True,
    )
    text = "Introductory text. This Agreement is governed by Delaware law. More text."
    cues = {"governinglaw": ["governed", "delaware"]}
    pipeline = ContractInferencePipeline(
        make_bundle(), tokenizer=tokenizer, category_cues=cues,
        max_tokens=32, overlap_tokens=4,
    )

    result = pipeline.predict_contract("real-chunker", text)

    assert result["chunks"]
    assert all(text[item["start"]:item["end"]] == item["text"] for item in result["chunks"])
    evidence = result["grounded"][0].evidence
    assert text[evidence.start:evidence.end] == evidence.quote
    assert evidence.quote == "This Agreement is governed by Delaware law."


def test_contract_inference_routes_unmatched_prediction_to_unknown(monkeypatch):
    import clausetriage.inference as inference

    text = "A clause with no helpful cue."
    chunks = [Chunk(contract_id="c-1", start=0, end=len(text), text=text, chunk_index=0)]
    monkeypatch.setattr(inference, "chunk_sentence_windows", lambda *args: chunks)
    pipeline = ContractInferencePipeline(make_bundle(), tokenizer=object(), category_cues={})

    result = pipeline.predict_contract("c-1", text)

    assert result["grounded"] == []
    assert len(result["unknown"]) == 1
    assert result["unknown"][0].reason == "no_cue_match"


def test_contract_inference_handles_empty_contract(monkeypatch):
    import clausetriage.inference as inference

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


# ----- Risk scoring -----
def _single_chunk_pipeline(monkeypatch, text, cues, **kwargs):
    """A pipeline whose chunker returns the whole text as one chunk."""
    import clausetriage.inference as inference

    chunks = [Chunk(contract_id="c-1", start=0, end=len(text), text=text, chunk_index=0)]
    monkeypatch.setattr(inference, "chunk_sentence_windows", lambda *args: chunks)
    return ContractInferencePipeline(make_bundle(), tokenizer=object(), category_cues=cues, **kwargs)


def test_contract_inference_scores_grounded_clause_and_contract(monkeypatch):
    """A grounded prediction gets a ClauseScore, and one clause gives triage = 0.5c + 0.5c/3 = 2c/3."""
    text = "This Agreement is governed by Delaware law."
    result = _single_chunk_pipeline(monkeypatch, text, {"governinglaw": ["governed", "delaware"]}) \
        .predict_contract("c-1", text)

    (clause,) = result["clause_scores"]
    assert clause.category == "Governing Law" and clause.has_evidence
    assert clause.probability == pytest.approx(0.9)
    contract = result["contract_score"]
    assert contract.contract_id == "c-1" and contract.rank == 1 and contract.unknown_count == 0
    assert contract.triage_score == pytest.approx(2 * clause.composite_score / 3, abs=1e-4)
    assert contract.weight_version_id == clause.weight_version_id


def test_contract_inference_scores_unknown_predictions(monkeypatch):
    """An ungrounded prediction is still scored, without evidence, and counted as Unknown."""
    text = "A clause with no helpful cue."
    result = _single_chunk_pipeline(monkeypatch, text, {}).predict_contract("c-1", text)

    (clause,) = result["clause_scores"]
    assert not clause.has_evidence
    assert result["contract_score"].unknown_count == 1
    assert result["contract_score"].triage_score > 0


def test_contract_inference_empty_contract_scores_zero(monkeypatch):
    """A contract with no chunks still gets a ContractScore of 0."""
    import clausetriage.inference as inference

    monkeypatch.setattr(inference, "chunk_sentence_windows", lambda *args: [])
    result = ContractInferencePipeline(make_bundle(), tokenizer=object(), category_cues={}) \
        .predict_contract("c-empty", "")

    assert result["clause_scores"] == []
    assert result["contract_score"].triage_score == 0.0


def test_contract_inference_uses_given_weight_set(monkeypatch):
    """Scores record the weight version passed to the pipeline, not the default."""
    from clausetriage.risk.schema import load_weight_set

    custom = load_weight_set().model_copy(update={"weight_version_id": "test-v2", "triage_max_weight": 1.0})
    text = "A clause with no helpful cue."
    result = _single_chunk_pipeline(monkeypatch, text, {}, weight_set=custom).predict_contract("c-1", text)

    (clause,) = result["clause_scores"]
    assert clause.weight_version_id == "test-v2"
    assert result["contract_score"].triage_score == pytest.approx(clause.composite_score)
