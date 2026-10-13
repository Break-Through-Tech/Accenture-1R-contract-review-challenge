"""Pinned PR #48 integration checks using the exact saved tokenizer and development IDs."""

import hashlib
import json
import subprocess

import pytest
from transformers import AutoTokenizer

from src.chunking.evaluation import evaluate_chunks, validate_chunks
from src.chunking.gwen_adapter import GWEN_COMMIT, GWEN_SOURCE_PATH, load_gwen_prototype, saved_development
from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION, token_length
from src.labels.categories import CATEGORY_TO_INDEX
from src.labels.loader import AnswerSpan, Contract
from src.labels.projection import project_labels


@pytest.fixture(scope="module")
def tokenizer():
    """Use our pinned real tokenizer locally, with no download or alias substitution."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION,
                                        use_fast=True, cache_dir=".venv/hf-cache", local_files_only=True)


@pytest.fixture
def saved_case(tmp_path):
    """Construct a 408-ID split manifest whose order differs from dataset order."""
    contracts = [Contract(contract_id=str(i), text="source", spans=[]) for i in range(408)]
    dataset = tmp_path / "training_fixture.json"
    dataset.write_text("fixture source", encoding="utf-8")
    metadata = {"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "split": {"development_ids": [str(i) for i in reversed(range(326))],
                          "validation_ids": [str(i) for i in range(326, 408)],
                          "seed": 42, "val_fraction": 0.2, "validation_evaluated": False}}
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
    return contracts, dataset, report, metadata


# Pin provenance and preserve the original algorithm rather than repairing it.
def test_pinned_adapter_source_hash_matches_git_blob() -> None:
    """The executed implementation is the exact reviewed PR #48 source blob."""
    prototype, digest = load_gwen_prototype()
    blob = subprocess.check_output(["git", "show", f"{GWEN_COMMIT}:{GWEN_SOURCE_PATH}"])
    assert digest == hashlib.sha256(blob).hexdigest()
    assert prototype.__code__.co_filename == f"git:{GWEN_COMMIT}:{GWEN_SOURCE_PATH}"


def test_gwen_explicit_480_120_real_tokenizer_and_projection(tokenizer) -> None:
    """The explicit prototype settings retain exact slices and fit real input token counts."""
    prototype, _ = load_gwen_prototype()
    answer = "DISTRIBUTOR AGREEMENT"
    text = answer + "\n" + "Both parties agree to the terms of this contract. " * 300
    contract = Contract(contract_id="T", text=text, spans=[
        AnswerSpan(category="Document Name", start=0, end=len(answer), text=answer)])
    chunks = prototype(contract, tokenizer, window_tokens=480, overlap_tokens=120)
    validate_chunks(contract, chunks)
    assert len(chunks) > 1
    assert all(token_length(chunk.text, tokenizer) <= 512 for chunk in chunks)
    labels = project_labels(chunks, contract.spans)
    index = CATEGORY_TO_INDEX["Document Name"]
    assert all(lc.labels[index] == int(lc.chunk.start < len(answer) and lc.chunk.end > 0) for lc in labels)
    report = evaluate_chunks([contract], {"T": chunks})
    assert report.metrics.contained_pct == 100
    assert report.metrics.unique_annotations == 1


def test_adapter_does_not_repair_nonadvancing_unicode_windows(tokenizer) -> None:
    """A pathological one-token Unicode setting preserves Gwen's output and fails our audit."""
    prototype, _ = load_gwen_prototype()
    contract = Contract(contract_id="T", text="🙂🙂", spans=[])
    chunks = prototype(contract, tokenizer, window_tokens=1, overlap_tokens=0)
    with pytest.raises(ValueError, match="advance"):
        validate_chunks(contract, chunks)


# Saved IDs are selected verbatim; no random split is recomputed.
def test_saved_development_preserves_order_and_excludes_validation(saved_case) -> None:
    """The saved order is authoritative even when the dataset order differs."""
    contracts, dataset, report, metadata = saved_case
    selected, returned = saved_development(contracts, report, dataset)
    assert [c.contract_id for c in selected] == metadata["split"]["development_ids"]
    assert not {c.contract_id for c in selected} & set(metadata["split"]["validation_ids"])
    assert returned == metadata


def test_saved_development_changed_dataset_is_rejected(saved_case) -> None:
    """Dataset byte changes invalidate reuse of the saved comparison split/settings."""
    contracts, dataset, report, _ = saved_case
    dataset.write_text("changed source")
    with pytest.raises(ValueError, match="Dataset does not match"):
        saved_development(contracts, report, dataset)


@pytest.mark.parametrize("change", ["duplicate", "overlap", "validation_evaluated"])
def test_saved_development_invalid_manifest_is_rejected(saved_case, change: str) -> None:
    """Duplicated or overlapping IDs and evaluated validation manifests cannot be reused."""
    contracts, dataset, report, metadata = saved_case
    split = metadata["split"]
    if change == "duplicate":
        split["development_ids"][1] = split["development_ids"][0]
    elif change == "overlap":
        split["validation_ids"][0] = split["development_ids"][0]
    else:
        split["validation_evaluated"] = True
    report.write_text(json.dumps({"metadata": metadata}))
    with pytest.raises(ValueError, match="Saved split"):
        saved_development(contracts, report, dataset)
