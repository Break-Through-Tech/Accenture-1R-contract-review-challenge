"""Tests for clausetriage/review/export.py: export contents, determinism, offset validation, conflicts and leakage."""
import json

import numpy as np
import pytest
from pydantic import ValidationError
from sqlalchemy import text

from clausetriage.labels.categories import CATEGORY_TO_INDEX
from clausetriage.review.export import (
    build_training_manifest, check_no_evaluation_overlap, export_training_examples, read_examples, text_sha256,
    to_partial_label_matrix, write_export,
)
from clausetriage.review.schema import TrainingManifest
from tests import review_fixtures as fixtures
from tests.review_fixtures import T0, TEXT_3, verdict
from tests.test_artifact_store import MODEL_VERSION, TEXT_1, TEXT_2, THRESHOLDS, weights_a

# Bind the shared fixtures under their own names (see tests/test_store_rescoring.py)
make_pipeline = fixtures.make_pipeline
store = fixtures.store
review_store = fixtures.review_store


def manifest(training: dict[str, str] | None = None, evaluation: dict[str, str] | None = None) -> TrainingManifest:
    """c-1 may train and c-2 is held out (by text hash), unless overridden; c-3 is listed nowhere."""
    return TrainingManifest(source="fixture", split_method="by hand",
                            training=training if training is not None else {"c-1": text_sha256(TEXT_1)},
                            evaluation=evaluation if evaluation is not None else {"c-2": text_sha256(TEXT_2)})


def give_feedback(store) -> None:
    """Feedback across every outcome: three resolved c-1 keys, a c-1 conflict, a held-out and an unlisted contract."""
    verdict(store, "c-1:3", "Insurance", "accept", "ana", minute=1)               # Unknown confirmed -> 1
    verdict(store, "c-1:0", "Governing Law", "accept", "ana", minute=2)           # grounded, then changed -> 0
    verdict(store, "c-1:0", "Governing Law", "reject", "ana", minute=3)
    verdict(store, "c-1:2", "Insurance", "missing", "ben", minute=4)              # missed clause -> 1
    verdict(store, "c-1:1", "Termination For Convenience", "accept", "ana", minute=5)  # conflict
    verdict(store, "c-1:1", "Termination For Convenience", "reject", "ben", minute=6)
    verdict(store, "c-2:0", "Insurance", "accept", "ana", minute=7)               # held-out contract
    verdict(store, "c-3:0", "Uncapped Liability", "accept", "ana", minute=8)      # not in the manifest


# ----- Contents -----
def test_export_examples_golden(review_store):
    """Three c-1 examples in chunk order with labels 0, 1, 1 and their chunk offsets and provenance."""
    give_feedback(review_store)
    export = export_training_examples(review_store, manifest())
    assert [(e.chunk_id, e.category, e.label, e.start, e.end) for e in export.examples] == [
        ("c-1:0", "Governing Law", 0, 0, 80), ("c-1:2", "Insurance", 1, 80, 160), ("c-1:3", "Insurance", 1, 120, 167)]
    first = export.examples[0]
    assert first.chunk_text == TEXT_1[0:80]
    assert (first.verdicts, first.reviewers, first.model_versions) == (["reject"], ["ana"], [MODEL_VERSION])
    assert (first.category_index, first.contract_sha256) == (CATEGORY_TO_INDEX["Governing Law"], text_sha256(TEXT_1))
    assert first.first_reviewed_at.minute == 2 and first.last_reviewed_at.minute == 3
    assert [v.feedback_id for v in first.votes] == [3]
    assert export.examples[1].verdicts == ["missing"] and export.examples[1].reviewers == ["ben"]
    assert (export.example_count, export.positive_count, export.test_data_used) == (3, 2, False)


def test_export_excludes_and_reports_conflicts_and_leakage(review_store):
    """The conflict, the held-out contract and the unlisted contract are each excluded with their reason."""
    give_feedback(review_store)
    export = export_training_examples(review_store, manifest())
    assert [(x.chunk_id, x.reason) for x in export.excluded] == [
        ("c-1:1", "conflict"), ("c-2:0", "evaluation_contract"), ("c-3:0", "not_in_training_manifest")]
    assert export.excluded_counts == {"conflict": 1, "evaluation_contract": 1, "not_in_training_manifest": 1}
    assert export.excluded[0].detail == "reviewers disagree: ana=accept, ben=reject"
    assert [v.reviewer for v in export.excluded[0].votes] == ["ana", "ben"]


def test_export_model_version_filter(review_store):
    """Feedback under another model version is left out when a version is given and included otherwise."""
    review_store.put_probabilities("c-1:3", "other-model:1", ["Governing Law"], [0.1])
    verdict(review_store, "c-1:3", "Governing Law", "missing", "ana", model_version="other-model:1")
    assert export_training_examples(review_store, manifest(), model_version=MODEL_VERSION).examples == []
    (example,) = export_training_examples(review_store, manifest()).examples
    assert (example.chunk_id, example.label, example.model_versions) == ("c-1:3", 1, ["other-model:1"])


def test_export_unknown_model_version_raises(review_store):
    """A model version that was never stored is an error, not an empty export."""
    with pytest.raises(KeyError):
        export_training_examples(review_store, manifest(), model_version="never-stored:1")


def test_export_empty_store_has_no_examples(review_store):
    """No feedback, no examples and no exclusions."""
    export = export_training_examples(review_store, manifest())
    assert (export.examples, export.excluded, export.excluded_counts) == ([], [], {})


# ----- Determinism and files -----
def test_write_export_is_byte_identical_and_reads_back(review_store, tmp_path):
    """Two exports of the same store write identical bytes, and examples.jsonl loads back equal."""
    give_feedback(review_store)
    first = write_export(export_training_examples(review_store, manifest()), tmp_path / "a")
    second = write_export(export_training_examples(review_store, manifest()), tmp_path / "b")
    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]
    examples = read_examples(first[0])
    assert examples == export_training_examples(review_store, manifest()).examples
    report = json.loads(first[1].read_text(encoding="utf-8"))
    assert (report["example_count"], report["test_data_used"], "examples" in report) == (3, False, False)
    assert report["manifest_sha256"] == manifest().manifest_sha256
    assert len(first[0].read_text(encoding="utf-8").splitlines()) == 3


# ----- Offset validation -----
@pytest.mark.parametrize("statement,detail", [
    ("UPDATE chunks SET text = upper(text) WHERE chunk_id = 'c-1:3'", "chunk text does not equal"),
    ("UPDATE chunks SET start = 121 WHERE chunk_id = 'c-1:3'", "chunk text does not equal"),
    ("UPDATE contracts SET text = upper(text) WHERE contract_id = 'c-1'", "SHA-256"),
])
def test_export_tampered_offsets_are_excluded(review_store, statement, detail):
    """A chunk or contract changed behind the store's back is reported as offset_mismatch, never exported."""
    verdict(review_store, "c-1:3", "Insurance", "accept", "ana")
    with review_store.engine.begin() as connection:
        connection.execute(text(statement))
    export = export_training_examples(review_store, manifest())
    (excluded,) = export.excluded
    assert (export.examples, excluded.reason) == ([], "offset_mismatch")
    assert detail in excluded.detail


# ----- Leakage safeguard -----
def test_export_held_out_text_under_another_id_is_excluded(review_store, make_pipeline):
    """A copy of held-out c-2 stored as 'c-2-copy' is still an evaluation contract (matched by text)."""
    result = make_pipeline(weights_a()).predict_contract("c-2-copy", TEXT_2)
    review_store.save_inference_result(result, text=TEXT_2, model_version=MODEL_VERSION, thresholds=THRESHOLDS,
                                       weight_set=weights_a())
    verdict(review_store, "c-2-copy:0", "Insurance", "accept", "ana")
    (excluded,) = export_training_examples(review_store, manifest()).excluded
    assert (excluded.contract_id, excluded.reason) == ("c-2-copy", "evaluation_contract")


def test_export_listed_id_with_other_text_is_excluded(review_store):
    """c-3 listed for training with a different text is refused: identity is the exact text, not the name."""
    verdict(review_store, "c-3:0", "Uncapped Liability", "accept", "ana")
    listed = manifest(training={"c-1": text_sha256(TEXT_1), "c-3": text_sha256(TEXT_3 + " ")})
    (excluded,) = export_training_examples(review_store, listed).excluded
    assert (excluded.reason, excluded.detail) == ("not_in_training_manifest", "contract text differs from the manifest's")


def test_export_training_text_under_another_id_is_exported(review_store):
    """c-3's exact text listed under another name may train: content, not the id, decides."""
    verdict(review_store, "c-3:0", "Uncapped Liability", "accept", "ana")
    (example,) = export_training_examples(review_store, manifest(training={"renamed": text_sha256(TEXT_3)})).examples
    assert (example.contract_id, example.label) == ("c-3", 1)


def test_export_held_out_contract_is_excluded_even_when_conflicted(review_store):
    """Leakage is reported before conflicts, so a held-out contract never shows up as a pending conflict."""
    verdict(review_store, "c-2:0", "Insurance", "accept", "ana")
    verdict(review_store, "c-2:0", "Insurance", "reject", "ben", minute=1)
    (excluded,) = export_training_examples(review_store, manifest()).excluded
    assert excluded.reason == "evaluation_contract"


@pytest.mark.parametrize("training,evaluation", [
    ({"c-1": "a"}, {"c-1": "b"}),
    ({"c-1": "a"}, {"c-2": "a"}),
])
def test_manifest_overlapping_training_and_evaluation_raises(training, evaluation):
    """A manifest can't list one contract (by id or by text) as both trainable and held out."""
    with pytest.raises(ValidationError, match="overlap"):
        TrainingManifest(source="x", split_method="x", training=training, evaluation=evaluation)


@pytest.mark.parametrize("name", ["test.json", "CUADv1.json", "Test.JSON"])
def test_build_training_manifest_refuses_test_files(tmp_path, name):
    """Files holding the test contracts are refused by name, before anything is read (the file doesn't exist)."""
    with pytest.raises(ValueError, match="held-out test contracts"):
        build_training_manifest(tmp_path / name)


def write_cuad(path, count: int) -> None:
    """A minimal CUAD-format file with `count` unannotated contracts named k-0, k-1, ..."""
    data = {"data": [{"title": f"k-{i}", "paragraphs": [{"context": f"Contract number {i}.", "qas": []}]}
                     for i in range(count)]}
    path.write_text(json.dumps(data), encoding="utf-8")


def test_build_training_manifest_splits_by_contract(tmp_path):
    """Five contracts at val_fraction 0.2 give four training and one held-out contract, keyed by text hash."""
    path = tmp_path / "train_separate_questions.json"
    write_cuad(path, 5)
    built = build_training_manifest(path, val_fraction=0.2, seed=42)
    assert (len(built.training), len(built.evaluation), built.seed, built.val_fraction) == (4, 1, 42, 0.2)
    assert set(built.training) | set(built.evaluation) == {f"k-{i}" for i in range(5)}
    (held_out,) = built.evaluation
    assert built.evaluation[held_out] == text_sha256(f"Contract number {held_out[2:]}.")
    assert built == build_training_manifest(path, val_fraction=0.2, seed=42)


def test_check_no_evaluation_overlap_raises_on_held_out_text(review_store):
    """A training run that passes its evaluation texts catches an example from one of them."""
    verdict(review_store, "c-1:3", "Insurance", "accept", "ana")
    examples = export_training_examples(review_store, manifest()).examples
    check_no_evaluation_overlap(examples, [TEXT_2, TEXT_3])
    with pytest.raises(ValueError, match="c-1"):
        check_no_evaluation_overlap(examples, [TEXT_1])


# ----- Training shape -----
def test_to_partial_label_matrix_golden(review_store):
    """Two c-1:3 labels and one c-1:0 label become two rows with three known entries; the rest are masked."""
    verdict(review_store, "c-1:0", "Governing Law", "reject", "ana")
    verdict(review_store, "c-1:3", "Insurance", "accept", "ana")
    verdict(review_store, "c-1:3", "Governing Law", "missing", "ana")
    examples = export_training_examples(review_store, manifest()).examples
    chunk_ids, texts, labels, mask = to_partial_label_matrix(examples)
    assert (chunk_ids, texts) == (["c-1:0", "c-1:3"], [TEXT_1[0:80], TEXT_1[120:167]])
    assert (labels.shape, labels.dtype, mask.dtype) == ((2, 41), np.int8, np.bool_)
    law, insurance = CATEGORY_TO_INDEX["Governing Law"], CATEGORY_TO_INDEX["Insurance"]
    assert np.argwhere(mask).tolist() == sorted([[0, law], [1, law], [1, insurance]])
    assert (labels[0, law], labels[1, law], labels[1, insurance], labels.sum()) == (0, 1, 1, 2)


def test_export_timestamps_are_utc(review_store):
    """Provenance timestamps are timezone-aware UTC."""
    verdict(review_store, "c-1:3", "Insurance", "accept", "ana")
    (example,) = export_training_examples(review_store, manifest()).examples
    assert example.first_reviewed_at == T0 and example.first_reviewed_at.utcoffset().total_seconds() == 0
