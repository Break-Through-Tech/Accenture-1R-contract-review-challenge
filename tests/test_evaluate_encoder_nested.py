import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_encoder_nested as runner
from scripts.evaluate_encoder_nested import (
    REQUIRED_RESULT_FILES,
    SEEDS,
    SplitMismatchError,
    baseline_dir_for,
    chunk_table,
    config_fingerprint,
    is_seed_complete,
    latest_checkpoint,
    parse_seeds,
    prune_checkpoints,
    results_dir_for,
    run_seed,
    split_contracts,
    split_provenance,
    verify_chunks_against_baseline,
    verify_split_against_baseline,
)
from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.labels.loader import AnswerSpan, Contract, load_cuad
from src.models.encoder.config import TrainConfig
from src.models.encoder.hub import HubOptions, load_pinned_tokenizer
from src.pipeline import prepare_contracts, to_model_inputs

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA = "d" * 64


def make_config(**overrides) -> TrainConfig:
    """Build the sentence_token TrainConfig the runner expects, with any fields overridden."""
    values = {
        "model_name": TOKENIZER_NAME, "model_revision": TOKENIZER_REVISION,
        "chunking_method": "sentence_token", "max_length": 512, "window_chars": 800,
        "overlap_chars": 200, "overlap_tokens": 64, "val_fraction": 0.2, "seed": 42,
        "lr": 3.0e-5, "epochs": 4, "batch_size": 16, "grad_accum": 2, "fp16": True,
        "pos_weight_cap": 50.0, "mlflow_experiment": "test",
    }
    values.update(overrides)
    return TrainConfig(**values)


def make_contracts(count: int = 10) -> list[Contract]:
    """Build short contracts C0..C<count-1>; even ones carry a Parties span on their first word."""
    contracts = []
    for index in range(count):
        text = f"Agreement number {index} between Alpha and Beta. The parties agree to the terms."
        spans = [AnswerSpan(category="Parties", start=0, end=9, text="Agreement")] if index % 2 == 0 else []
        contracts.append(Contract(contract_id=f"C{index}", text=text, spans=spans))
    return contracts


@pytest.fixture(scope="module")
def tokenizer():
    """Load the pinned DistilRoBERTa tokenizer from the local cache (no weights, no network)."""
    return load_pinned_tokenizer(TOKENIZER_NAME, TOKENIZER_REVISION, HubOptions(local_files_only=True))


@pytest.fixture(scope="module")
def train_contracts():
    """Load the official CUAD training file once (test.json is never read)."""
    return load_cuad(ROOT / "data/cuad/train_separate_questions.json")


def write_baseline(baseline_root: Path, contracts: list[Contract], tokenizer, seed: int = 42) -> Path:
    """Write a TF-IDF-style baseline run (report.json + word_char_tfidf npz labels) matching contracts."""
    groups = split_contracts(contracts, seed)
    provenance = split_provenance(groups, seed)
    labels = {name: to_model_inputs(prepare_contracts(groups[name], tokenizer, 512, 64))[1]
              for name in groups}
    provenance["chunk_counts"] = {name: len(labels[name]) for name in groups}
    baseline_dir = baseline_dir_for(baseline_root, seed)
    baseline_dir.mkdir(parents=True)
    report = {"data_sha256": DATA_SHA, "split": provenance, "test_data_used": False}
    (baseline_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    for name in ("calibration", "evaluation"):
        np.savez_compressed(baseline_dir / f"word_char_tfidf_{name}.npz", y_true=labels[name],
                            y_prob=np.zeros(labels[name].shape, dtype=np.float32))
    return baseline_dir


class FakeTraining:
    """Stand-ins for fit and scoring that record calls and write a finished best/ checkpoint."""

    def __init__(self):
        """Start with no recorded calls."""
        self.fit_calls = []
        self.score_calls = 0

    def fit(self, config, train_chunks, val_chunks, resume_from_checkpoint=None):
        """Record the call and write the files fit leaves in <output_dir>/best."""
        self.fit_calls.append({"config": config, "train": len(train_chunks), "val": len(val_chunks),
                               "resume": resume_from_checkpoint})
        best = Path(config.output_dir) / "best"
        best.mkdir(parents=True, exist_ok=True)
        (best / "log_history.json").write_text(json.dumps([{"epoch": 1, "eval_macro_aupr": 0.5}]))
        (best / "train_config.json").write_text(config.model_dump_json())
        return {"eval_macro_aupr": 0.5}

    def score(self, checkpoint_dir, partitions, max_length, batch_size):
        """Return near-perfect probabilities derived from each chunk's labels."""
        self.score_calls += 1
        return {name: np.asarray([item.labels for item in chunks], dtype=np.float32) * 0.8 + 0.1
                for name, chunks in partitions.items()}


@pytest.fixture
def setup(tmp_path, tokenizer):
    """A tiny contract set with a matching baseline, result/work roots and fake training."""
    contracts = make_contracts()
    write_baseline(tmp_path / "baselines", contracts, tokenizer)
    fake = FakeTraining()

    def run(**overrides):
        """Call run_seed for seed 42 on the tiny setup with any keyword overrides."""
        kwargs = dict(seed=42, contracts=contracts, config=make_config(), tokenizer=tokenizer,
                      data_sha256=DATA_SHA, results_root=tmp_path / "results", work_root=tmp_path / "work",
                      baseline_root=tmp_path / "baselines", fit_fn=fake.fit, score_fn=fake.score)
        kwargs.update(overrides)
        return run_seed(**kwargs)

    return {"run": run, "fake": fake, "tmp": tmp_path, "contracts": contracts}


# ----- paths and seed handling -----
def test_results_dir_for_names_one_folder_per_seed(tmp_path):
    """Checks each seed gets <root>/encoder_nested_seed<seed>."""
    assert results_dir_for(tmp_path, 7) == tmp_path / "encoder_nested_seed7"


@pytest.mark.parametrize("seed, expected", [
    (42, "word_char_tfidf_nested"), (7, "word_char_tfidf_nested_seed7"), (101, "word_char_tfidf_nested_seed101"),
])
def test_baseline_dir_for_matches_saved_tfidf_layout(seed, expected):
    """Checks seed 42 maps to the primary folder and other seeds to their _seed<seed> folders."""
    assert baseline_dir_for(Path("results"), seed) == Path("results") / expected


@pytest.mark.parametrize("seed", SEEDS)
def test_baseline_dir_for_every_predeclared_seed_exists_in_the_repo(seed):
    """Checks a saved TF-IDF run is committed for each of the five predeclared seeds."""
    assert (baseline_dir_for(ROOT / "results", seed) / "report.json").is_file()


def test_parse_seeds_keeps_predeclared_seeds_in_given_order():
    """Checks a valid subset is returned unchanged."""
    assert parse_seeds([101, 42]) == [101, 42]


@pytest.mark.parametrize("seeds, message", [([42, 42], "unique"), ([42, 5], "predeclared")])
def test_parse_seeds_rejects_duplicates_and_unknown_seeds(seeds, message):
    """Checks duplicates and seeds without a saved baseline are refused."""
    with pytest.raises(ValueError, match=message):
        parse_seeds(seeds)


def test_main_unknown_seed_exits_with_usage_error():
    """Checks the command line rejects a seed outside the predeclared five before loading anything."""
    with pytest.raises(SystemExit):
        runner.main(["--seeds", "3"])


# ----- config fingerprint -----
def test_config_fingerprint_ignores_output_dir_but_not_training_settings(tmp_path):
    """Checks moving the work folder keeps the fingerprint, while changing epochs changes it."""
    base = make_config()

    assert config_fingerprint(base) == config_fingerprint(base.model_copy(update={"output_dir": tmp_path}))
    assert config_fingerprint(base) != config_fingerprint(make_config(epochs=3))


# ----- completion state and checkpoints -----
def write_complete_results(result_dir: Path, status: str = "complete") -> None:
    """Create every required result file, with report.json carrying the given status."""
    result_dir.mkdir(parents=True)
    for name in REQUIRED_RESULT_FILES:
        (result_dir / name).write_text("x")
    (result_dir / "report.json").write_text(json.dumps({"status": status}))


def test_is_seed_complete_all_files_and_complete_status_is_true(tmp_path):
    """Checks a folder with every file and status complete counts as done."""
    write_complete_results(tmp_path / "r")
    assert is_seed_complete(tmp_path / "r")


@pytest.mark.parametrize("missing", REQUIRED_RESULT_FILES[1:])
def test_is_seed_complete_any_missing_file_is_false(tmp_path, missing):
    """Checks a seed interrupted before writing every file is not treated as done."""
    write_complete_results(tmp_path / "r")
    (tmp_path / "r" / missing).unlink()
    assert not is_seed_complete(tmp_path / "r")


@pytest.mark.parametrize("report_text", ['{"status": "running"}', "{not json"])
def test_is_seed_complete_unfinished_or_corrupt_report_is_false(tmp_path, report_text):
    """Checks a report without status complete, or a half-written one, does not count as done."""
    write_complete_results(tmp_path / "r")
    (tmp_path / "r" / "report.json").write_text(report_text)
    assert not is_seed_complete(tmp_path / "r")


def test_latest_checkpoint_compares_steps_numerically(tmp_path):
    """Checks checkpoint-120 beats checkpoint-60 and checkpoint-9 (not string order), ignoring best/."""
    for name in ("checkpoint-9", "checkpoint-60", "checkpoint-120", "best"):
        (tmp_path / name).mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-120"


@pytest.mark.parametrize("make_dir", [True, False])
def test_latest_checkpoint_none_when_no_checkpoints(tmp_path, make_dir):
    """Checks an empty or missing training folder has no checkpoint to resume."""
    train_dir = tmp_path / "train"
    if make_dir:
        train_dir.mkdir()
    assert latest_checkpoint(train_dir) is None


def test_prune_checkpoints_removes_epoch_checkpoints_and_keeps_best(tmp_path):
    """Checks only checkpoint-N folders are deleted."""
    for name in ("checkpoint-1", "checkpoint-2", "best"):
        (tmp_path / name).mkdir()

    removed = prune_checkpoints(tmp_path)

    assert {path.name for path in removed} == {"checkpoint-1", "checkpoint-2"}
    assert [path.name for path in tmp_path.iterdir()] == ["best"]


# ----- split verification -----
@pytest.mark.parametrize("seed", SEEDS)
def test_split_contracts_reproduces_saved_tfidf_split_digests(train_contracts, seed):
    """Checks the runner's split equals the saved TF-IDF split for each predeclared seed."""
    baseline = json.loads((baseline_dir_for(ROOT / "results", seed) / "report.json").read_text(encoding="utf-8"))
    provenance = split_provenance(split_contracts(train_contracts, seed), seed)

    assert provenance["contract_counts"] == baseline["split"]["contract_counts"]
    assert provenance["contract_id_sha256"] == baseline["split"]["contract_id_sha256"]


def test_split_contracts_partitions_are_disjoint_and_cover_all_contracts():
    """Checks 10 contracts split 6/2/2 with every contract in exactly one partition."""
    groups = split_contracts(make_contracts(10), 42)
    ids = [{contract.contract_id for contract in group} for group in groups.values()]

    assert [len(group) for group in groups.values()] == [6, 2, 2]
    assert set().union(*ids) == {f"C{i}" for i in range(10)}
    assert sum(len(group) for group in ids) == 10


def make_matching_report(seed: int = 42) -> tuple[dict, dict]:
    """Return a split provenance for the tiny contracts and a baseline report that matches it."""
    provenance = split_provenance(split_contracts(make_contracts(), seed), seed)
    report = {"data_sha256": DATA_SHA, "test_data_used": False, "split": json.loads(json.dumps(provenance))}
    return provenance, report


def test_verify_split_against_baseline_matching_report_passes():
    """Checks an identical split, data hash and test flag raise nothing."""
    provenance, report = make_matching_report()
    verify_split_against_baseline(provenance, DATA_SHA, report)


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r.update(data_sha256="e" * 64), "data_sha256"),
    (lambda r: r.update(test_data_used=True), "test_data_used"),
    (lambda r: r["split"].update(seed=7), "seed"),
    (lambda r: r["split"].update(evaluation_fraction=0.25), "evaluation_fraction"),
    (lambda r: r["split"]["contract_counts"].update(evaluation=3), "evaluation contract count"),
    (lambda r: r["split"]["contract_id_sha256"].update(calibration="0" * 64), "calibration contract-ID digest"),
])
def test_verify_split_against_baseline_any_difference_raises(mutate, message):
    """Checks each kind of split difference fails loudly and is named in the message."""
    provenance, report = make_matching_report()
    mutate(report)
    with pytest.raises(SplitMismatchError, match=message):
        verify_split_against_baseline(provenance, DATA_SHA, report)


def test_verify_chunks_against_baseline_wrong_chunk_count_raises(tmp_path):
    """Checks a different chunk count (e.g. another tokenizer revision) stops the run."""
    report = {"split": {"chunk_counts": {"training": 5, "calibration": 2, "evaluation": 2}}}
    with pytest.raises(SplitMismatchError, match="training has 6 chunks"):
        verify_chunks_against_baseline(42, {"training": 6, "calibration": 2, "evaluation": 2}, {}, report, tmp_path)


def test_verify_chunks_against_baseline_label_difference_raises(tmp_path):
    """Checks a single flipped evaluation label is caught against the saved npz."""
    labels = {"calibration": np.zeros((2, 41), dtype=np.int8), "evaluation": np.zeros((2, 41), dtype=np.int8)}
    for name, values in labels.items():
        np.savez_compressed(tmp_path / f"word_char_tfidf_{name}.npz", y_true=values)
    changed = {name: values.copy() for name, values in labels.items()}
    changed["evaluation"][1, 3] = 1
    report = {"split": {"chunk_counts": {"training": 4, "calibration": 2, "evaluation": 2}}}
    counts = {"training": 4, "calibration": 2, "evaluation": 2}

    verify_chunks_against_baseline(42, counts, labels, report, tmp_path)
    with pytest.raises(SplitMismatchError, match="evaluation labels differ"):
        verify_chunks_against_baseline(42, counts, changed, report, tmp_path)


# ----- chunk identifiers -----
def test_chunk_table_lists_identifiers_in_score_row_order(tokenizer):
    """Checks the table has contract_id, chunk_index, start, end for each chunk, in input order."""
    chunks = prepare_contracts(make_contracts(2), tokenizer, 512, 64)
    table = chunk_table(chunks)

    assert list(table.columns) == ["contract_id", "chunk_index", "start", "end"]
    assert table.to_dict(orient="records") == [
        {"contract_id": "C0", "chunk_index": 0, "start": 0, "end": len(make_contracts(2)[0].text)},
        {"contract_id": "C1", "chunk_index": 0, "start": 0, "end": len(make_contracts(2)[1].text)},
    ]


# ----- run_seed: full flow with fake training -----
def test_run_seed_writes_every_result_file_and_a_complete_report(setup):
    """Checks a fresh seed trains once, writes the full layout and a complete, provenance-rich report."""
    status = setup["run"]()
    result_dir = setup["tmp"] / "results" / "encoder_nested_seed42"
    report = json.loads((result_dir / "report.json").read_text(encoding="utf-8"))

    assert status["status"] == "complete"
    assert is_seed_complete(result_dir)
    assert (result_dir / "training_history.json").is_file()
    assert setup["fake"].fit_calls[0]["resume"] is None
    assert setup["fake"].fit_calls[0]["config"].seed == 42
    assert (setup["fake"].fit_calls[0]["train"], setup["fake"].fit_calls[0]["val"]) == (6, 2)
    assert report["split"]["contract_counts"] == {"training": 6, "calibration": 2, "evaluation": 2}
    assert report["baseline_verified_against"] == "word_char_tfidf_nested"
    assert report["test_data_used"] is False
    assert report["training"]["config_sha256"] == config_fingerprint(
        make_config().model_copy(update={"seed": 42}))


def test_run_seed_saved_scores_align_with_saved_chunk_rows(setup):
    """Checks evaluation npz rows and evaluation_chunks.csv rows describe the same chunks."""
    setup["run"]()
    result_dir = setup["tmp"] / "results" / "encoder_nested_seed42"
    predictions = np.load(result_dir / "evaluation_predictions.npz")
    chunks = pd.read_csv(result_dir / "evaluation_chunks.csv")
    evaluation_ids = [c.contract_id for c in split_contracts(setup["contracts"], 42)["evaluation"]]

    assert predictions["y_prob"].shape == predictions["y_true"].shape == (2, 41)
    assert chunks["contract_id"].tolist() == evaluation_ids


def test_run_seed_near_perfect_scores_give_golden_macro_metrics(setup):
    """Checks macro-F1 uses the fixed 41-category convention: Parties perfect, all else 0 -> 1/41."""
    status = setup["run"]()
    labels = np.load(setup["tmp"] / "results/encoder_nested_seed42/evaluation_predictions.npz")["y_true"]
    supported = int((labels.sum(axis=0) > 0).sum())

    assert supported == 1 and labels[:, CATEGORY_TO_INDEX["Parties"]].sum() >= 1
    assert status["macro_metrics"]["macro_f1"] == pytest.approx(1 / 41)
    assert status["macro_metrics"]["macro_aupr"] == pytest.approx(1.0)


def test_run_seed_completed_seed_is_skipped_without_training(setup):
    """Checks a second invocation neither trains nor scores and reports the seed as skipped."""
    setup["run"]()
    status = setup["run"]()

    assert status["status"] == "skipped"
    assert len(setup["fake"].fit_calls) == 1
    assert setup["fake"].score_calls == 1


def test_run_seed_completed_seed_with_different_config_raises(setup):
    """Checks finished results from another config are not silently reused."""
    setup["run"]()
    with pytest.raises(RuntimeError, match="different training config"):
        setup["run"](config=make_config(epochs=3))


def test_run_seed_force_retrains_a_completed_seed(setup):
    """Checks --force reruns training and scoring even when results are complete."""
    setup["run"]()
    setup["run"](force=True)

    assert len(setup["fake"].fit_calls) == 2
    assert setup["fake"].score_calls == 2


def test_run_seed_interrupted_training_resumes_from_latest_checkpoint(setup):
    """Checks a seed cut off mid-training continues from its highest checkpoint-N."""
    train_dir = setup["tmp"] / "work" / "seed42" / "train"
    for name in ("checkpoint-282", "checkpoint-564"):
        (train_dir / name).mkdir(parents=True)

    setup["run"]()

    assert Path(setup["fake"].fit_calls[0]["resume"]).name == "checkpoint-564"
    assert not list(train_dir.glob("checkpoint-*"))


def test_run_seed_keep_checkpoints_leaves_epoch_checkpoints(setup):
    """Checks --keep-checkpoints skips pruning after the seed finishes."""
    train_dir = setup["tmp"] / "work" / "seed42" / "train"
    (train_dir / "checkpoint-10").mkdir(parents=True)

    setup["run"](keep_checkpoints=True)

    assert (train_dir / "checkpoint-10").is_dir()


def test_run_seed_finished_training_without_results_only_rescores(setup):
    """Checks a seed interrupted after training (best/ saved) is scored without retraining."""
    setup["run"]()
    (setup["tmp"] / "results" / "encoder_nested_seed42" / "report.json").unlink()

    status = setup["run"]()

    assert status["status"] == "complete"
    assert len(setup["fake"].fit_calls) == 1
    assert setup["fake"].score_calls == 2


def test_run_seed_half_trained_run_with_different_config_raises(setup):
    """Checks a checkpoint started with another config is not resumed with this one."""
    work_dir = setup["tmp"] / "work" / "seed42"
    work_dir.mkdir(parents=True)
    (work_dir / "run_config.json").write_text(json.dumps({"config_sha256": "other"}))

    with pytest.raises(RuntimeError, match="different config"):
        setup["run"]()
    assert setup["fake"].fit_calls == []


def test_run_seed_dry_run_verifies_without_training_or_writing_results(setup):
    """Checks --dry-run stops after verification: no training, no results folder."""
    status = setup["run"](dry_run=True)

    assert status["status"] == "verified"
    assert status["chunk_counts"] == {"training": 6, "calibration": 2, "evaluation": 2}
    assert setup["fake"].fit_calls == []
    assert not (setup["tmp"] / "results").exists()


def test_run_seed_split_mismatch_stops_before_training(setup):
    """Checks a baseline with a different evaluation digest raises before any training."""
    report_path = setup["tmp"] / "baselines" / "word_char_tfidf_nested" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["split"]["contract_id_sha256"]["evaluation"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(SplitMismatchError, match="evaluation contract-ID digest"):
        setup["run"]()
    assert setup["fake"].fit_calls == []


def test_run_seed_wrong_training_data_hash_stops_before_training(setup):
    """Checks a different training file (data hash) is refused."""
    with pytest.raises(SplitMismatchError, match="data_sha256"):
        setup["run"](data_sha256="f" * 64)


def test_run_seed_missing_baseline_raises(setup):
    """Checks a seed without a saved TF-IDF run cannot be trained unverified."""
    with pytest.raises(SplitMismatchError, match="No saved TF-IDF baseline"):
        setup["run"](seed=7)


def test_run_seed_thresholds_cover_every_category(setup):
    """Checks thresholds.json has one entry per category, in CATEGORIES order."""
    setup["run"]()
    thresholds = json.loads((setup["tmp"] / "results/encoder_nested_seed42/thresholds.json").read_text())

    assert list(thresholds) == CATEGORIES
