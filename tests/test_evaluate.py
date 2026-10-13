import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification
from src.chunking.chunk import Chunk
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.labels.projection import LabeledChunk
from src.models.encoder.config import TrainConfig
from src.models.encoder.evaluate import (
    DEFAULT_THRESHOLD,
    MIN_VAL_POSITIVES,
    build_report,
    evaluate_checkpoint,
    main,
    save_evaluation,
    summarize,
    tune_thresholds,
)
from src.models.encoder.train import load_data

# Three categories: "golden" (AP 5/6), "perfect" (AP 1.0) and "nopos" (no positives)
GOLDEN_CATEGORIES = ["golden", "perfect", "nopos"]
GOLDEN_TRUE = np.array([[1, 1, 0],
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0]])
GOLDEN_PROB = np.array([[0.9, 0.9, 0.5],
                        [0.8, 0.8, 0.6],
                        [0.7, 0.2, 0.7],
                        [0.1, 0.1, 0.8]])


def golden_report() -> pd.DataFrame:
    """Tune thresholds and build the report for the three-category golden example."""
    thresholds = tune_thresholds(GOLDEN_TRUE, GOLDEN_PROB, GOLDEN_CATEGORIES)
    return build_report(GOLDEN_TRUE, GOLDEN_PROB, thresholds, GOLDEN_CATEGORIES)


def row(report: pd.DataFrame, category: str) -> pd.Series:
    """Return the report row of one category."""
    return report.loc[report["category"] == category].iloc[0]


# ----- tune_thresholds -----
def test_tune_thresholds_picks_the_f1_optimal_threshold_per_category():
    """Checks hand-worked thresholds: 'golden' peaks at 0.7 (P 2/3, R 1) and 'perfect' at 0.8."""
    thresholds = tune_thresholds(GOLDEN_TRUE, GOLDEN_PROB, GOLDEN_CATEGORIES)

    assert thresholds["golden"] == pytest.approx(0.7)
    assert thresholds["perfect"] == pytest.approx(0.8)


def test_tune_thresholds_category_without_positives_gets_the_default_without_warning():
    """Checks an untunable category gets DEFAULT_THRESHOLD and raises no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        thresholds = tune_thresholds(GOLDEN_TRUE, GOLDEN_PROB, GOLDEN_CATEGORIES)

    assert thresholds["nopos"] == DEFAULT_THRESHOLD == 0.5


def test_tune_thresholds_keeps_category_order_and_float_values():
    """Checks the mapping has one plain float per category, in the order of the categories."""
    thresholds = tune_thresholds(GOLDEN_TRUE, GOLDEN_PROB, GOLDEN_CATEGORIES)

    assert list(thresholds.keys()) == GOLDEN_CATEGORIES
    assert all(type(t) is float for t in thresholds.values())


def test_tune_thresholds_no_category_with_positives_gives_all_defaults():
    """Checks an all-negative set (nothing to tune) returns the default for every category."""
    thresholds = tune_thresholds(np.zeros((4, 3), dtype=int), GOLDEN_PROB, GOLDEN_CATEGORIES)

    assert thresholds == {"golden": 0.5, "perfect": 0.5, "nopos": 0.5}


def test_tune_thresholds_category_positive_everywhere_predicts_every_chunk_positive():
    """Checks an all-positive category gets a threshold at or below its lowest probability."""
    y_true = np.ones((4, 1), dtype=int)
    y_prob = np.array([[0.5], [0.6], [0.7], [0.8]])

    thresholds = tune_thresholds(y_true, y_prob, ["everywhere"])

    assert thresholds["everywhere"] <= 0.5


def test_tune_thresholds_is_independent_of_other_categories():
    """Checks a category's threshold does not change when other categories are added or removed."""
    alone = tune_thresholds(GOLDEN_TRUE[:, :1], GOLDEN_PROB[:, :1], ["golden"])
    together = tune_thresholds(GOLDEN_TRUE, GOLDEN_PROB, GOLDEN_CATEGORIES)

    assert alone["golden"] == together["golden"]


# ----- build_report -----
def test_build_report_golden_category_has_hand_computed_metrics():
    """Checks 'golden' at threshold 0.7: precision 2/3, recall 1, F1 0.8, support 2, AUPR 5/6."""
    golden = row(golden_report(), "golden")

    assert golden["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert golden["recall"] == pytest.approx(1.0)
    assert golden["f1"] == pytest.approx(0.8)
    assert golden["support"] == 2
    assert golden["aupr"] == pytest.approx(5 / 6)
    assert golden["threshold"] == pytest.approx(0.7)


def test_build_report_perfect_category_scores_one_everywhere():
    """Checks a perfectly ranked category has precision, recall, F1 and AUPR of 1.0."""
    perfect = row(golden_report(), "perfect")

    assert (perfect["precision"], perfect["recall"], perfect["f1"], perfect["aupr"]) == (1.0, 1.0, 1.0, 1.0)


def test_build_report_category_without_positives_has_nan_metrics_not_zero():
    """Checks undefined metrics are NaN (not a misleading 0) and the threshold is still reported."""
    nopos = row(golden_report(), "nopos")

    assert nopos["support"] == 0
    assert np.isnan([nopos["precision"], nopos["recall"], nopos["f1"], nopos["aupr"]]).all()
    assert nopos["threshold"] == 0.5


def test_build_report_is_sorted_by_support_most_positives_first():
    """Checks rows are ordered by support descending, so the unscored category comes last."""
    report = golden_report()

    assert report["support"].is_monotonic_decreasing
    assert report["category"].iloc[-1] == "nopos"


def test_build_report_flags_categories_below_the_minimum_support():
    """Checks low_support is True exactly when support is below MIN_VAL_POSITIVES."""
    n = MIN_VAL_POSITIVES
    y_true = np.array([[1, 1]] * n + [[0, 1]] * 3)
    y_prob = np.linspace(0.9, 0.1, num=n + 3).reshape(-1, 1).repeat(2, axis=1)
    thresholds = {"thin": 0.5, "enough": 0.5}

    report = build_report(y_true, y_prob, thresholds, ["thin", "enough"])

    assert row(report, "enough")["support"] == n + 3
    assert not row(report, "enough")["low_support"]
    assert row(report, "thin")["support"] == n
    assert not row(report, "thin")["low_support"]
    assert bool(row(build_report(y_true[: n - 1], y_prob[: n - 1], thresholds, ["thin", "enough"]),
                    "thin")["low_support"])


def test_build_report_has_the_documented_columns():
    """Checks the report columns match what the docstring promises."""
    assert list(golden_report().columns) == [
        "category", "precision", "recall", "f1", "support", "aupr", "threshold", "low_support",
    ]


# ----- summarize -----
def test_summarize_macro_averages_skip_categories_without_positives():
    """Checks macro F1 = (0.8 + 1) / 2, macro AUPR = (5/6 + 1) / 2 and the unscored count."""
    summary = summarize(golden_report())

    assert summary["macro_f1"] == pytest.approx(0.9)
    assert summary["macro_aupr"] == pytest.approx(11 / 12)
    assert summary["macro_recall"] == pytest.approx(1.0)
    assert summary["num_categories"] == 3
    assert summary["num_unscored"] == 1


def test_summarize_lists_the_low_support_categories():
    """Checks every category with fewer than MIN_VAL_POSITIVES positives is named."""
    assert set(summarize(golden_report())["low_support_categories"]) == {"golden", "perfect", "nopos"}


def test_summarize_is_json_serializable_even_when_nothing_can_be_scored():
    """Checks an all-NaN report gives None macros, which json can write (NaN is not valid JSON)."""
    thresholds = {c: 0.5 for c in GOLDEN_CATEGORIES}
    report = build_report(np.zeros((4, 3), dtype=int), GOLDEN_PROB, thresholds, GOLDEN_CATEGORIES)

    summary = summarize(report)

    assert summary["macro_f1"] is None
    assert summary["macro_aupr"] is None
    json.dumps(summary, allow_nan=False)


# ----- save_evaluation -----
def make_labeled_chunks(count: int) -> list[LabeledChunk]:
    """Build count short unlabeled chunks, from contracts c0, c1, ..., with chunk_index 0."""
    chunks = []
    for i in range(count):
        text = f"clause {i}"
        chunk = Chunk(contract_id=f"c{i}", start=3, end=3 + len(text), text=text, chunk_index=0)
        chunks.append(LabeledChunk(chunk=chunk, labels=[0] * len(CATEGORIES)))
    return chunks


def test_save_evaluation_writes_every_output_file(tmp_path):
    """Checks thresholds go next to the model and the rest into the eval folder."""
    report = golden_report()
    thresholds = {"golden": 0.7}

    eval_dir = save_evaluation(tmp_path, make_labeled_chunks(4), GOLDEN_PROB, thresholds, report, summarize(report))

    assert eval_dir == tmp_path / "eval"
    assert (tmp_path / "thresholds.json").exists()
    for name in ["val_probs.npy", "val_chunks.csv", "metrics.csv", "summary.json"]:
        assert (eval_dir / name).exists()


def test_save_evaluation_probabilities_and_chunk_ids_line_up_row_for_row(tmp_path):
    """Checks the saved probabilities reload exactly and val_chunks.csv has one matching row each."""
    chunks = make_labeled_chunks(4)
    report = golden_report()

    eval_dir = save_evaluation(tmp_path, chunks, GOLDEN_PROB, {}, report, summarize(report))

    assert np.array_equal(np.load(eval_dir / "val_probs.npy"), GOLDEN_PROB)
    ids = pd.read_csv(eval_dir / "val_chunks.csv")
    assert list(ids.columns) == ["contract_id", "chunk_index", "start", "end"]
    assert ids["contract_id"].tolist() == ["c0", "c1", "c2", "c3"]
    assert ids["start"].tolist() == [3, 3, 3, 3]
    assert ids["end"].tolist() == [3 + len(f"clause {i}") for i in range(4)]


def test_save_evaluation_thresholds_and_summary_reload_as_written(tmp_path):
    """Checks thresholds.json and summary.json parse back to exactly what was given."""
    report = golden_report()
    summary = summarize(report)
    thresholds = {"golden": 0.7, "perfect": 0.8, "nopos": 0.5}

    eval_dir = save_evaluation(tmp_path, make_labeled_chunks(4), GOLDEN_PROB, thresholds, report, summary)

    assert json.loads((tmp_path / "thresholds.json").read_text(encoding="utf-8")) == thresholds
    assert json.loads((eval_dir / "summary.json").read_text(encoding="utf-8")) == summary


def test_save_evaluation_metrics_csv_keeps_nan_as_missing(tmp_path):
    """Checks the metrics table reloads with the same rows, and unscored metrics stay missing."""
    report = golden_report()

    eval_dir = save_evaluation(tmp_path, make_labeled_chunks(4), GOLDEN_PROB, {}, report, summarize(report))
    reloaded = pd.read_csv(eval_dir / "metrics.csv")

    assert reloaded["category"].tolist() == report["category"].tolist()
    assert np.isnan(reloaded.loc[reloaded["category"] == "nopos", "f1"]).all()


# ----- evaluate_checkpoint: end to end on a tiny model and a small CUAD file -----
def make_tiny_model() -> RobertaForSequenceClassification:
    """Build a tiny randomly initialized RoBERTa classifier (no download) with a 41-way multi-label head."""
    roberta_config = RobertaConfig(
        vocab_size=50265, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=514, num_labels=len(CATEGORIES),
        problem_type="multi_label_classification", id2label=dict(enumerate(CATEGORIES)),
        label2id=CATEGORY_TO_INDEX, initializer_range=0.5,
    )
    return RobertaForSequenceClassification(roberta_config)


def make_config() -> TrainConfig:
    """Build a TrainConfig with small chunking sizes and max_length for the tiny end-to-end run."""
    return TrainConfig(
        model_name="distilbert/distilroberta-base", max_length=32, window_chars=20, overlap_chars=5,
        val_fraction=0.2, seed=42, lr=3.0e-5, epochs=1, batch_size=4, grad_accum=1, fp16=False,
        pos_weight_cap=50.0, output_dir="outputs/encoder", mlflow_experiment="test-experiment",
    )


def write_cuad_json(tmp_path, num_contracts: int = 10) -> Path:
    """Write a CUAD shaped JSON of num_contracts 60 character contracts, each with a Parties span at [0:5]."""
    data = {"data": []}
    for i in range(num_contracts):
        qas = [{"id": f"C{i}__Parties_0", "answers": [{"text": "Alpha", "answer_start": 0}]}]
        data["data"].append({"title": f"C{i}", "paragraphs": [{"context": "Alpha " * 10, "qas": qas}]})
    path = tmp_path / "train.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    """Save a tiny checkpoint, run evaluate_checkpoint on it once, and return the pieces to inspect."""
    root = tmp_path_factory.mktemp("evaluation")
    data_path = write_cuad_json(root)
    checkpoint = root / "best"
    config = make_config()
    make_tiny_model().save_pretrained(checkpoint)
    AutoTokenizer.from_pretrained("distilbert/distilroberta-base").save_pretrained(checkpoint)
    (checkpoint / "train_config.json").write_text(config.model_dump_json(), encoding="utf-8")

    report = evaluate_checkpoint(checkpoint, train_path=data_path, batch_size=3)
    return {"report": report, "checkpoint": checkpoint, "config": config, "data_path": data_path}


def test_evaluate_checkpoint_reports_every_category(evaluated):
    """Checks the returned report has one row per category and the documented columns."""
    report = evaluated["report"]

    assert sorted(report["category"]) == sorted(CATEGORIES)
    assert "aupr" in report.columns


def test_evaluate_checkpoint_uses_the_validation_split_from_training(evaluated):
    """Checks the saved chunk ids are exactly the validation chunks load_data produces for the saved config."""
    _, val_chunks = load_data(evaluated["config"], evaluated["data_path"])

    ids = pd.read_csv(evaluated["checkpoint"] / "eval" / "val_chunks.csv")

    assert len(ids) == len(val_chunks) == 8
    assert ids["contract_id"].tolist() == [lc.chunk.contract_id for lc in val_chunks]
    assert ids["start"].tolist() == [lc.chunk.start for lc in val_chunks]


def test_evaluate_checkpoint_saves_one_probability_row_per_validation_chunk(evaluated):
    """Checks val_probs.npy is (num_val_chunks, 41) with values between 0 and 1."""
    probs = np.load(evaluated["checkpoint"] / "eval" / "val_probs.npy")

    assert probs.shape == (8, len(CATEGORIES))
    assert ((probs >= 0) & (probs <= 1)).all()


def test_evaluate_checkpoint_counts_the_positives_in_the_validation_split(evaluated):
    """Checks Parties support is 2 (one positive first chunk in each of the 2 val contracts), others 0."""
    report = evaluated["report"]

    assert row(report, "Parties")["support"] == 2
    assert (report.loc[report["category"] != "Parties", "support"] == 0).all()
    assert row(report, "Parties")["aupr"] >= 0


def test_evaluate_checkpoint_writes_thresholds_for_every_category_next_to_the_model(evaluated):
    """Checks thresholds.json sits in the checkpoint folder with a value for each of the 41 categories."""
    thresholds = json.loads((evaluated["checkpoint"] / "thresholds.json").read_text(encoding="utf-8"))

    assert sorted(thresholds) == sorted(CATEGORIES)
    assert thresholds["Governing Law"] == DEFAULT_THRESHOLD


def test_evaluate_checkpoint_writes_a_summary_with_unscored_categories(evaluated):
    """Checks the summary counts the 40 categories that have no validation positives."""
    summary = json.loads((evaluated["checkpoint"] / "eval" / "summary.json").read_text(encoding="utf-8"))

    assert summary["num_categories"] == 41
    assert summary["num_unscored"] == 40


# ----- main: command-line options -----
def test_main_passes_the_checkpoint_and_batch_size_to_evaluate_checkpoint(monkeypatch):
    """Checks --checkpoint is converted to a Path and --batch-size to an int."""
    calls = {}
    monkeypatch.setattr("src.models.encoder.evaluate.evaluate_checkpoint",
                        lambda checkpoint, batch_size: calls.update(checkpoint=checkpoint, batch_size=batch_size))

    main(["--checkpoint", "outputs/encoder/best", "--batch-size", "64"])

    assert calls == {"checkpoint": Path("outputs/encoder/best"), "batch_size": 64}


def test_main_uses_batch_size_32_by_default(monkeypatch):
    """Checks the default scoring batch size."""
    calls = {}
    monkeypatch.setattr("src.models.encoder.evaluate.evaluate_checkpoint",
                        lambda checkpoint, batch_size: calls.update(batch_size=batch_size))

    main(["--checkpoint", "x"])

    assert calls["batch_size"] == 32


def test_main_without_a_checkpoint_exits_with_usage_error():
    """Checks the --checkpoint option is required."""
    with pytest.raises(SystemExit):
        main([])
