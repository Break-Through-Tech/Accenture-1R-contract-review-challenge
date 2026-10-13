import json
import warnings
from pathlib import Path
import pytest
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, TrainingArguments)
from src.chunking.chunk import Chunk
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.labels.projection import LabeledChunk
from src.metrics import compute_transformer_metrics
from src.models.encoder.config import TrainConfig
from src.models.encoder.dataset import ChunkCollator
from src.models.encoder.loss import WeightedBCETrainer
from src.models.encoder.train import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    TRAIN_PATH,
    build_model,
    build_trainer,
    build_training_args,
    fit,
    load_data,
    main,
    train,
)


def make_config(**overrides) -> TrainConfig:
    """Build a valid TrainConfig with small chunking sizes, with any given fields overridden."""
    values = {
        "model_name": "distilbert/distilroberta-base",
        "max_length": 512,
        "window_chars": 20,
        "overlap_chars": 5,
        "val_fraction": 0.2,
        "seed": 42,
        "lr": 3.0e-5,
        "epochs": 4,
        "batch_size": 16,
        "grad_accum": 2,
        "fp16": False,
        "pos_weight_cap": 50.0,
        "mlflow_experiment": "test-experiment",
    }
    values.update(overrides)
    return TrainConfig(**values)


def write_cuad_json(tmp_path, num_contracts: int = 10) -> Path:
    """Write a CUAD shaped JSON of num_contracts 60 character contracts, each with a Parties span at [0:5]."""
    data = {"data": []}
    for i in range(num_contracts):
        context = "Alpha " * 10
        qas = [{"id": f"C{i}__Parties_0", "answers": [{"text": "Alpha", "answer_start": 0}]}]
        data["data"].append({"title": f"C{i}", "paragraphs": [{"context": context, "qas": qas}]})

    path = tmp_path / "train.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ----- build_training_args: config fields reach the arguments -----
def test_build_training_args_returns_training_arguments():
    """Checks the result is a Hugging Face TrainingArguments object."""
    assert isinstance(build_training_args(make_config()), TrainingArguments)


def test_build_training_args_config_fields_map_to_matching_arguments():
    """Checks each optimization and seed field in the config reaches its argument."""
    args = build_training_args(make_config(lr=1.0e-4, epochs=3, batch_size=8, grad_accum=4, seed=7))

    assert args.learning_rate == 1.0e-4
    assert args.num_train_epochs == 3
    assert args.per_device_train_batch_size == 8
    assert args.gradient_accumulation_steps == 4
    assert args.seed == 7
    assert args.data_seed == 7


def test_build_training_args_eval_batch_size_is_twice_the_train_batch_size():
    """Checks evaluation uses a batch twice the training batch, since it needs no gradients."""
    assert build_training_args(make_config(batch_size=8)).per_device_eval_batch_size == 16


def test_build_training_args_selects_best_checkpoint_by_macro_aupr_each_epoch():
    """Checks it evaluates and saves per epoch and keeps the best macro_aupr checkpoint."""
    args = build_training_args(make_config())

    assert args.eval_strategy == "epoch"
    assert args.save_strategy == "epoch"
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "macro_aupr"
    assert args.greater_is_better is True


def test_build_training_args_reporting_is_off_until_mlflow_is_added():
    """Checks no external tracker is enabled, so MLflow does not start logging on its own."""
    assert build_training_args(make_config()).report_to == []


# ----- build_training_args: output directory -----
def test_build_training_args_output_dir_set_in_config_is_used(tmp_path):
    """Checks a configured output_dir is passed through unchanged."""
    args = build_training_args(make_config(output_dir=tmp_path / "run"))

    assert Path(args.output_dir) == tmp_path / "run"


def test_build_training_args_unset_output_dir_falls_back_to_default():
    """Checks a missing output_dir uses the default, not a folder named 'None'."""
    args = build_training_args(make_config(output_dir=None))

    assert Path(args.output_dir) == DEFAULT_OUTPUT_DIR


# ----- build_training_args: fp16 depends on CUDA -----
def test_build_training_args_fp16_without_cuda_is_turned_off_with_warning(monkeypatch):
    """Checks fp16 requested on a machine without CUDA falls back to full precision and warns."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.warns(UserWarning, match="fp16"):
        args = build_training_args(make_config(fp16=True))

    assert args.fp16 is False


def test_build_training_args_fp16_off_in_config_stays_off_without_warning(monkeypatch):
    """Checks fp16=False in the config stays off and raises no warning."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        args = build_training_args(make_config(fp16=False))

    assert args.fp16 is False


# ----- load_data: split and chunking on a small CUAD file -----
def test_load_data_returns_non_empty_lists_of_labeled_chunks(tmp_path):
    """Checks both splits are lists of LabeledChunk with at least one chunk each."""
    train, val = load_data(make_config(), write_cuad_json(tmp_path))

    assert train and val
    assert all(isinstance(lc, LabeledChunk) for lc in train + val)


def test_load_data_splits_whole_contracts_with_no_leakage(tmp_path):
    """Checks 10 contracts split 8/2 with no contract in both, and none dropped."""
    train, val = load_data(make_config(val_fraction=0.2), write_cuad_json(tmp_path, num_contracts=10))

    train_ids = {lc.chunk.contract_id for lc in train}
    val_ids = {lc.chunk.contract_id for lc in val}

    assert len(train_ids) == 8
    assert len(val_ids) == 2
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {f"C{i}" for i in range(10)}


def test_load_data_same_seed_gives_the_same_split(tmp_path):
    """Checks loading twice with the same config gives identical chunks in identical order."""
    path = write_cuad_json(tmp_path)

    first = load_data(make_config(), path)
    second = load_data(make_config(), path)

    assert first == second


def test_load_data_chunks_follow_the_configured_window_and_overlap(tmp_path):
    """Checks chunks are at most window long and start window - overlap apart within a contract."""
    train, _ = load_data(make_config(window_chars=20, overlap_chars=5), write_cuad_json(tmp_path))

    first_contract = [lc.chunk for lc in train if lc.chunk.contract_id == train[0].chunk.contract_id]

    assert all(c.end - c.start <= 20 for c in first_contract)
    assert [c.start for c in first_contract] == [0, 15, 30, 45]


def test_load_data_projects_answer_spans_onto_the_chunks(tmp_path):
    """Checks the Parties span at [0:5] labels only the first chunk of each contract."""
    train, _ = load_data(make_config(window_chars=20, overlap_chars=5), write_cuad_json(tmp_path))
    parties = CATEGORY_TO_INDEX["Parties"]

    for lc in train:
        assert lc.labels[parties] == (1 if lc.chunk.chunk_index == 0 else 0)


# ----- load_data: only the official train file is the default input -----
def test_train_path_points_to_the_official_train_file_not_the_test_file():
    """Checks the default input is train_separate_questions.json, never test.json."""
    assert TRAIN_PATH.name == "train_separate_questions.json"
    assert "test" not in TRAIN_PATH.name


# ----- Shared helpers for the model-level tests -----
def make_tiny_model() -> RobertaForSequenceClassification:
    """Build a tiny randomly initialized RoBERTa classifier (no download) with a 41-way multi-label head."""
    roberta_config = RobertaConfig(
        vocab_size=50265, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=514, num_labels=len(CATEGORIES),
        problem_type="multi_label_classification", id2label=dict(enumerate(CATEGORIES)),
        label2id=CATEGORY_TO_INDEX,
    )
    return RobertaForSequenceClassification(roberta_config)


def make_labeled_chunks(num_chunks: int, num_positive: int, prefix: str = "c") -> list[LabeledChunk]:
    """Build num_chunks short chunks, the first num_positive of them labeled Parties."""
    chunks = []
    for i in range(num_chunks):
        text = f"This Agreement is made between the parties, clause {i}."
        labels = [0] * len(CATEGORIES)
        if i < num_positive:
            labels[CATEGORY_TO_INDEX["Parties"]] = 1
        chunk = Chunk(contract_id=f"{prefix}{i}", start=0, end=len(text), text=text, chunk_index=0)
        chunks.append(LabeledChunk(chunk=chunk, labels=labels))
    return chunks


@pytest.fixture(scope="module")
def tokenizer():
    """Load the DistilRoBERTa tokenizer once for the module (tokenizer files only, no weights)."""
    return AutoTokenizer.from_pretrained("distilbert/distilroberta-base")


# ----- build_model: arguments passed to from_pretrained -----
def test_build_model_requests_a_41_way_multi_label_head_with_category_names(monkeypatch):
    """Checks from_pretrained gets the model name, 41 labels, multi-label type and both label maps."""
    calls = []

    def fake_from_pretrained(*args, **kwargs):
        """Record the call and return a placeholder instead of downloading a model."""
        calls.append((args, kwargs))
        return "model"

    monkeypatch.setattr(AutoModelForSequenceClassification, "from_pretrained", fake_from_pretrained)

    result = build_model(make_config(model_name="some/model"))

    args, kwargs = calls[0]
    assert result == "model"
    assert args == ("some/model",)
    assert kwargs["num_labels"] == 41
    assert kwargs["problem_type"] == "multi_label_classification"
    assert kwargs["id2label"][0] == CATEGORIES[0]
    assert kwargs["id2label"][40] == CATEGORIES[40]
    assert kwargs["label2id"] == CATEGORY_TO_INDEX


# ----- build_trainer: datasets, weights and wiring -----
def test_build_trainer_returns_weighted_trainer_with_both_datasets(tmp_path, tokenizer):
    """Checks the Trainer type, dataset sizes, collator and metric function it is wired with."""
    config = make_config(output_dir=tmp_path)
    trainer = build_trainer(config, make_tiny_model(), tokenizer,
                            make_labeled_chunks(8, 2), make_labeled_chunks(4, 1, "v"))

    assert isinstance(trainer, WeightedBCETrainer)
    assert len(trainer.train_dataset) == 8
    assert len(trainer.eval_dataset) == 4
    assert isinstance(trainer.data_collator, ChunkCollator)
    assert trainer.compute_metrics is compute_transformer_metrics


def test_build_trainer_pos_weight_comes_from_the_train_labels_only(tmp_path, tokenizer):
    """Checks pos_weight is neg/pos of the train set (6/2 = 3.0 for Parties) and ignores validation."""
    config = make_config(output_dir=tmp_path)
    train_chunks = make_labeled_chunks(8, 2)
    val_all_negative = make_labeled_chunks(4, 0, "v")
    val_all_positive = make_labeled_chunks(4, 4, "v")

    first = build_trainer(config, make_tiny_model(), tokenizer, train_chunks, val_all_negative)
    second = build_trainer(config, make_tiny_model(), tokenizer, train_chunks, val_all_positive)

    parties = CATEGORY_TO_INDEX["Parties"]
    assert first.pos_weight[parties].item() == 3.0
    assert first.pos_weight.sum().item() == 3.0 + 40 * 1.0
    assert torch.equal(first.pos_weight, second.pos_weight)


def test_build_trainer_pos_weight_is_capped_by_the_config(tmp_path, tokenizer):
    """Checks a raw weight of 15 (1 positive in 16) is clipped to the configured cap of 5."""
    config = make_config(output_dir=tmp_path, pos_weight_cap=5.0)
    trainer = build_trainer(config, make_tiny_model(), tokenizer,
                            make_labeled_chunks(16, 1), make_labeled_chunks(4, 1, "v"))

    assert trainer.pos_weight[CATEGORY_TO_INDEX["Parties"]].item() == 5.0


# ----- fit: end to end on a tiny model -----
def test_fit_trains_evaluates_and_saves_the_best_checkpoint(tmp_path, tokenizer, monkeypatch):
    """Checks fit runs one epoch on a tiny model, returns macro AUPR and saves a reloadable checkpoint."""
    monkeypatch.setattr("src.models.encoder.train.build_model", lambda config: make_tiny_model())
    config = make_config(output_dir=tmp_path / "run", epochs=1, batch_size=4, grad_accum=1, max_length=32)
    chunks = make_labeled_chunks(16, 8)

    metrics = fit(config, chunks, chunks)

    best = tmp_path / "run" / "best"
    assert 0.0 <= metrics["eval_macro_aupr"] <= 1.0
    assert (best / "train_config.json").exists()
    assert (best / "tokenizer.json").exists()
    reloaded = RobertaForSequenceClassification.from_pretrained(best)
    assert reloaded.config.id2label[0] == CATEGORIES[0]
    assert reloaded.config.num_labels == 41


def test_fit_saved_train_config_round_trips_to_the_same_config(tmp_path, tokenizer, monkeypatch):
    """Checks the config written next to the model loads back equal to the one that was used."""
    monkeypatch.setattr("src.models.encoder.train.build_model", lambda config: make_tiny_model())
    config = make_config(output_dir=tmp_path / "run", epochs=1, batch_size=4, grad_accum=1, max_length=32)

    fit(config, make_labeled_chunks(8, 4), make_labeled_chunks(8, 4))

    saved = (tmp_path / "run" / "best" / "train_config.json").read_text(encoding="utf-8")
    assert TrainConfig.model_validate_json(saved) == config


def test_fit_prints_the_final_macro_aupr(tmp_path, tokenizer, monkeypatch, capsys):
    """Checks fit reports eval_macro_aupr on stdout for the notebook and Colab logs."""
    monkeypatch.setattr("src.models.encoder.train.build_model", lambda config: make_tiny_model())
    config = make_config(output_dir=tmp_path / "run", epochs=1, batch_size=4, grad_accum=1, max_length=32)

    fit(config, make_labeled_chunks(8, 4), make_labeled_chunks(8, 4))

    assert "eval_macro_aupr:" in capsys.readouterr().out


# ----- train: loads the data and delegates to fit -----
def test_train_loads_the_data_then_calls_fit_with_it(monkeypatch):
    """Checks train passes the config and both chunk lists from load_data to fit, returning its metrics."""
    config = make_config()
    calls = {}

    def fake_fit(cfg, train_chunks, val_chunks):
        """Record the arguments fit receives and return fixed metrics."""
        calls["args"] = (cfg, train_chunks, val_chunks)
        return {"eval_macro_aupr": 0.5}

    monkeypatch.setattr("src.models.encoder.train.load_data", lambda cfg: ("train-chunks", "val-chunks"))
    monkeypatch.setattr("src.models.encoder.train.fit", fake_fit)

    result = train(config)

    assert calls["args"] == (config, "train-chunks", "val-chunks")
    assert result == {"eval_macro_aupr": 0.5}


# ----- main: command-line options -----
def test_main_without_arguments_loads_the_default_config_and_trains(monkeypatch):
    """Checks no arguments means DEFAULT_CONFIG_PATH is loaded and the result is passed to train."""
    calls = {}

    def fake_load_config(path):
        """Record the path that was loaded and return a placeholder config."""
        calls["path"] = path
        return "config"

    monkeypatch.setattr("src.models.encoder.train.load_config", fake_load_config)
    monkeypatch.setattr("src.models.encoder.train.train", lambda config: calls.update(config=config))

    main([])

    assert calls["path"] == DEFAULT_CONFIG_PATH
    assert calls["config"] == "config"


def test_main_config_option_loads_the_given_path(monkeypatch):
    """Checks --config PATH is converted to a Path and loaded instead of the default."""
    calls = {}

    def fake_load_config(path):
        """Record the path that was loaded and return a placeholder config."""
        calls["path"] = path
        return "config"

    monkeypatch.setattr("src.models.encoder.train.load_config", fake_load_config)
    monkeypatch.setattr("src.models.encoder.train.train", lambda config: None)

    main(["--config", "configs/other.yaml"])

    assert calls["path"] == Path("configs/other.yaml")


@pytest.mark.parametrize("argv", [["--bogus"], ["--smoke"]])
def test_main_unknown_option_exits_with_usage_error(argv):
    """Checks unknown options (including the removed --smoke) make argparse exit with an error."""
    with pytest.raises(SystemExit):
        main(argv)


def test_default_config_path_is_the_encoder_yaml_in_the_repo():
    """Checks the default config points at configs/encoder.yaml, and that the file exists."""
    assert DEFAULT_CONFIG_PATH.name == "encoder.yaml"
    assert DEFAULT_CONFIG_PATH.parent.name == "configs"
    assert DEFAULT_CONFIG_PATH.exists()
