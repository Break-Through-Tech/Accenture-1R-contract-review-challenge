from pathlib import Path
import numpy as np
import pytest
import torch
from pydantic import ValidationError
from transformers import AutoTokenizer, RobertaConfig, RobertaForSequenceClassification
from src.chunking.chunk import Chunk
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.models.encoder.config import TrainConfig
from src.models.encoder.predict import load_checkpoint, load_train_config, predict_proba

TEXTS = [
    "Short.",
    "A somewhat longer sentence for padding purposes here.",
    "Mid length text.",
    "Another clause that is quite a bit longer than most of the others in this small list.",
    "Fifth.",
]


def make_tiny_model() -> RobertaForSequenceClassification:
    """
    Build a tiny randomly initialized RoBERTa classifier (no download) with a 41-way multi-label head.
    The large initializer_range makes the scores react strongly to the input text, so tests about
    truncation, padding and batching can fail if the code is wrong (with the default small weights
    every input would score almost the same).
    """
    roberta_config = RobertaConfig(
        vocab_size=50265, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=514, num_labels=len(CATEGORIES),
        problem_type="multi_label_classification", id2label=dict(enumerate(CATEGORIES)),
        label2id=CATEGORY_TO_INDEX, initializer_range=0.5,
    )
    return RobertaForSequenceClassification(roberta_config)


def make_chunks(texts: list[str]) -> list[Chunk]:
    """Build one single-chunk contract per text, in order."""
    return [Chunk(contract_id=f"c{i}", start=0, end=len(t), text=t, chunk_index=0) for i, t in enumerate(texts)]


def make_config(**overrides) -> TrainConfig:
    """Build a valid TrainConfig with distinctive inference values, with any fields overridden."""
    values = {
        "model_name": "distilbert/distilroberta-base",
        "max_length": 384,
        "window_chars": 800,
        "overlap_chars": 200,
        "val_fraction": 0.2,
        "seed": 42,
        "lr": 3.0e-5,
        "epochs": 4,
        "batch_size": 16,
        "grad_accum": 2,
        "fp16": False,
        "pos_weight_cap": 50.0,
        "output_dir": "outputs/encoder",
        "mlflow_experiment": "test-experiment",
    }
    values.update(overrides)
    return TrainConfig(**values)


@pytest.fixture(scope="module")
def tokenizer():
    """Load the DistilRoBERTa tokenizer once for the module (tokenizer files only, no weights)."""
    return AutoTokenizer.from_pretrained("distilbert/distilroberta-base")


@pytest.fixture(scope="module")
def model():
    """Build one tiny model for the tests that only read from it."""
    return make_tiny_model()


@pytest.fixture
def saved_checkpoint(tmp_path, tokenizer, model):
    """Save the tiny model and tokenizer into a fresh folder and return its path."""
    folder = tmp_path / "best"
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder


# ----- predict_proba: output shape and values -----
def test_predict_proba_returns_float32_probabilities_one_row_per_chunk(model, tokenizer):
    """Checks shape (num_chunks, 41), dtype float32 and every value between 0 and 1."""
    probs = predict_proba(model, tokenizer, make_chunks(TEXTS), max_length=32, batch_size=2)

    assert probs.shape == (len(TEXTS), len(CATEGORIES))
    assert probs.dtype == np.float32
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_predict_proba_empty_list_gives_empty_array_with_one_column_per_category(model, tokenizer):
    """Checks no chunks yields a (0, 41) float32 array instead of raising."""
    probs = predict_proba(model, tokenizer, [], max_length=32)

    assert probs.shape == (0, len(CATEGORIES))
    assert probs.dtype == np.float32


def test_predict_proba_matches_sigmoid_of_the_model_logits(model, tokenizer):
    """Checks each probability is exactly sigmoid(logit) for a chunk scored directly through the model."""
    enc = tokenizer([TEXTS[1]], truncation=True, max_length=32, padding=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        expected = torch.sigmoid(model(**enc).logits).numpy()

    probs = predict_proba(model, tokenizer, make_chunks([TEXTS[1]]), max_length=32)

    assert np.allclose(probs, expected, atol=1e-6)


# ----- predict_proba: ordering and batching -----
def test_predict_proba_row_order_follows_chunk_order(model, tokenizer):
    """Checks row i of a batched call equals the score of chunk i scored alone."""
    chunks = make_chunks(TEXTS)

    together = predict_proba(model, tokenizer, chunks, max_length=32, batch_size=3)
    alone = np.concatenate([predict_proba(model, tokenizer, [c], max_length=32) for c in chunks])

    assert np.allclose(together, alone, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2, 5, 100])
def test_predict_proba_result_does_not_depend_on_batch_size(model, tokenizer, batch_size):
    """Checks batching and padding do not change the scores, including a batch larger than the input."""
    chunks = make_chunks(TEXTS)

    reference = predict_proba(model, tokenizer, chunks, max_length=32, batch_size=1)
    batched = predict_proba(model, tokenizer, chunks, max_length=32, batch_size=batch_size)

    assert np.allclose(reference, batched, atol=1e-5)


def test_predict_proba_different_chunks_get_different_scores(model, tokenizer):
    """Checks scores depend on the chunk text, so the rows are not all the same."""
    probs = predict_proba(model, tokenizer, make_chunks(TEXTS[:2]), max_length=32)

    assert not np.allclose(probs[0], probs[1])


# ----- predict_proba: truncation -----
def test_predict_proba_text_past_max_length_does_not_affect_the_score(model, tokenizer):
    """Checks two chunks that differ only after max_length tokens get identical scores."""
    prefix = " ".join(["alpha beta gamma delta"] * 6)
    chunks = make_chunks([prefix + " ending one", prefix + " a completely different ending"])

    probs = predict_proba(model, tokenizer, chunks, max_length=8)

    assert np.allclose(probs[0], probs[1], atol=1e-6)


def test_predict_proba_very_long_chunk_is_truncated_without_error(model, tokenizer):
    """Checks a chunk far over the model limit is scored instead of raising an indexing error."""
    probs = predict_proba(model, tokenizer, make_chunks(["word " * 3000]), max_length=512)

    assert probs.shape == (1, len(CATEGORIES))


# ----- predict_proba: inference mode -----
def test_predict_proba_leaves_the_model_in_eval_mode():
    """Checks the model is switched to eval mode even when it starts in train mode."""
    train_mode_model = make_tiny_model().train()

    predict_proba(train_mode_model, AutoTokenizer.from_pretrained("distilbert/distilroberta-base"),
                  make_chunks(TEXTS[:1]), max_length=32)

    assert train_mode_model.training is False


def test_predict_proba_is_deterministic_because_dropout_is_off(tokenizer):
    """Checks scoring twice gives identical results even for a model that starts with dropout active."""
    train_mode_model = make_tiny_model().train()
    chunks = make_chunks(TEXTS)

    first = predict_proba(train_mode_model, tokenizer, chunks, max_length=32)
    second = predict_proba(train_mode_model, tokenizer, chunks, max_length=32)

    assert np.array_equal(first, second)


def test_predict_proba_computes_no_gradients(tokenizer):
    """Checks scoring leaves no gradients on the model parameters."""
    fresh_model = make_tiny_model()

    predict_proba(fresh_model, tokenizer, make_chunks(TEXTS), max_length=32)

    assert all(p.grad is None for p in fresh_model.parameters())


# ----- load_checkpoint -----
@pytest.mark.parametrize("as_type", [str, Path])
def test_load_checkpoint_gives_the_same_predictions_as_the_saved_model(
        saved_checkpoint, model, tokenizer, as_type):
    """Checks a saved model reloads (from a str or a Path) and scores chunks identically."""
    loaded_model, loaded_tokenizer = load_checkpoint(as_type(saved_checkpoint), device="cpu")
    chunks = make_chunks(TEXTS)

    original = predict_proba(model, tokenizer, chunks, max_length=32)
    reloaded = predict_proba(loaded_model, loaded_tokenizer, chunks, max_length=32)

    assert np.allclose(original, reloaded, atol=1e-6)


def test_load_checkpoint_returns_an_eval_mode_model_on_the_requested_device(saved_checkpoint):
    """Checks the loaded model is in eval mode and on the CPU when device='cpu'."""
    loaded_model, _ = load_checkpoint(saved_checkpoint, device="cpu")

    assert loaded_model.training is False
    assert loaded_model.device.type == "cpu"


def test_load_checkpoint_keeps_the_category_names_and_label_count(saved_checkpoint):
    """Checks the 41 category names are stored in the checkpoint, so no label arguments are needed."""
    loaded_model, _ = load_checkpoint(saved_checkpoint, device="cpu")

    assert loaded_model.config.num_labels == 41
    assert loaded_model.config.id2label[0] == CATEGORIES[0]
    assert loaded_model.config.id2label[40] == CATEGORIES[40]


def test_load_checkpoint_returns_a_working_tokenizer(saved_checkpoint, tokenizer):
    """Checks the loaded tokenizer encodes text exactly like the original one."""
    _, loaded_tokenizer = load_checkpoint(saved_checkpoint, device="cpu")

    assert loaded_tokenizer(TEXTS[1])["input_ids"] == tokenizer(TEXTS[1])["input_ids"]


def test_load_checkpoint_explicit_device_wins_even_when_cuda_is_available(saved_checkpoint, monkeypatch):
    """Checks device='cpu' is honored, not overridden by automatic GPU selection."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    loaded_model, _ = load_checkpoint(saved_checkpoint, device="cpu")

    assert loaded_model.device.type == "cpu"


def test_load_checkpoint_without_device_uses_the_cpu_when_no_gpu_is_available(saved_checkpoint, monkeypatch):
    """Checks automatic device selection falls back to the CPU when CUDA is unavailable."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    loaded_model, _ = load_checkpoint(saved_checkpoint)

    assert loaded_model.device.type == "cpu"


def test_load_checkpoint_missing_folder_raises_os_error(tmp_path):
    """Checks a nonexistent checkpoint folder raises OSError."""
    with pytest.raises(OSError):
        load_checkpoint(tmp_path / "does_not_exist", device="cpu")


# ----- load_train_config -----
@pytest.mark.parametrize("as_type", [str, Path])
def test_load_train_config_round_trips_the_saved_config(tmp_path, as_type):
    """Checks a config written the way fit writes it loads back equal, from a str or a Path."""
    config = make_config()
    (tmp_path / "train_config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")

    assert load_train_config(as_type(tmp_path)) == config


def test_load_train_config_supplies_the_values_inference_must_reuse(tmp_path):
    """Checks max_length, window_chars and overlap_chars come through unchanged."""
    config = make_config(max_length=384, window_chars=640, overlap_chars=64)
    (tmp_path / "train_config.json").write_text(config.model_dump_json(), encoding="utf-8")

    loaded = load_train_config(tmp_path)

    assert (loaded.max_length, loaded.window_chars, loaded.overlap_chars) == (384, 640, 64)


def test_load_train_config_missing_file_raises_file_not_found(tmp_path):
    """Checks a checkpoint folder without train_config.json raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_train_config(tmp_path)


def test_load_train_config_invalid_value_raises_validation_error(tmp_path):
    """Checks bounds are enforced on a config read from disk (val_fraction must be below 1)."""
    broken = make_config().model_dump_json().replace('"val_fraction":0.2', '"val_fraction":2.0')
    (tmp_path / "train_config.json").write_text(broken, encoding="utf-8")

    with pytest.raises(ValidationError):
        load_train_config(tmp_path)


def test_load_train_config_unknown_key_raises_validation_error(tmp_path):
    """Checks a file with an extra field is rejected, matching the strict TrainConfig."""
    with_extra = make_config().model_dump_json().rstrip("}") + ',"learning_rat":0.1}'
    (tmp_path / "train_config.json").write_text(with_extra, encoding="utf-8")

    with pytest.raises(ValidationError):
        load_train_config(tmp_path)
