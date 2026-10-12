from pathlib import Path
import pytest
import yaml
from pydantic import ValidationError
from src.models.encoder.config import TrainConfig, load_config

REAL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "encoder.yaml"


def valid_config_dict(**overrides) -> dict:
    """Build a complete, valid config mapping, with any given fields overridden."""
    config = {
        "model_name": "distilbert/distilroberta-base",
        "max_length": 512,
        "window_chars": 1500,
        "overlap_chars": 200,
        "val_fraction": 0.2,
        "seed": 42,
        "lr": 3.0e-5,
        "epochs": 4,
        "batch_size": 16,
        "grad_accum": 2,
        "fp16": True,
        "pos_weight_cap": 50.0,
        "mlflow_experiment": "test-experiment",
    }
    config.update(overrides)
    return config


def write_yaml(path: Path, data) -> Path:
    """Write data to a YAML file at path and return the path."""
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ----- TrainConfig: valid construction -----
def test_train_config_valid_values_construct_with_expected_fields():
    """Checks a complete valid mapping builds a config holding the given values."""
    config = TrainConfig(**valid_config_dict())

    assert config.window_chars == 1500
    assert config.overlap_chars == 200
    assert config.lr == 3.0e-5
    assert config.fp16 is True


def test_train_config_output_dir_omitted_defaults_to_none():
    """Checks output_dir is optional and defaults to None."""
    assert TrainConfig(**valid_config_dict()).output_dir is None


def test_train_config_output_dir_string_is_converted_to_path():
    """Checks a string output_dir is coerced to a Path."""
    config = TrainConfig(**valid_config_dict(output_dir="outputs/encoder"))

    assert config.output_dir == Path("outputs/encoder")


# ----- TrainConfig: bounds and cross-field rules -----
@pytest.mark.parametrize("overrides", [
    {"window_chars": 100, "overlap_chars": 100},   # overlap equal to window
    {"window_chars": 100, "overlap_chars": 150},   # overlap larger than window
    {"window_chars": 0},                           # empty window
    {"overlap_chars": -1},                         # negative overlap
    {"val_fraction": 0.0},                         # empty validation split
    {"val_fraction": 1.0},                         # empty train split
    {"max_length": 0},                             # no tokens
    {"max_length": 513},                           # above the 512-token model limit
    {"lr": 0.0},                                   # no learning
    {"epochs": 0},                                 # no training
    {"batch_size": 0},                             # empty batch
    {"grad_accum": 0},                             # would divide by zero
    {"pos_weight_cap": 0.5},                       # would down-weight positives
])
def test_train_config_out_of_range_value_raises_validation_error(overrides):
    """Checks each out-of-range field value or bad window/overlap pair is rejected."""
    with pytest.raises(ValidationError):
        TrainConfig(**valid_config_dict(**overrides))


@pytest.mark.parametrize("overrides", [
    {"max_length": 512},                           # exactly the model limit
    {"overlap_chars": 0},                          # no overlap
    {"window_chars": 100, "overlap_chars": 99},    # largest overlap that still advances
    {"grad_accum": 1},                             # no accumulation
    {"pos_weight_cap": 1.0},                       # smallest allowed cap
])
def test_train_config_boundary_value_is_accepted(overrides):
    """Checks values exactly on an allowed boundary still validate."""
    TrainConfig(**valid_config_dict(**overrides))


def test_train_config_overlap_not_less_than_window_error_names_both_values():
    """Checks the overlap error message reports the offending sizes."""
    with pytest.raises(ValidationError, match=r"overlap_chars \(100\).*window_chars \(100\)"):
        TrainConfig(**valid_config_dict(window_chars=100, overlap_chars=100))


# ----- TrainConfig: strictness and immutability -----
def test_train_config_unknown_key_raises_validation_error():
    """Checks a misspelled or extra key is rejected rather than silently ignored."""
    with pytest.raises(ValidationError):
        TrainConfig(**valid_config_dict(learning_rat=3.0e-5))


def test_train_config_missing_required_field_raises_validation_error():
    """Checks omitting a required field fails instead of using a hidden default."""
    data = valid_config_dict()
    del data["seed"]

    with pytest.raises(ValidationError):
        TrainConfig(**data)


def test_train_config_assigning_to_a_field_raises_validation_error():
    """Checks the config is frozen, so the logged config is what actually ran."""
    config = TrainConfig(**valid_config_dict())

    with pytest.raises(ValidationError):
        config.seed = 7


# ----- load_config: YAML loading -----
def test_load_config_real_encoder_yaml_validates():
    """Checks configs/encoder.yaml loads, so the file and TrainConfig cannot drift apart."""
    config = load_config(REAL_CONFIG_PATH)

    assert config.model_name == "distilbert/distilroberta-base"
    assert config.overlap_chars < config.window_chars


def test_load_config_valid_yaml_file_returns_matching_config(tmp_path):
    """Checks load_config returns a TrainConfig equal to one built from the same mapping."""
    path = write_yaml(tmp_path / "config.yaml", valid_config_dict())

    assert load_config(path) == TrainConfig(**valid_config_dict())


def test_load_config_accepts_str_path(tmp_path):
    """Checks load_config accepts a plain string path as well as a Path."""
    path = write_yaml(tmp_path / "config.yaml", valid_config_dict())

    assert load_config(str(path)).seed == 42


def test_load_config_invalid_value_in_yaml_raises_validation_error(tmp_path):
    """Checks bounds are enforced on values that come from a YAML file."""
    path = write_yaml(tmp_path / "config.yaml", valid_config_dict(val_fraction=1.5))

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_scientific_notation_without_dot_is_coerced_to_float(tmp_path):
    """Checks YAML 3e-5 (a string to PyYAML) is coerced by Pydantic to the float 3e-05."""
    path = tmp_path / "config.yaml"
    data = valid_config_dict()
    del data["lr"]
    path.write_text(yaml.safe_dump(data) + "lr: 3e-5\n", encoding="utf-8")

    assert load_config(path).lr == 3e-5


@pytest.mark.parametrize("content", [
    "",                       # empty file -> None
    "- a\n- b\n",             # top level is a list
    "just a string\n",        # top level is a scalar
])
def test_load_config_non_mapping_top_level_raises_value_error(tmp_path, content):
    """Checks empty files and non-mapping YAML raise a clear ValueError."""
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_load_config_missing_file_raises_file_not_found(tmp_path):
    """Checks a nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")
