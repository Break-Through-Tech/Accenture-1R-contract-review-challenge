from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self
from pathlib import Path
import yaml


class TrainConfig(BaseModel):
    """
    All inputs for one encoder training run, loaded from configs/encoder.yaml.
    Frozen after creation so the config logged to MLflow is exactly what ran, and
    unknown keys are rejected so a typo in the YAML fails loudly. Chunking sizes are
    in characters; max_length is in tokens (RoBERTa-style models cap at 512).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Model and chunking: which encoder to fine-tune and how contracts are chunked
    model_name: str
    max_length: int = Field(ge=1, le=512)
    window_chars: int = Field(gt=0)
    overlap_chars: int = Field(ge=0)

    # Split and optimization: grouped val split, learning rate, batch size and precision
    val_fraction: float = Field(gt=0, lt=1)
    seed: int
    lr: float = Field(gt=0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(ge=1)
    grad_accum: int = Field(ge=1)  # effective batch size = batch_size * grad_accum
    fp16: bool  # mixed precision; GPU only
    pos_weight_cap: float = Field(ge=1)  # upper limit on neg/pos per-category loss weight

    # Outputs: where artifacts are written and which MLflow experiment logs the run
    output_dir: Path | None = None
    mlflow_experiment: str

    # Model Validator; ensures the overlap is smaller than the window so chunking advances
    @model_validator(mode="after")
    def check_overlap_consistency(self) -> Self:
        if self.overlap_chars >= self.window_chars:
            raise ValueError(
                f"overlap_chars ({self.overlap_chars}) must be less than "
                f"window_chars ({self.window_chars})"
            )
        return self


def load_config(path: str | Path) -> TrainConfig:
    """
    Load and validate a TrainConfig from a YAML file.
    Raises ValueError if the file is empty or its top level is not a mapping, and
    pydantic.ValidationError on a missing, unknown or out-of-range field.
    """
    # Read the whole YAML file as UTF-8 into a plain Python object
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Reject empty files and non-mapping top levels (safe_load gives None or a list)
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {str(path)!r} must contain a mapping of field names to values, "
            f"got {type(data).__name__}"
        )

    # Unpack the keys as keyword arguments so Pydantic validates every field
    return TrainConfig(**data)