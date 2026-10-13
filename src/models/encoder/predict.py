import numpy as np
import torch
from pathlib import Path
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizerBase)
from src.chunking.chunk import Chunk
from src.models.encoder.config import TrainConfig


def predict_proba(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    chunks: list[Chunk],
    max_length: int = 512,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Score chunks with a trained classifier and return per-category probabilities.
    Returns a float32 array of shape (len(chunks), num_labels) where row i belongs to
    chunks[i]. Each probability is the sigmoid of the model's logit. max_length must match
    the value the model was trained with. Runs in eval mode without gradients, on whatever
    device the model is on.
    """
    # Handle empty input, since the tokenizer raises on an empty list
    if not chunks:
        return np.zeros((0, model.config.num_labels), dtype=np.float32)

    # Switch off dropout so a chunk always gets the same score
    model.eval()
    all_probs = []

    with torch.no_grad():
        for start in range(0, len(chunks), batch_size):
            # Tokenize one batch, padded to its longest chunk, and move it to the model's device
            texts = [c.text for c in chunks[start:start + batch_size]]
            enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}

            # Turn the logits into probabilities and keep them on the CPU as numpy
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs)


def load_checkpoint(
    checkpoint_dir: str | Path,
    device: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a saved classifier and its tokenizer from a checkpoint folder (e.g. <output_dir>/best).
    The model's config carries the category names, so no label arguments are needed. The model
    goes to device, or to the GPU if one is available and the CPU otherwise when device is
    None, and is returned in eval mode.
    """
    # Load the weights and tokenizer files saved together by fit
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))

    # Move the model to the requested device, choosing automatically if none was given
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Ready the model for inference
    model.eval()
    return model, tokenizer


def load_train_config(checkpoint_dir: str | Path) -> TrainConfig:
    """
    Load the TrainConfig that fit saved next to a checkpoint as train_config.json.
    Supplies the values inference must reuse: max_length for predict_proba, and
    window_chars and overlap_chars so contracts are chunked as they were in training.
    Raises FileNotFoundError if the file is missing and pydantic.ValidationError if it is invalid.
    """
    # Locate the config file saved in the checkpoint folder
    path = Path(checkpoint_dir) / "train_config.json"

    # Parse and validate it back into a TrainConfig
    return TrainConfig.model_validate_json(path.read_text(encoding="utf-8"))