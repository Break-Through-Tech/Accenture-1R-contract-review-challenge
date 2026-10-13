import warnings
import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import (TrainingArguments, PreTrainedModel,
                          AutoModelForSequenceClassification,
                          PreTrainedTokenizerBase, AutoTokenizer, 
                          set_seed)
from src.models.encoder.config import TrainConfig, load_config
from src.models.encoder.dataset import ChunkDataset, ChunkCollator, build_labeled_chunks
from src.models.encoder.loss import WeightedBCETrainer, compute_pos_weight
from src.labels.loader import load_cuad
from src.labels.splits import split_by_contract
from src.labels.projection import LabeledChunk
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.metrics import compute_transformer_metrics

DEFAULT_OUTPUT_DIR = Path("outputs/encoder")
TRAIN_PATH = Path(__file__).resolve().parents[3] / "data/cuad/train_separate_questions.json"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/encoder.yaml"


def build_training_args(config: TrainConfig) -> TrainingArguments:
    """
    Map a TrainConfig onto Hugging Face TrainingArguments for the encoder run.
    Evaluates and saves every epoch and keeps the checkpoint with the best macro AUPR.
    fp16 is turned off with a warning when CUDA is unavailable, so one config runs on
    CPU and GPU. Reporting to external trackers is off until MLflow is added (Step 9).
    """
    # Mixed precision needs a GPU, so fall back to full precision without CUDA
    use_fp16 = config.fp16 and torch.cuda.is_available()
    if config.fp16 and not use_fp16:
        warnings.warn("fp16 requested but CUDA is unavailable; training in full precision")

    output_dir = config.output_dir if config.output_dir is not None else DEFAULT_OUTPUT_DIR

    return TrainingArguments(
        # Where checkpoints are written
        output_dir=str(output_dir),

        # Optimization: effective batch size = batch_size * grad_accum
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=2 * config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        warmup_steps=0.1,  # a float below 1 is a fraction of the total steps
        weight_decay=0.01,
        fp16=use_fp16,

        # Reproducibility
        seed=config.seed,
        data_seed=config.seed,

        # Evaluate and save each epoch, keeping the best checkpoint by macro AUPR
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_aupr",
        greater_is_better=True,
        save_total_limit=1,

        # Logging: console only for now, MLflow comes in Step 9
        logging_steps=50,
        report_to="none",
    )


def load_data(config: TrainConfig, train_path: Path = TRAIN_PATH) -> tuple[list[LabeledChunk], list[LabeledChunk]]:
    """
    Load the official CUAD train file and return (train, val) lists of LabeledChunk.
    Splits whole contracts with the config's seeded val_fraction, so no contract is in both,
    then chunks each side with the config's window and overlap. Only the train file is ever
    read; test.json is never touched. train_path can be overridden for tests.
    """
    # Load every contract in the official train file
    contracts = load_cuad(train_path)

    # Split by contract so no contract leaks across train and val
    train_contracts, val_contracts = split_by_contract(
        contracts, val_fraction=config.val_fraction, seed=config.seed
    )

    # Chunk and label each side with the configured window and overlap
    train_chunks = build_labeled_chunks(train_contracts, config.window_chars, config.overlap_chars)
    val_chunks = build_labeled_chunks(val_contracts, config.window_chars, config.overlap_chars)
    return train_chunks, val_chunks


def build_model(config: TrainConfig) -> PreTrainedModel:
    """
    Load the pretrained encoder with a new multi-label classification head over CATEGORIES.
    The head has one logit per category and starts untrained. The category names are stored
    in the model config, so a saved checkpoint knows its own column order. The first call
    downloads the weights.
    """
    # Map column index <-> category name so the saved model is self describing
    id2label = dict(enumerate(CATEGORIES))

    # Load the pretrained weights with a fresh 41-way multi-label head
    return AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(CATEGORIES),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=CATEGORY_TO_INDEX,
    )


def build_trainer(
    config: TrainConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_chunks: list[LabeledChunk],
    val_chunks: list[LabeledChunk],
) -> WeightedBCETrainer:
    """
    Assemble the weighted-BCE Trainer for the given model and labeled chunks.
    Tokenizes both splits, computes pos_weight from the train labels only (capped at
    config.pos_weight_cap), pads batches with ChunkCollator and selects checkpoints by
    macro AUPR on the validation set (see build_training_args).
    """
    # Tokenize the train and validation chunks
    train_dataset = ChunkDataset(train_chunks, tokenizer, config.max_length)
    val_dataset = ChunkDataset(val_chunks, tokenizer, config.max_length)

    # Per-category loss weights from the train labels only
    train_labels = np.array([lc.labels for lc in train_chunks])
    pos_weight = compute_pos_weight(train_labels, config.pos_weight_cap)

    # Wire the model, data, collator and metric into the Trainer
    return WeightedBCETrainer(
        pos_weight=pos_weight,
        model=model,
        args=build_training_args(config),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=ChunkCollator(tokenizer),
        compute_metrics=compute_transformer_metrics,
        processing_class=tokenizer,
    )


def fit(
    config: TrainConfig,
    train_chunks: list[LabeledChunk],
    val_chunks: list[LabeledChunk],
) -> dict:
    """
    Fine-tune the encoder on the given labeled chunks and return the final eval metrics.
    Seeds everything, trains, then evaluates the best epoch (by macro AUPR) on the
    validation chunks. Saves the best model, tokenizer and the config used to
    <output_dir>/best, and prints eval_macro_aupr. Takes the chunks as arguments so callers
    such as the smoke script can pass a subset.
    """
    # Seed first, since the new classification head is initialized randomly
    set_seed(config.seed)

    # Load the tokenizer and model, and assemble the trainer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = build_model(config)
    trainer = build_trainer(config, model, tokenizer, train_chunks, val_chunks)

    # Fine-tune; the best epoch's weights are reloaded when training ends
    trainer.train()

    # Evaluate the best weights on the validation set
    metrics = trainer.evaluate()

    # Save the best model with the config it was trained with
    output_dir = Path(trainer.args.output_dir) / "best"
    trainer.save_model(str(output_dir))
    (output_dir / "train_config.json").write_text(config.model_dump_json(indent=2))

    print(f"eval_macro_aupr: {metrics['eval_macro_aupr']:.4f}")
    return metrics


def train(config: TrainConfig) -> dict:
    """
    Run one full encoder training job from a TrainConfig and return the final eval metrics.
    Loads and chunks the official train data (split by contract), then calls fit.
    """
    train_chunks, val_chunks = load_data(config)
    return fit(config, train_chunks, val_chunks)


def main(argv: list[str] | None = None) -> None:
    """
    Command-line entry point: python -m src.models.encoder.train [--config PATH].
    Loads the YAML config and runs train. argv defaults to the process arguments and can
    be passed explicitly in tests.
    """
    # Define the command-line options
    parser = argparse.ArgumentParser(description="Fine-tune the DistilRoBERTa clause classifier.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to the YAML training config")
    args = parser.parse_args(argv)

    # Load and validate the config, then run the training job
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()