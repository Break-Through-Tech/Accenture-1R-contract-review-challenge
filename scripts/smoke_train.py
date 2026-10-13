"""
Smoke test for the encoder training pipeline, run before a full training job.
Overfits a tiny subset of the train chunks (and evaluates on those same chunks), so a
macro AUPR near 1.0 shows that labels, logits, collator and loss weights are aligned.
It is a development check, not part of the production pipeline.

Run from the repo root: python -m scripts.smoke_train [--config PATH]
"""
import argparse
import random
import sys
from pathlib import Path
from src.labels.projection import LabeledChunk
from src.models.encoder.config import TrainConfig, load_config
from src.models.encoder.train import DEFAULT_CONFIG_PATH, fit, load_data

CHUNKS_PER_GROUP = 32
PASS_THRESHOLD = 0.9
SMOKE_OUTPUT_DIR = Path("outputs/smoke")


def make_smoke_subset(chunks: list[LabeledChunk], per_group: int, seed: int) -> list[LabeledChunk]:
    """
    Return a seeded random sample of up to per_group labeled and per_group unlabeled chunks.
    A chunk is labeled if it has at least one category. Sampling (not taking the first
    chunks) spreads the subset across contracts and categories. Labeled chunks come first.
    """
    # Split into chunks that have at least one label and chunks that have none
    labeled = [c for c in chunks if any(c.labels)]
    unlabeled = [c for c in chunks if not any(c.labels)]

    # Sample each group with the same seeded generator so the subset is reproducible
    rng = random.Random(seed)
    sampled_labeled = rng.sample(labeled, min(per_group, len(labeled)))
    sampled_unlabeled = rng.sample(unlabeled, min(per_group, len(unlabeled)))
    return sampled_labeled + sampled_unlabeled


def make_smoke_config(config: TrainConfig) -> TrainConfig:
    """
    Return a copy of config tuned to overfit a tiny set quickly on a laptop CPU.
    Higher learning rate, more epochs, short sequences (max_length 128), a small batch,
    no fp16 and a throwaway output folder, so a real run's outputs are never overwritten.
    Long sequences with a big batch can run out of memory on a CPU. Truncating is fine
    here because identical inputs still map to identical labels. The original is unchanged.
    """
    return config.model_copy(update={
        "lr": 1.0e-4,
        "epochs": 10,
        "max_length": 128,
        "batch_size": 8,
        "grad_accum": 1,
        "fp16": False,
        "output_dir": SMOKE_OUTPUT_DIR,
    })


def main(argv: list[str] | None = None) -> int:
    """
    Run the smoke test and return a process exit code: 0 if macro AUPR on the overfit
    subset reaches PASS_THRESHOLD, 1 otherwise.
    """
    # Read the options and build the smoke configuration
    parser = argparse.ArgumentParser(description="Smoke test the encoder training pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to the YAML training config to derive the smoke run from")
    args = parser.parse_args(argv)
    config = make_smoke_config(load_config(args.config))

    # Overfit a small subset of the train chunks, evaluating on the same chunks
    train_chunks, _ = load_data(config)
    subset = make_smoke_subset(train_chunks, CHUNKS_PER_GROUP, config.seed)
    print(f"Smoke subset: {len(subset)} chunks")
    metrics = fit(config, subset, subset)

    # Pass only if the model memorized the subset
    macro_aupr = metrics["eval_macro_aupr"]
    passed = macro_aupr >= PASS_THRESHOLD
    print(f"SMOKE {'PASS' if passed else 'FAIL'}: eval_macro_aupr={macro_aupr:.4f} "
          f"(threshold {PASS_THRESHOLD})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
