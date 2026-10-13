import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.labels.categories import CATEGORIES
from src.labels.projection import LabeledChunk
from src.metrics import compute_aupr, compute_base_metrics
from src.models.encoder.predict import load_checkpoint, load_train_config, predict_proba
from src.models.encoder.train import TRAIN_PATH, load_data
from src.thresholds import apply_thresholds, find_best_thresholds

DEFAULT_THRESHOLD = 0.5
MIN_VAL_POSITIVES = 5


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, categories: list[str]) -> dict[str, float]:
    """
    Find the F1-optimal probability threshold for each category on (y_true, y_prob).
    A category with no positives cannot be tuned and gets DEFAULT_THRESHOLD. Returns a
    mapping from category name to threshold, in the order of categories.
    """
    # Tune only the categories that have at least one positive
    has_positives = y_true.sum(axis=0) > 0
    tunable = [c for c, tunable_flag in zip(categories, has_positives) if tunable_flag]
    tuned = find_best_thresholds(y_true[:, has_positives], y_prob[:, has_positives], tunable)

    # Everything else falls back to the default threshold
    return {c: float(tuned.get(c, DEFAULT_THRESHOLD)) for c in categories}


def build_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict[str, float],
    categories: list[str],
) -> pd.DataFrame:
    """
    Build the per-category evaluation table, sorted by support (most positives first).
    Columns: category, precision, recall, f1 (at the given thresholds), support, aupr,
    threshold and low_support (fewer than MIN_VAL_POSITIVES positives). Metrics of a
    category with no positives are NaN, not 0, because they are undefined.
    """
    # Precision, recall, F1 and support at the tuned thresholds
    y_pred = apply_thresholds(y_prob, categories, thresholds)
    report = compute_base_metrics(y_true, y_pred, categories)

    # Threshold-free AUPR and the threshold used, matched to each category
    report["aupr"] = report["category"].map(pd.Series(compute_aupr(y_true, y_prob), index=categories))
    report["threshold"] = report["category"].map(thresholds)

    # Undefined metrics are NaN, and thin categories are flagged
    report.loc[report["support"] == 0, ["precision", "recall", "f1"]] = np.nan
    report["low_support"] = report["support"] < MIN_VAL_POSITIVES
    return report


def summarize(report: pd.DataFrame) -> dict:
    """
    Reduce the per-category report to macro averages and counts, ready for JSON.
    Macro averages skip categories with no positives, and are None if none can be scored.
    """
    def macro(column: str) -> float | None:
        """Mean of a report column ignoring NaN, or None when every value is NaN."""
        value = report[column].mean()
        return None if np.isnan(value) else float(value)

    return {
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_f1": macro("f1"),
        "macro_aupr": macro("aupr"),
        "num_categories": int(len(report)),
        "num_unscored": int((report["support"] == 0).sum()),
        "low_support_categories": report.loc[report["low_support"], "category"].tolist(),
    }


def save_evaluation(
    checkpoint_dir: str | Path,
    val_chunks: list[LabeledChunk],
    probs: np.ndarray,
    thresholds: dict[str, float],
    report: pd.DataFrame,
    summary: dict,
) -> Path:
    """
    Write the evaluation outputs and return the eval folder.
    thresholds.json goes next to the model so the checkpoint is self-contained for inference.
    <checkpoint_dir>/eval gets val_probs.npy (raw probabilities, row i is row i of
    val_chunks.csv), val_chunks.csv (contract_id, chunk_index, start, end), metrics.csv and
    summary.json. Keeping the raw probabilities means thresholds can be retuned without a rerun.
    """
    checkpoint_dir = Path(checkpoint_dir)
    eval_dir = checkpoint_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # The thresholds belong to the model artifact
    (checkpoint_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    # Raw probabilities and the chunks they belong to, row for row
    np.save(eval_dir / "val_probs.npy", probs)
    pd.DataFrame([
        {"contract_id": lc.chunk.contract_id, "chunk_index": lc.chunk.chunk_index,
         "start": lc.chunk.start, "end": lc.chunk.end}
        for lc in val_chunks
    ]).to_csv(eval_dir / "val_chunks.csv", index=False)

    # The per-category table and the macro summary
    report.to_csv(eval_dir / "metrics.csv", index=False)
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return eval_dir


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    train_path: Path = TRAIN_PATH,
    batch_size: int = 32,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate a trained checkpoint on its validation split and save the results.
    Reads the training config saved with the checkpoint, so the validation split (same seed
    and fraction) and the chunking are exactly those used in training, then scores the
    validation chunks, tunes per-category thresholds on them and reports per-category
    precision, recall, F1 and AUPR. These metrics use thresholds tuned on the same data,
    so they are optimistic; the final numbers come from test.json. Returns the report.
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Rebuild the same validation chunks the model was trained against
    config = load_train_config(checkpoint_dir)
    _, val_chunks = load_data(config, train_path)

    # Score them with the trained model, with the same max_length as in training
    model, tokenizer = load_checkpoint(checkpoint_dir, device)
    probs = predict_proba(model, tokenizer, [lc.chunk for lc in val_chunks],
                          max_length=config.max_length, batch_size=batch_size)

    # Tune thresholds and build the per-category report
    y_true = np.array([lc.labels for lc in val_chunks])
    thresholds = tune_thresholds(y_true, probs, CATEGORIES)
    report = build_report(y_true, probs, thresholds, CATEGORIES)
    summary = summarize(report)

    # Save everything and print the headline numbers
    eval_dir = save_evaluation(checkpoint_dir, val_chunks, probs, thresholds, report, summary)
    print(f"macro_f1: {summary['macro_f1']:.4f}  macro_aupr: {summary['macro_aupr']:.4f}  "
          f"unscored: {summary['num_unscored']}  low support: {len(summary['low_support_categories'])}")
    print(f"Saved evaluation to {eval_dir}")
    return report


def main(argv: list[str] | None = None) -> None:
    """
    Command-line entry point: python -m src.models.encoder.evaluate --checkpoint PATH.
    Evaluates the checkpoint on its validation split. argv defaults to the process arguments.
    """
    # Define the command-line options
    parser = argparse.ArgumentParser(description="Evaluate a trained clause classifier on its validation split.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Checkpoint folder, e.g. outputs/encoder/best")
    parser.add_argument("--batch-size", type=int, default=32, help="Scoring batch size")
    args = parser.parse_args(argv)

    # Evaluate and save the results
    evaluate_checkpoint(args.checkpoint, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
