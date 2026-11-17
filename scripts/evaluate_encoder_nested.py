"""Train and evaluate DistilRoBERTa on the same nested CUAD splits as the saved TF-IDF runs.

For each predeclared seed (42, 7, 17, 23, 101) the contracts in train_separate_questions.json
are split exactly as scripts/evaluate_word_char_tfidf.py split them: 20% evaluation, then 20%
of the rest for threshold calibration. Before any training the split is checked against the
saved TF-IDF report for that seed (contract-ID digests, counts, data hash, chunk counts and the
calibration/evaluation label matrices), and the run stops on any mismatch. test.json is never
read.

Each seed writes its own directory, results/encoder_nested_seed<seed>/, in the layout the other
experiments use. The run is resumable: a finished seed is skipped, and an interrupted seed
continues from its last saved epoch. Checkpoints go to a separate work directory and are not
part of the results.

Run from the repo root:
    python -m scripts.evaluate_encoder_nested                 # all five seeds
    python -m scripts.evaluate_encoder_nested --dry-run       # verify splits only, no training
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_reranker_multiseed import _macro_aupr, _macro_f1, _threshold_vector
from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.labels.categories import CATEGORIES
from src.labels.loader import Contract, load_cuad
from src.labels.projection import LabeledChunk
from src.labels.splits import split_by_contract
from src.models.encoder.config import TrainConfig, load_config
from src.models.encoder.evaluate import build_report, tune_thresholds
from src.models.encoder.experiment_utils import digest_contract_ids
from src.models.encoder.hub import HubOptions, export_hub_options, load_pinned_tokenizer, resolve_hub_options
from src.pipeline import prepare_contracts, to_model_inputs

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/cuad/train_separate_questions.json"
SEEDS = (42, 7, 17, 23, 101)
EVALUATION_FRACTION = 0.2
CALIBRATION_FRACTION = 0.2
PARTITIONS = ("training", "calibration", "evaluation")
RESULT_PREFIX = "encoder_nested_seed"
BASELINE_MODEL = "word_char_tfidf"
EXPERIMENT_NAME = "distilroberta_nested_holdout"
MODEL_NAME = "distilroberta_multilabel"

# A seed is complete only when every one of these exists and report.json says so
REQUIRED_RESULT_FILES = (
    "report.json", "per_category_metrics.csv", "thresholds.json",
    "calibration_predictions.npz", "evaluation_predictions.npz",
    "calibration_chunks.csv", "evaluation_chunks.csv",
)

# fit(config, train_chunks, val_chunks, resume_from_checkpoint) and
# score(checkpoint_dir, {name: chunks}, max_length, batch_size) -> {name: probabilities}
FitFn = Callable[..., dict]
ScoreFn = Callable[[Path, dict[str, list[LabeledChunk]], int, int], dict[str, np.ndarray]]


class SplitMismatchError(RuntimeError):
    """Raised when a seed's split or chunks differ from the saved TF-IDF baseline run."""


# ----- paths and completion state -----
def results_dir_for(results_root: Path, seed: int) -> Path:
    """Return the results directory of one seed: <results_root>/encoder_nested_seed<seed>."""
    return Path(results_root) / f"{RESULT_PREFIX}{seed}"


def baseline_dir_for(baseline_root: Path, seed: int) -> Path:
    """
    Return the saved word/word+char TF-IDF run for a seed. Seed 42 was the primary run and
    lives in word_char_tfidf_nested/; the others in word_char_tfidf_nested_seed<seed>/.
    """
    name = "word_char_tfidf_nested" if seed == 42 else f"word_char_tfidf_nested_seed{seed}"
    return Path(baseline_root) / name


def config_fingerprint(config: TrainConfig) -> str:
    """
    SHA-256 of the training settings that affect results. output_dir is excluded, so moving
    the work directory (for example to Google Drive) does not invalidate finished seeds.
    """
    payload = json.dumps(config.model_dump(mode="json", exclude={"output_dir"}), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_seed_complete(result_dir: Path) -> bool:
    """True when every required result file exists and report.json has status "complete"."""
    result_dir = Path(result_dir)
    if not all((result_dir / name).is_file() for name in REQUIRED_RESULT_FILES):
        return False
    try:
        report = json.loads((result_dir / "report.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return report.get("status") == "complete"


def latest_checkpoint(train_dir: Path) -> Path | None:
    """Return the checkpoint-N folder with the highest N under train_dir, or None if there is none."""
    train_dir = Path(train_dir)
    if not train_dir.is_dir():
        return None
    steps = [
        (int(match.group(1)), path) for path in train_dir.iterdir()
        if path.is_dir() and (match := re.fullmatch(r"checkpoint-(\d+)", path.name))
    ]
    return max(steps)[1] if steps else None


def prune_checkpoints(train_dir: Path) -> list[Path]:
    """Delete the intermediate checkpoint-N folders under train_dir (best/ is kept); return them."""
    removed = []
    for path in sorted(Path(train_dir).glob("checkpoint-*")):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path)
    return removed


# ----- split construction and verification -----
def split_contracts(contracts: list[Contract], seed: int) -> dict[str, list[Contract]]:
    """
    Split contracts into training/calibration/evaluation exactly as the TF-IDF runs did:
    evaluation first (EVALUATION_FRACTION), then calibration out of the rest. Raises
    SplitMismatchError if any contract lands in two partitions.
    """
    development, evaluation = split_by_contract(contracts, val_fraction=EVALUATION_FRACTION, seed=seed)
    training, calibration = split_by_contract(development, val_fraction=CALIBRATION_FRACTION, seed=seed)
    groups = {"training": training, "calibration": calibration, "evaluation": evaluation}

    # No contract may appear in two partitions
    ids = [{contract.contract_id for contract in group} for group in groups.values()]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise SplitMismatchError(f"Seed {seed}: contract split leakage detected")
    return groups


def split_provenance(groups: dict[str, list[Contract]], seed: int) -> dict:
    """Describe a split the way the TF-IDF report.json does: seed, fractions, counts and ID digests."""
    return {
        "method": "split_by_contract",
        "seed": seed,
        "evaluation_fraction": EVALUATION_FRACTION,
        "calibration_fraction_of_non_evaluation": CALIBRATION_FRACTION,
        "contract_counts": {name: len(groups[name]) for name in PARTITIONS},
        "contract_id_sha256": {
            name: digest_contract_ids([contract.contract_id for contract in groups[name]])
            for name in PARTITIONS
        },
    }


def load_baseline_report(baseline_dir: Path) -> dict:
    """Read a saved TF-IDF report.json, raising SplitMismatchError if the run is missing."""
    path = Path(baseline_dir) / "report.json"
    if not path.is_file():
        raise SplitMismatchError(f"No saved TF-IDF baseline report at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_split_against_baseline(provenance: dict, data_sha256: str, baseline_report: dict) -> None:
    """
    Compare a freshly computed split with a saved TF-IDF report and raise SplitMismatchError
    listing every difference (training data hash, test-data flag, seed, fractions, contract
    counts, contract-ID digests). Called before any chunking or training.
    """
    baseline_split = baseline_report.get("split", {})
    problems = []

    # Same training file and the baseline never touched test.json
    if baseline_report.get("data_sha256") != data_sha256:
        problems.append(f"data_sha256 {data_sha256} != baseline {baseline_report.get('data_sha256')}")
    if baseline_report.get("test_data_used") is not False:
        problems.append("baseline report does not state test_data_used: false")

    # Same split recipe, sizes and contract IDs in every partition
    for key in ("seed", "evaluation_fraction", "calibration_fraction_of_non_evaluation"):
        if baseline_split.get(key) != provenance[key]:
            problems.append(f"{key} {provenance[key]} != baseline {baseline_split.get(key)}")
    for name in PARTITIONS:
        ours = provenance["contract_counts"][name]
        theirs = baseline_split.get("contract_counts", {}).get(name)
        if ours != theirs:
            problems.append(f"{name} contract count {ours} != baseline {theirs}")
        ours = provenance["contract_id_sha256"][name]
        theirs = baseline_split.get("contract_id_sha256", {}).get(name)
        if ours != theirs:
            problems.append(f"{name} contract-ID digest {ours[:12]} != baseline {str(theirs)[:12]}")

    if problems:
        raise SplitMismatchError(
            f"Seed {provenance['seed']}: split differs from the saved TF-IDF run:\n  - " + "\n  - ".join(problems)
        )


def verify_chunks_against_baseline(
    seed: int,
    chunk_counts: dict[str, int],
    labels: dict[str, np.ndarray],
    baseline_report: dict,
    baseline_dir: Path,
) -> None:
    """
    Check that chunking reproduced the baseline run: the same chunk count per partition and
    identical calibration and evaluation label matrices (row for row) as the saved
    word_char_tfidf .npz files. Raises SplitMismatchError on the first difference.
    """
    # Chunk counts recorded in the baseline report
    expected_counts = baseline_report.get("split", {}).get("chunk_counts", {})
    for name in PARTITIONS:
        if chunk_counts[name] != expected_counts.get(name):
            raise SplitMismatchError(
                f"Seed {seed}: {name} has {chunk_counts[name]} chunks, baseline has "
                f"{expected_counts.get(name)}; tokenizer or chunker settings differ"
            )

    # Labels row for row against the saved baseline predictions
    for name in ("calibration", "evaluation"):
        path = Path(baseline_dir) / f"{BASELINE_MODEL}_{name}.npz"
        if not path.is_file():
            raise SplitMismatchError(f"Seed {seed}: missing baseline predictions {path}")
        saved = np.load(path)["y_true"]
        if saved.shape != labels[name].shape or not np.array_equal(saved, labels[name]):
            raise SplitMismatchError(f"Seed {seed}: {name} labels differ from {path}")


# ----- per-seed outputs -----
def chunk_table(chunks: list[LabeledChunk]) -> pd.DataFrame:
    """One row per chunk (contract_id, chunk_index, start, end), in the same order as the score rows."""
    return pd.DataFrame(
        [{"contract_id": item.chunk.contract_id, "chunk_index": item.chunk.chunk_index,
          "start": item.chunk.start, "end": item.chunk.end} for item in chunks],
        columns=["contract_id", "chunk_index", "start", "end"],
    )


def environment_info() -> dict:
    """Library versions, GPU name and git commit, recorded in report.json for provenance."""
    import torch
    import transformers

    # The commit is best effort; a zip download has no git metadata
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit,
    }


def default_fit(config: TrainConfig, train_chunks: list[LabeledChunk], val_chunks: list[LabeledChunk],
                resume_from_checkpoint: str | None = None) -> dict:
    """Run src.models.encoder.train.fit (imported lazily so --dry-run never loads the Trainer)."""
    from src.models.encoder.train import fit

    return fit(config, train_chunks, val_chunks, resume_from_checkpoint=resume_from_checkpoint)


def default_score(checkpoint_dir: Path, partitions: dict[str, list[LabeledChunk]],
                  max_length: int, batch_size: int) -> dict[str, np.ndarray]:
    """Load a saved checkpoint on the GPU if there is one and score every partition's chunks."""
    from src.models.encoder.predict import load_checkpoint, predict_proba

    model, tokenizer = load_checkpoint(checkpoint_dir)
    return {
        name: predict_proba(model, tokenizer, [item.chunk for item in chunks],
                            max_length=max_length, batch_size=batch_size)
        for name, chunks in partitions.items()
    }


def write_seed_results(
    result_dir: Path,
    labels: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
    chunks: dict[str, list[LabeledChunk]],
    thresholds: dict[str, float],
    report_frame: pd.DataFrame,
    history_path: Path | None,
) -> None:
    """
    Write one seed's scores, chunk identifiers, thresholds and metrics (everything except
    report.json, which the caller writes last so an interrupted write is never "complete").
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    # Raw probabilities and labels, plus the chunk each row belongs to
    for name in ("calibration", "evaluation"):
        np.savez_compressed(result_dir / f"{name}_predictions.npz",
                            y_true=labels[name], y_prob=probabilities[name].astype(np.float32))
        chunk_table(chunks[name]).to_csv(result_dir / f"{name}_chunks.csv", index=False)

    # Calibration-fitted thresholds and the per-category evaluation table
    (result_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    report_frame.to_csv(result_dir / "per_category_metrics.csv", index=False)

    # Per-epoch loss and calibration AUPR, when the trainer saved them
    if history_path is not None and history_path.is_file():
        shutil.copyfile(history_path, result_dir / "training_history.json")


def run_seed(
    seed: int,
    contracts: list[Contract],
    config: TrainConfig,
    tokenizer,
    data_sha256: str,
    results_root: Path,
    work_root: Path,
    baseline_root: Path,
    batch_size: int = 32,
    force: bool = False,
    dry_run: bool = False,
    keep_checkpoints: bool = False,
    fit_fn: FitFn = default_fit,
    score_fn: ScoreFn = default_score,
) -> dict:
    """
    Train, calibrate and evaluate one seed, or skip it if its results are already complete.
    Returns a status dict (seed, status, result_dir, and macro metrics when evaluated).
    Raises SplitMismatchError when the split or chunks differ from the saved TF-IDF run, and
    RuntimeError when finished results or a half-trained checkpoint came from a different
    training config. fit_fn and score_fn are injectable so tests run without model weights.
    """
    result_dir = results_dir_for(results_root, seed)
    work_dir = Path(work_root) / f"seed{seed}"
    run_config = config.model_copy(update={"seed": seed, "output_dir": work_dir / "train"})
    fingerprint = config_fingerprint(run_config)

    # Resume: a finished seed is skipped, unless it was produced by a different config
    if is_seed_complete(result_dir) and not force:
        saved = json.loads((result_dir / "report.json").read_text(encoding="utf-8"))
        if saved.get("training", {}).get("config_sha256") != fingerprint:
            raise RuntimeError(
                f"Seed {seed}: {result_dir} was produced with a different training config. "
                "Use --force to overwrite it or choose another --results-root."
            )
        print(f"[seed {seed}] complete, skipping ({result_dir})")
        return {"seed": seed, "status": "skipped", "result_dir": str(result_dir),
                "macro_metrics": saved.get("macro_metrics")}

    # Verify the contract split against the saved TF-IDF run before spending any compute
    groups = split_contracts(contracts, seed)
    provenance = split_provenance(groups, seed)
    baseline_dir = baseline_dir_for(baseline_root, seed)
    baseline_report = load_baseline_report(baseline_dir)
    verify_split_against_baseline(provenance, data_sha256, baseline_report)

    # Chunk every partition with the shared sentence-aware chunker, then verify chunks and labels
    chunks = {name: prepare_contracts(groups[name], tokenizer, config.max_length, config.overlap_tokens)
              for name in PARTITIONS}
    labels = {name: to_model_inputs(chunks[name])[1] for name in PARTITIONS}
    provenance["chunk_counts"] = {name: len(chunks[name]) for name in PARTITIONS}
    verify_chunks_against_baseline(seed, provenance["chunk_counts"], labels, baseline_report, baseline_dir)
    print(f"[seed {seed}] split and chunks match {baseline_dir.name}: {provenance['chunk_counts']}")
    if dry_run:
        return {"seed": seed, "status": "verified", "result_dir": str(result_dir),
                "chunk_counts": provenance["chunk_counts"]}

    # A half-trained checkpoint may only be resumed with the config that started it
    work_dir.mkdir(parents=True, exist_ok=True)
    marker = work_dir / "run_config.json"
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("config_sha256") != fingerprint:
        if not force:
            raise RuntimeError(f"Seed {seed}: {work_dir} holds a run with a different config; "
                               "use --force or another --work-root")
    if force:
        shutil.rmtree(work_dir / "train", ignore_errors=True)
    marker.write_text(json.dumps({"config_sha256": fingerprint,
                                  "config": run_config.model_dump(mode="json")}, indent=2) + "\n",
                      encoding="utf-8")

    # Train unless a finished checkpoint already exists; continue from the last epoch if interrupted
    best_dir = work_dir / "train" / "best"
    started = time.perf_counter()
    if (best_dir / "train_config.json").is_file():
        print(f"[seed {seed}] trained checkpoint found, skipping training ({best_dir})")
        trained_now = False
    else:
        resume = latest_checkpoint(work_dir / "train")
        print(f"[seed {seed}] training on {len(chunks['training'])} chunks"
              + (f", resuming from {resume.name}" if resume else ""))
        fit_fn(run_config, chunks["training"], chunks["calibration"],
               resume_from_checkpoint=str(resume) if resume else None)
        trained_now = True
    training_seconds = time.perf_counter() - started

    # Score calibration and evaluation chunks; fit thresholds on calibration only
    probabilities = score_fn(best_dir, {name: chunks[name] for name in ("calibration", "evaluation")},
                             config.max_length, batch_size)
    thresholds = tune_thresholds(labels["calibration"], probabilities["calibration"], CATEGORIES)
    report_frame = build_report(labels["evaluation"], probabilities["evaluation"], thresholds, CATEGORIES)

    # Macro metrics with the same conventions as the TF-IDF comparison scripts
    macro = {
        "macro_f1": _macro_f1(labels["evaluation"], probabilities["evaluation"], _threshold_vector(thresholds)),
        "macro_aupr": _macro_aupr(labels["evaluation"], probabilities["evaluation"]),
        "macro_f1_supported_only": float(report_frame["f1"].mean(skipna=True)),
    }
    write_seed_results(result_dir, labels, probabilities, chunks, thresholds, report_frame,
                       best_dir / "log_history.json")

    # report.json last: its "complete" status is what makes the seed skippable next time
    report = {
        "experiment": EXPERIMENT_NAME,
        "model": MODEL_NAME,
        "status": "complete",
        "data": "data/cuad/train_separate_questions.json",
        "data_sha256": data_sha256,
        "split": provenance,
        "baseline_verified_against": baseline_dir.name,
        "chunking": {"method": "chunk_sentence_windows", "tokenizer": TOKENIZER_NAME,
                     "tokenizer_revision": TOKENIZER_REVISION, "max_tokens": config.max_length,
                     "overlap_tokens": config.overlap_tokens},
        "training": {"config_sha256": fingerprint, "config": run_config.model_dump(mode="json", exclude={"output_dir"}),
                     "checkpoint_selection": "best epoch by macro AUPR on calibration contracts",
                     "trained_in_this_invocation": trained_now,
                     "training_seconds_this_invocation": round(training_seconds, 1)},
        "thresholds_fit_on": "calibration contracts only",
        "metrics_on": "held-out evaluation contracts, never used for model/threshold fitting",
        "macro_f1_convention": "fixed 41-category macro; zero-support categories contribute F1=0",
        "macro_aupr_convention": "mean over categories with at least one evaluation positive",
        "macro_metrics": macro,
        "environment": environment_info(),
        "test_data_used": False,
    }
    (result_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Intermediate epoch checkpoints are large and no longer needed once results exist
    if not keep_checkpoints:
        prune_checkpoints(work_dir / "train")
    print(f"[seed {seed}] done: macro-F1 {macro['macro_f1']:.4f}, macro-AUPR {macro['macro_aupr']:.4f} "
          f"-> {result_dir}")
    return {"seed": seed, "status": "complete", "result_dir": str(result_dir), "macro_metrics": macro,
            "training_seconds": training_seconds}


def parse_seeds(seeds: list[int]) -> list[int]:
    """
    Validate requested seeds: unique and drawn from the predeclared SEEDS (the only seeds with a
    saved TF-IDF baseline). Returns them in the order given. Raises ValueError otherwise.
    """
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be unique, got {seeds}")
    unknown = [seed for seed in seeds if seed not in SEEDS]
    if unknown:
        raise ValueError(f"Seeds {unknown} are not predeclared; choose from {list(SEEDS)}")
    return list(seeds)


def build_parser() -> argparse.ArgumentParser:
    """Command-line options for the multi-seed runner."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/encoder.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                        help="Subset of the predeclared seeds 42 7 17 23 101 (default: all five)")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results",
                        help="Parent of the encoder_nested_seed<seed>/ result folders")
    parser.add_argument("--work-root", type=Path, default=ROOT / "outputs/encoder_nested",
                        help="Checkpoints (seed<seed>/train/); point at Google Drive to survive disconnects")
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results",
                        help="Folder holding the saved word_char_tfidf_nested* runs")
    parser.add_argument("--batch-size", type=int, default=32, help="Scoring batch size")
    parser.add_argument("--hf-cache-dir", type=Path, default=None,
                        help="Hugging Face cache folder (default: $CLAUSETRIAGE_HF_CACHE or the HF default)")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=None,
                        help="Never download from the Hub (default: $CLAUSETRIAGE_HF_OFFLINE or $HF_HUB_OFFLINE)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify splits, chunk counts and labels against the TF-IDF runs; do not train")
    parser.add_argument("--force", action="store_true", help="Re-run seeds even if their results are complete")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep per-epoch checkpoints after a seed finishes (best/ is always kept)")
    parser.add_argument("--register", action="store_true",
                        help="Register each checkpoint trained in this invocation in the local MLflow registry")
    return parser


def register_seed(config: TrainConfig, seed: int, result_dir: Path, work_root: Path, batch_size: int) -> dict:
    """Register one finished seed's checkpoint and thresholds in the local MLflow registry."""
    from src.models.encoder.predict import load_checkpoint
    from src.registry.bundle import ClauseModelBundle
    from src.registry.mlflow_registry import register_bundle
    from src.registry.predictors import EncoderPredictor

    report = json.loads((result_dir / "report.json").read_text(encoding="utf-8"))
    thresholds = json.loads((result_dir / "thresholds.json").read_text(encoding="utf-8"))
    model, tokenizer = load_checkpoint(Path(work_root) / f"seed{seed}" / "train" / "best")
    bundle = ClauseModelBundle(
        predictor=EncoderPredictor(model, tokenizer, config.max_length, batch_size),
        categories=CATEGORIES, thresholds=thresholds, model_name=MODEL_NAME,
        config=report["training"]["config"], seed=seed,
    )
    registered = register_bundle(
        bundle, experiment_name="clause-review-encoder", registry_name="clause-classifier",
        metrics={f"evaluation_{key}": float(value) for key, value in report["macro_metrics"].items()},
        metadata=report, tracking_uri=ROOT / "outputs/mlflow/tracking.db",
    )
    return {"run_id": registered.run_id, "model_version": registered.model_version}


def main(argv: list[str] | None = None) -> int:
    """Run (or verify) every requested seed in order and print a status table; return an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as error:
        parser.error(str(error))

    # The encoder and the shared chunker must use the same pinned tokenizer revision
    config = load_config(args.config)
    if config.chunking_method != "sentence_token":
        parser.error("nested comparison requires chunking_method=sentence_token")
    if config.model_name != TOKENIZER_NAME or config.model_revision != TOKENIZER_REVISION:
        parser.error("encoder and shared chunker must use the pinned common DistilRoBERTa revision")

    # Resolve cache/offline settings once and share them with fit and build_model
    hub: HubOptions = resolve_hub_options(args.hf_cache_dir, args.offline)
    export_hub_options(hub)
    print(f"Hugging Face: cache={hub.cache_dir or 'default'} offline={hub.local_files_only}")
    tokenizer = load_pinned_tokenizer(TOKENIZER_NAME, TOKENIZER_REVISION, hub)

    contracts = load_cuad(TRAIN_PATH)
    data_sha256 = hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()

    # Run seeds in the predeclared order; report elapsed time so the rest can be extrapolated
    statuses = []
    for seed in seeds:
        status = run_seed(
            seed, contracts, config, tokenizer, data_sha256,
            results_root=args.results_root, work_root=args.work_root, baseline_root=args.baseline_root,
            batch_size=args.batch_size, force=args.force, dry_run=args.dry_run,
            keep_checkpoints=args.keep_checkpoints,
        )
        if args.register and status["status"] == "complete":
            status["registry"] = register_seed(config, seed, Path(status["result_dir"]),
                                               args.work_root, args.batch_size)
        statuses.append(status)
        if "training_seconds" in status:
            print(f"[seed {seed}] training took {status['training_seconds'] / 60:.1f} min")

    print(json.dumps(statuses, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
