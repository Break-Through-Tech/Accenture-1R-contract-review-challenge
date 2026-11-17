"""
Export resolved reviewer feedback from an artifact store as labelled chunk examples for training.
Only contracts in the training manifest are exported; held-out contracts, contracts outside the
manifest (uploads, anything from test.json) and unresolved reviewer conflicts are excluded and
reported. Never reads data/cuad/test.json. See docs/feedback-review-queue.md.

Run from the repo root:
    python -m scripts.export_feedback --store outputs/artifact_store.sqlite
    python -m scripts.export_feedback --store <db> --model-version clause-classifier-tfidf:1 --output <folder>
    python -m scripts.export_feedback --store <db> --manifest my_manifest.json

The default manifest is the training partition of train_separate_questions.json under the project split
(split_by_contract, --val-fraction 0.2, --seed 42); its validation partition is held out. --manifest
loads a TrainingManifest JSON instead (for example one saved with --save-manifest and then edited to
add uploaded contracts on purpose).

Writes to --output (default outputs/feedback_export/, which git ignores):
    examples.jsonl  one labelled (chunk, category) example per line
    report.json     counts, exclusions with reasons, manifest identity and the SHA-256 of examples.jsonl
"""
import argparse
from collections.abc import Sequence
from pathlib import Path

from src.review.export import build_training_manifest, export_training_examples, write_export
from src.review.schema import TrainingManifest
from src.store.artifact_store import ArtifactStore

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "cuad" / "train_separate_questions.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "feedback_export"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", type=Path, required=True, help="artifact store SQLite file to read")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="folder for examples.jsonl and report.json")
    parser.add_argument("--model-version", help="only feedback given under this model version (default: all)")
    parser.add_argument("--manifest", type=Path, help="TrainingManifest JSON to use instead of building one")
    parser.add_argument("--train", type=Path, default=TRAIN_PATH, help="CUAD training file for the default manifest")
    parser.add_argument("--seed", type=int, default=42, help="split seed for the default manifest")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="held-out fraction for the default manifest")
    parser.add_argument("--save-manifest", type=Path, help="also write the manifest used to this JSON file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load or build the manifest, export the store's feedback, write the files and print a summary."""
    args = parse_args(argv)
    # Opening a missing path would create an empty store, so refuse it instead
    if not args.store.is_file():
        raise SystemExit(f"no artifact store at {args.store}")
    if args.manifest is not None:
        manifest = TrainingManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    else:
        manifest = build_training_manifest(args.train, val_fraction=args.val_fraction, seed=args.seed)
    if args.save_manifest is not None:
        args.save_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.save_manifest.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")

    store = ArtifactStore(args.store)
    try:
        export = export_training_examples(store, manifest, model_version=args.model_version)
    finally:
        store.close()
    examples_path, report_path = write_export(export, args.output)
    print(f"{export.example_count} examples ({export.positive_count} positive) -> {examples_path}")
    excluded = ", ".join(f"{reason} {count}" for reason, count in export.excluded_counts.items()) or "none"
    print(f"excluded: {excluded}; report -> {report_path}")


if __name__ == "__main__":
    main()
