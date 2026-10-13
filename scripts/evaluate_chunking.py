"""Run the training-only boundary comparison with explicit character configurations."""

import argparse
import csv
import json
from pathlib import Path

from src.chunking.evaluation import evaluate_boundaries
from src.labels.loader import load_cuad

TRAIN_PATH = Path(__file__).resolve().parents[1] / "data/cuad/train_separate_questions.json"
# Reference settings first; the remaining settings explore overlap and window size.
CONFIGURATIONS = [(800, 200), (1500, 200), (800, 0), (1500, 0), (3000, 200)]


def main() -> None:
    """Load official training annotations once and save metrics and grounded examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs=2, type=int, action="append",
                        metavar=("WINDOW_CHARS", "OVERLAP_CHARS"),
                        help="Repeat to override the documented comparison settings.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    # The caller intentionally exposes no dataset option: only training is read.
    contracts = load_cuad(TRAIN_PATH)
    reports = [evaluate_boundaries(contracts, window, overlap)
               for window, overlap in (args.config or CONFIGURATIONS)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [report.metrics.model_dump() for report in reports]
    with (args.output_dir / "chunking_evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "dataset": "data/cuad/train_separate_questions.json",
        "contract_records": len(contracts),
        "deduplication_key": ["contract_id", "category", "start", "end"],
        "reports": [report.model_dump() for report in reports],
    }
    (args.output_dir / "chunking_examples.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    # Print the comparison and a few examples; full excerpts remain in the JSON.
    for report in reports:
        m = report.metrics
        print(f"window={m.window_chars}, overlap={m.overlap_chars}: "
              f"raw={m.raw_annotations}, unique={m.unique_annotations}, "
              f"duplicates={m.duplicate_annotations}; cut={m.cut_pct:.2f}%, "
              f"contained={m.contained_pct:.2f}%, cut/not contained={m.cut_not_contained_pct:.2f}%, "
              f"longer than window={m.longer_than_window_annotations}")
        for example in report.examples:
            print(f"  {example.kind}: {example.category} [{example.start}, {example.end}); "
                  f"cutting={example.cutting_chunks}; containing={example.containing_chunks}")
            print(f"    {example.annotation_preview!r}")
    print(f"Saved summary and examples to {args.output_dir}")


if __name__ == "__main__":
    main()
