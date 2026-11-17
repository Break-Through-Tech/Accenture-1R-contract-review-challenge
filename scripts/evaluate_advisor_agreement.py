"""
Measure agreement between the model's clause risk scores and the advisor's returned packet.
Needs only the packet key and the returned CSV: rescoring under another weight set reads the
inputs stored in the key, so no model, cache or contract text is loaded.

Run from the repo root:
    python -m scripts.evaluate_advisor_agreement --key <advisor_packet_key.json> \\
        --returned <returned_packet.csv> --reviewer "<name>"
    ... --weights src/risk/weights/<other>.json   # rescore under another weight version

Writes to --output:
    report.json: inputs, every metric with its bootstrap interval, caveats and the threshold sweep
    items.csv: one row per judged clause, the model's score and rating beside the advisor's answers
    confusion_matrix.csv: advisor bucket (rows) by model bucket (columns)
    threshold_sweep.csv: agreement at every (medium, high) threshold setting on the grid
    advisor_rankings.csv: the AdvisorRanking records, in the advisor_rankings table's columns
"""
import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.risk.agreement import (
    BUCKETS,
    AdvisorRanking,
    AgreementReport,
    evaluate_agreement,
    load_key,
    load_returned_packet,
)
from src.risk.schema import load_weight_set

ROOT = Path(__file__).resolve().parents[1]
WRONG_CATEGORY_FLAG = "WRONG CATEGORY"


def flagged_wrong_category(notes: str) -> bool:
    """True when the advisor's note starts with WRONG CATEGORY (any case), per the instructions."""
    return notes.strip().upper().startswith(WRONG_CATEGORY_FLAG)


def _metric_rows(report: AgreementReport) -> list[dict]:
    """One row per headline metric (value, interval bounds, clauses used) for the console summary."""
    names = ("spearman_rank", "spearman_bucket", "exact_rate", "adjacent_rate", "weighted_kappa_linear",
             "weighted_kappa_quadratic", "high_precision", "high_recall")
    rows = []
    for name in names:
        estimate = getattr(report, name)
        if estimate is not None:
            rows.append({"metric": name, "value": estimate.value, "ci_low": estimate.ci_low,
                         "ci_high": estimate.ci_high, "n": estimate.n})
    return rows


def write_outputs(output: Path, report: AgreementReport, rankings: Sequence[AdvisorRanking], inputs: dict) -> None:
    """Write report.json and the four CSV tables to output."""
    output.mkdir(parents=True, exist_ok=True)
    body = {"inputs": inputs, **report.model_dump(mode="json")}
    (output / "report.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    pd.DataFrame([i.model_dump() for i in report.items]).to_csv(output / "items.csv", index=False)
    pd.DataFrame([r.model_dump() for r in rankings], columns=list(AdvisorRanking.model_fields)).to_csv(
        output / "advisor_rankings.csv", index=False)
    if report.confusion_matrix is not None:
        pd.DataFrame(report.confusion_matrix, index=[f"advisor_{b}" for b in BUCKETS],
                     columns=[f"model_{b}" for b in BUCKETS]).to_csv(output / "confusion_matrix.csv")
    if report.threshold_sweep is not None:
        pd.DataFrame([p.model_dump() for p in report.threshold_sweep.points]).to_csv(
            output / "threshold_sweep.csv", index=False)


def main(argv: Sequence[str] | None = None) -> AgreementReport:
    """Join the returned packet with its key, score, compute agreement, and write the report and tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--key", type=Path, required=True, help="advisor_packet_key.json from build_advisor_packet")
    parser.add_argument("--returned", type=Path, required=True, help="the packet CSV the advisor filled in")
    parser.add_argument("--reviewer", required=True, help="who filled in the packet (stored on every record)")
    parser.add_argument("--ranking-set", default=None, help="defaults to the key's packet_id")
    parser.add_argument("--weights", type=Path, default=None,
                        help="rescore under this weight set; default: the scores stored in the key")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "advisor_agreement")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--sweep-step", type=float, default=0.025)
    parser.add_argument("--sweep-half-width", type=float, default=0.15)
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="a threshold setting is stable if its exact rate is within this of the base rate")
    parser.add_argument("--exclude-wrong-category", action="store_true",
                        help="drop clauses the advisor's notes flag as WRONG CATEGORY (detection errors)")
    args = parser.parse_args(argv)

    # Join the returned packet with its key, optionally dropping flagged detection errors
    key = load_key(args.key)
    returned = load_returned_packet(args.returned, key, args.reviewer, args.ranking_set)
    flagged = sorted(j.item_id for j in returned.judged if flagged_wrong_category(j.notes))
    if args.exclude_wrong_category:
        kept = [j for j in returned.judged if j.item_id not in flagged]
        if not kept:
            raise ValueError("every judged clause is flagged WRONG CATEGORY; nothing left to evaluate")
        returned = returned.model_copy(update={"judged": kept})

    # Score (stored or rescored), evaluate, and write
    weight_set = load_weight_set(args.weights) if args.weights else None
    report = evaluate_agreement(returned, key, weight_set, n_boot=args.n_boot, seed=args.seed,
                                confidence=args.confidence, sweep_step=args.sweep_step,
                                sweep_half_width=args.sweep_half_width, sweep_tolerance=args.tolerance)
    inputs = {
        "key": str(args.key), "returned": str(args.returned), "weights": str(args.weights) if args.weights else None,
        "n_boot": args.n_boot, "bootstrap_seed": args.seed, "confidence": args.confidence,
        "flagged_wrong_category": flagged, "excluded_wrong_category": args.exclude_wrong_category,
        "skipped_items": returned.skipped,
    }
    write_outputs(args.output, report, [j.ranking for j in returned.judged], inputs)

    # Console summary: intervals and caveats, never the point estimates alone
    print(pd.DataFrame(_metric_rows(report)).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if report.threshold_sweep is not None:
        sweep = report.threshold_sweep
        print(f"High threshold stable over {sweep.high_stable_range}, Medium over {sweep.medium_stable_range} "
              f"(exact rate within {sweep.tolerance} of {sweep.base_exact_rate:.3f})")
    if report.scores_reproduced is False:
        print("WARNING: rescoring under the key's own weight set did not reproduce the stored scores; "
              "the scoring code has changed since the packet was built.")
    print("\n".join(f"- {line}" for line in report.caveats))
    return report


if __name__ == "__main__":
    main()
