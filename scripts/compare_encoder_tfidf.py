"""Compare a candidate model's saved seed runs with the saved TF-IDF runs on the same splits.

Reporting only: reads saved probabilities and calibration-fitted thresholds, never retrains,
never re-tunes, never reads test.json. For every seed it checks that both runs used the same
training data and contract split (contract-ID digests in report.json) and identical
evaluation labels, then reports:

- per-seed macro-F1 and macro-AUPR for both models and their difference
- a paired contract-level bootstrap 95% interval for each difference (the helper in
  scripts/analyze_reranker_multiseed.py, so intervals match the reranker analysis)
- wins per seed and a predeclared verdict (see verdict())
- per-category deltas averaged over seeds, grouped by evaluation support; rare categories
  are marked "insufficient support" and are not ranked

The default candidate is the DistilRoBERTa runner's output (results/encoder_nested_seed<seed>).
Any run with the same layout works; --score-key and --threshold-key adapt others, e.g. the
retrieval reranker:

    python -m scripts.compare_encoder_tfidf --candidate-template "results/retrieval_reranker_seed{seed}" \\
        --candidate-name reranker --score-key reranker_score --threshold-key reranker \\
        --output outputs/reranker_vs_tfidf
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_reranker_multiseed import _bootstrap_delta, _macro_aupr, _macro_f1, _threshold_vector
from scripts.evaluate_encoder_nested import PARTITIONS, ROOT, SEEDS, baseline_dir_for
from scripts.train_retrieval_reranker import _metrics_at_thresholds
from src.labels.categories import CATEGORIES

DEFAULT_TEMPLATE = "results/encoder_nested_seed{seed}"
DEFAULT_OUTPUT = ROOT / "results/encoder_vs_tfidf"
BASELINE_MODELS = ("word_char_tfidf", "word_tfidf")

# Support buckets by mean evaluation positives per seed, as in scripts/analyze_errors.py
RARE_SUPPORT = 10
COMMON_SUPPORT = 30

# Predeclared verdict rule (docs/encoder-comparison.md), fixed before any encoder result exists:
# the bootstrap interval may include zero on at most this many seeds (4 of 5 must exclude it)
CI_MISSES_ALLOWED = 1


class PairingError(ValueError):
    """Raised when a candidate run and a baseline run are not on the same data, split or labels."""


# ----- loading saved runs -----
def load_candidate(run_dir: Path, score_key: str = "y_prob", threshold_key: str | None = None) -> dict:
    """
    Load one candidate seed: report, evaluation labels and scores, thresholds and (if saved)
    the evaluation chunk table. threshold_key selects a nested entry in thresholds.json for
    runs that store several models' thresholds in one file.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "report.json").is_file():
        raise FileNotFoundError(f"No candidate report at {run_dir / 'report.json'}")
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    predictions = np.load(run_dir / "evaluation_predictions.npz")
    thresholds = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))
    if threshold_key is not None:
        thresholds = thresholds[threshold_key]
    chunks_path = run_dir / "evaluation_chunks.csv"
    return {
        "report": report,
        "y_true": predictions["y_true"].astype(np.uint8),
        "scores": predictions[score_key].astype(np.float32),
        "thresholds": {category: float(thresholds[category]) for category in CATEGORIES},
        "chunks": pd.read_csv(chunks_path) if chunks_path.is_file() else None,
    }


def load_baseline(baseline_dir: Path, model: str) -> dict:
    """Load one saved TF-IDF seed for one model (word_tfidf or word_char_tfidf)."""
    baseline_dir = Path(baseline_dir)
    report = json.loads((baseline_dir / "report.json").read_text(encoding="utf-8"))
    predictions = np.load(baseline_dir / f"{model}_evaluation.npz")
    thresholds = json.loads((baseline_dir / f"{model}_thresholds.json").read_text(encoding="utf-8"))
    return {
        "report": report,
        "y_true": predictions["y_true"].astype(np.uint8),
        "scores": predictions["y_prob"].astype(np.float32),
        "thresholds": {category: float(thresholds[category]) for category in CATEGORIES},
    }


def verify_pairing(seed: int, candidate: dict, baseline: dict) -> None:
    """
    Raise PairingError unless both runs state test_data_used: false, used the same training
    file, the same seed and contract-ID digests in every partition, and identical evaluation
    label matrices (so row i is the same chunk in both).
    """
    problems = []
    for name, run in (("candidate", candidate), ("baseline", baseline)):
        if run["report"].get("test_data_used") is not False:
            problems.append(f"{name} report does not state test_data_used: false")
    if candidate["report"].get("data_sha256") != baseline["report"].get("data_sha256"):
        problems.append("training data hashes differ")

    # Split identity from the digests each report recorded
    candidate_split = candidate["report"].get("split") or {}
    baseline_split = baseline["report"].get("split") or {}
    if candidate_split.get("seed") != seed or baseline_split.get("seed") != seed:
        problems.append(f"report seeds {candidate_split.get('seed')}/{baseline_split.get('seed')} != {seed}")
    for name in PARTITIONS:
        ours = (candidate_split.get("contract_id_sha256") or {}).get(name)
        theirs = (baseline_split.get("contract_id_sha256") or {}).get(name)
        if ours is None or ours != theirs:
            problems.append(f"{name} contract-ID digest differs or is missing")

    # Row-for-row label identity
    if candidate["y_true"].shape != baseline["y_true"].shape or not np.array_equal(
            candidate["y_true"], baseline["y_true"]):
        problems.append("evaluation label matrices differ")
    if problems:
        raise PairingError(f"Seed {seed}: runs are not paired:\n  - " + "\n  - ".join(problems))


# ----- contract groups for the bootstrap -----
def contract_rows_from_chunks(chunks: pd.DataFrame) -> list[np.ndarray]:
    """Group evaluation row indices by contract_id, contracts in order of first appearance."""
    contract_ids = chunks["contract_id"].astype(str).to_numpy()
    order = list(dict.fromkeys(contract_ids))
    return [np.flatnonzero(contract_ids == contract_id) for contract_id in order]


@lru_cache(maxsize=None)
def _rechunk_inputs(hub_cache: Path | None, offline: bool | None) -> tuple:
    """Load the pinned tokenizer and the CUAD training contracts once per process."""
    from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
    from src.labels.loader import load_cuad
    from src.models.encoder.experiment_utils import TRAIN_PATH
    from src.models.encoder.hub import load_pinned_tokenizer, resolve_hub_options

    tokenizer = load_pinned_tokenizer(TOKENIZER_NAME, TOKENIZER_REVISION, resolve_hub_options(hub_cache, offline))
    return tokenizer, load_cuad(TRAIN_PATH)


@lru_cache(maxsize=None)
def _rechunked_rows(seed: int, hub_cache: Path | None, offline: bool | None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Re-chunk one seed's evaluation contracts once per process; returns (labels, contract rows)."""
    from scripts.analyze_reranker_multiseed import _prepare_evaluation

    tokenizer, contracts = _rechunk_inputs(hub_cache, offline)
    rebuilt, _, rows = _prepare_evaluation(seed, tokenizer, contracts)
    return rebuilt, rows


def contract_rows_by_rechunking(seed: int, y_true: np.ndarray, hub_cache: Path | None,
                                offline: bool | None) -> list[np.ndarray]:
    """
    Rebuild contract groups for runs that saved no chunk table, by re-chunking the evaluation
    contracts with the pinned tokenizer (the reranker analysis does the same). Cached per seed,
    so comparing against both baselines chunks once. Raises PairingError if the rebuilt labels
    differ from the saved ones.
    """
    rebuilt, rows = _rechunked_rows(seed, hub_cache, offline)
    if not np.array_equal(rebuilt, y_true):
        raise PairingError(f"Seed {seed}: re-chunked evaluation labels differ from the saved labels")
    return rows


# ----- metrics -----
def support_bucket(mean_support: float) -> str:
    """Bucket a category by mean evaluation positives per seed: rare (<10), uncommon (10-29), common (30+)."""
    if mean_support < RARE_SUPPORT:
        return "rare"
    return "uncommon" if mean_support < COMMON_SUPPORT else "common"


def compare_seed(seed: int, candidate: dict, baseline: dict, contract_rows: list[np.ndarray],
                 replicates: int) -> tuple[dict, list[dict], dict]:
    """
    Compare one paired seed. Returns (seed row with macro metrics and deltas, per-category
    rows, paired contract-bootstrap 95% intervals). Thresholds stay as each run saved them.
    """
    y_true = candidate["y_true"]
    candidate_vector = _threshold_vector(candidate["thresholds"])
    baseline_vector = _threshold_vector(baseline["thresholds"])

    # Macro metrics on the full evaluation partition
    candidate_f1 = _macro_f1(y_true, candidate["scores"], candidate_vector)
    baseline_f1 = _macro_f1(y_true, baseline["scores"], baseline_vector)
    candidate_aupr = _macro_aupr(y_true, candidate["scores"])
    baseline_aupr = _macro_aupr(y_true, baseline["scores"])
    interval = _bootstrap_delta(y_true, baseline["scores"], candidate["scores"], baseline_vector,
                                candidate_vector, contract_rows, seed, replicates)
    seed_row = {
        "seed": seed, "evaluation_contracts": len(contract_rows), "evaluation_chunks": int(len(y_true)),
        "baseline_macro_f1": baseline_f1, "candidate_macro_f1": candidate_f1,
        "delta_macro_f1": candidate_f1 - baseline_f1,
        "delta_macro_f1_lower_95": interval["macro_f1_delta"]["lower_95"],
        "delta_macro_f1_upper_95": interval["macro_f1_delta"]["upper_95"],
        "baseline_macro_aupr": baseline_aupr, "candidate_macro_aupr": candidate_aupr,
        "delta_macro_aupr": candidate_aupr - baseline_aupr,
        "delta_macro_aupr_lower_95": interval["macro_aupr_delta"]["lower_95"],
        "delta_macro_aupr_upper_95": interval["macro_aupr_delta"]["upper_95"],
    }

    # Per-category precision/recall/F1/AUPR at each run's own thresholds
    base = _metrics_at_thresholds(y_true, baseline["scores"], baseline["thresholds"]).set_index("category")
    cand = _metrics_at_thresholds(y_true, candidate["scores"], candidate["thresholds"]).set_index("category")
    category_rows = [{
        "seed": seed, "category": category, "support": int(base.loc[category, "support"]),
        **{f"baseline_{metric}": float(base.loc[category, metric]) for metric in ("precision", "recall", "f1", "aupr")},
        **{f"candidate_{metric}": float(cand.loc[category, metric]) for metric in ("precision", "recall", "f1", "aupr")},
        "delta_f1": float(cand.loc[category, "f1"] - base.loc[category, "f1"]),
        "delta_aupr": float(cand.loc[category, "aupr"] - base.loc[category, "aupr"]),
    } for category in CATEGORIES]
    return seed_row, category_rows, interval


def summarize_categories(by_seed: pd.DataFrame) -> pd.DataFrame:
    """
    Average per-category results over seeds and bucket them by mean support. Categories with
    adequate support are ranked by mean F1 delta (1 = largest candidate gain); rare ones get
    status "insufficient support" and no rank, because their F1 swings by 0.3-0.4 between seeds.
    """
    summary = by_seed.groupby("category", sort=False).agg(
        mean_support=("support", "mean"), min_support=("support", "min"),
        baseline_f1=("baseline_f1", "mean"), candidate_f1=("candidate_f1", "mean"),
        mean_delta_f1=("delta_f1", "mean"), min_delta_f1=("delta_f1", "min"), max_delta_f1=("delta_f1", "max"),
        mean_delta_aupr=("delta_aupr", "mean"),
        f1_wins=("delta_f1", lambda values: int((values > 0).sum())),
        f1_losses=("delta_f1", lambda values: int((values < 0).sum())),
        seeds=("seed", "nunique"),
    ).reset_index()

    # Bucket, then rank only categories with enough support to be read individually
    summary["support_bucket"] = summary["mean_support"].map(support_bucket)
    summary["status"] = np.where(summary["support_bucket"] == "rare", "insufficient support", "ranked")
    ranked = summary["status"] == "ranked"
    summary["rank_by_delta_f1"] = pd.Series(pd.NA, index=summary.index, dtype="Int64")
    summary.loc[ranked, "rank_by_delta_f1"] = (
        summary.loc[ranked, "mean_delta_f1"].rank(ascending=False, method="min").astype("Int64"))
    order = {"common": 0, "uncommon": 1, "rare": 2}
    return (summary.assign(_bucket=summary["support_bucket"].map(order))
            .sort_values(["_bucket", "mean_delta_f1"], ascending=[True, False])
            .drop(columns="_bucket").reset_index(drop=True))


def summarize_buckets(categories: pd.DataFrame) -> pd.DataFrame:
    """Mean F1/AUPR deltas and win counts per support bucket, over the category summary."""
    return categories.groupby("support_bucket", sort=False).agg(
        categories=("category", "count"), baseline_f1=("baseline_f1", "mean"),
        candidate_f1=("candidate_f1", "mean"), mean_delta_f1=("mean_delta_f1", "mean"),
        mean_delta_aupr=("mean_delta_aupr", "mean"),
        categories_improved=("mean_delta_f1", lambda values: int((values > 0).sum())),
        categories_worse=("mean_delta_f1", lambda values: int((values < 0).sum())),
    ).reset_index()


def verdict(per_seed: pd.DataFrame, expected_seeds: tuple[int, ...] = SEEDS) -> dict:
    """
    Apply the predeclared rule for "clearly beats the baseline":
    - clearly_beats: all expected seeds present, macro-F1 delta > 0 on every seed, the paired
      bootstrap 95% lower bound > 0 on all but CI_MISSES_ALLOWED seeds (4 of 5), and mean
      macro-AUPR delta > 0 (F1 gains not bought by worse ranking)
    - does_not_beat: mean macro-F1 delta <= 0, or macro-F1 wins on at most half the seeds
    - not_clearly: anything in between (a real but unproven or mixed gain)
    - incomplete: fewer seeds than expected, so no verdict is given
    """
    n_seeds = int(len(per_seed))
    wins = int((per_seed["delta_macro_f1"] > 0).sum())
    ci_above_zero = int((per_seed["delta_macro_f1_lower_95"] > 0).sum())
    mean_f1 = float(per_seed["delta_macro_f1"].mean()) if n_seeds else float("nan")
    mean_aupr = float(per_seed["delta_macro_aupr"].mean()) if n_seeds else float("nan")
    facts = {"seeds_compared": n_seeds, "seeds_expected": len(expected_seeds), "macro_f1_wins": wins,
             "seeds_with_f1_ci_above_zero": ci_above_zero, "mean_delta_macro_f1": mean_f1,
             "mean_delta_macro_aupr": mean_aupr}

    # Decide in order: missing seeds, clear loss, clear win, otherwise in between
    if set(per_seed["seed"]) != set(expected_seeds):
        return {"verdict": "incomplete", **facts,
                "reason": f"{n_seeds} of {len(expected_seeds)} predeclared seeds available"}
    if mean_f1 <= 0 or wins <= n_seeds // 2:
        return {"verdict": "does_not_beat", **facts,
                "reason": "mean macro-F1 delta is not positive or the candidate wins at most half the seeds"}
    if wins == n_seeds and ci_above_zero >= max(n_seeds - CI_MISSES_ALLOWED, 1) and mean_aupr > 0:
        return {"verdict": "clearly_beats", **facts,
                "reason": "wins every seed, bootstrap interval above zero on enough seeds, and AUPR also improves"}
    return {"verdict": "not_clearly", **facts,
            "reason": "positive on average but fails at least one clear-win condition"}


def compare(
    seeds: tuple[int, ...] = SEEDS,
    candidate_template: str = DEFAULT_TEMPLATE,
    candidate_name: str = "distilroberta",
    baseline_model: str = "word_char_tfidf",
    baseline_root: Path = ROOT / "results",
    score_key: str = "y_prob",
    threshold_key: str | None = None,
    replicates: int = 1000,
    allow_missing: bool = False,
    hub_cache: Path | None = None,
    offline: bool | None = None,
    expected_seeds: tuple[int, ...] = SEEDS,
) -> dict:
    """
    Compare a candidate against one TF-IDF baseline model over the given seeds and return all
    tables plus the verdict. A missing candidate seed raises FileNotFoundError unless
    allow_missing, in which case it is skipped and the verdict is "incomplete". The verdict
    is judged against expected_seeds (the five predeclared seeds by default).
    """
    if replicates < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("replicates must be positive and seeds must be unique")
    per_seed, per_category, intervals, missing = [], [], {}, []

    for seed in seeds:
        # Load both runs for this seed, skipping candidate seeds that have not been run yet
        run_dir = Path(candidate_template.format(seed=seed))
        run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
        if not (run_dir / "report.json").is_file() and allow_missing:
            missing.append(seed)
            continue
        candidate = load_candidate(run_dir, score_key, threshold_key)
        baseline = load_baseline(baseline_dir_for(baseline_root, seed), baseline_model)
        verify_pairing(seed, candidate, baseline)

        # Contract groups from the saved chunk table, or by re-chunking for older runs
        rows = (contract_rows_from_chunks(candidate["chunks"]) if candidate["chunks"] is not None
                else contract_rows_by_rechunking(seed, candidate["y_true"], hub_cache, offline))
        if sum(len(group) for group in rows) != len(candidate["y_true"]):
            raise PairingError(f"Seed {seed}: chunk table does not cover every evaluation row")
        seed_row, category_rows, intervals[str(seed)] = compare_seed(seed, candidate, baseline, rows, replicates)
        per_seed.append(seed_row)
        per_category.extend(category_rows)

    if not per_seed:
        raise FileNotFoundError(f"No candidate runs found for {candidate_template}")
    seed_frame = pd.DataFrame(per_seed)
    category_frame = pd.DataFrame(per_category)
    categories = summarize_categories(category_frame)
    aggregate = {
        metric: {"mean": float(seed_frame[metric].mean()),
                 "std_between_seeds": float(seed_frame[metric].std(ddof=1)) if len(seed_frame) > 1 else None}
        for metric in ("baseline_macro_f1", "candidate_macro_f1", "delta_macro_f1",
                       "baseline_macro_aupr", "candidate_macro_aupr", "delta_macro_aupr")
    }
    aggregate["macro_f1_wins"] = int((seed_frame["delta_macro_f1"] > 0).sum())
    aggregate["macro_aupr_wins"] = int((seed_frame["delta_macro_aupr"] > 0).sum())
    return {
        "candidate": candidate_name, "candidate_template": candidate_template, "baseline": baseline_model,
        "seeds_requested": list(seeds), "seeds_missing": missing,
        "bootstrap_replicates_per_seed": replicates,
        "protocol": {
            "splits": "candidate and baseline report identical contract-ID digests; evaluation labels identical row for row",
            "thresholds": "each run's own saved thresholds, fit on calibration contracts only; nothing re-tuned here",
            "macro_f1": "fixed 41-category macro; zero-support categories contribute F1=0",
            "macro_aupr": "mean over categories with at least one evaluation positive",
            "bootstrap": "paired resampling of evaluation contracts with replacement, numpy default_rng(20261105 + seed)",
            "support_buckets": f"mean evaluation positives per seed: rare <{RARE_SUPPORT} (not ranked), "
                               f"uncommon {RARE_SUPPORT}-{COMMON_SUPPORT - 1}, common {COMMON_SUPPORT}+",
            "interpretation": "seeded evaluation sets overlap, so the seeds are not independent samples",
            "test_data_used": False,
        },
        # The verdict always demands the five predeclared seeds, whatever subset was compared
        "verdict": verdict(seed_frame, expected_seeds),
        "aggregate": aggregate,
        "per_seed": seed_frame,
        "paired_contract_bootstrap_95pct": intervals,
        "per_category_by_seed": category_frame,
        "per_category_summary": categories,
        "support_bucket_summary": summarize_buckets(categories),
    }


def render_markdown(result: dict) -> str:
    """Short human-readable summary: verdict, per-seed table and bucket table."""
    v = result["verdict"]
    lines = [
        f"# {result['candidate']} vs {result['baseline']}", "",
        f"**Verdict: {v['verdict']}** ({v['reason']}).", "",
        f"Seeds compared: {v['seeds_compared']} of {v['seeds_expected']}; macro-F1 wins {v['macro_f1_wins']}; "
        f"bootstrap 95% lower bound above zero on {v['seeds_with_f1_ci_above_zero']} seeds.", "",
        "| Seed | Baseline F1 | Candidate F1 | Delta F1 [95% CI] | Baseline AUPR | Candidate AUPR | Delta AUPR [95% CI] |",
        "|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in result["per_seed"].itertuples():
        lines.append(
            f"| {row.seed} | {row.baseline_macro_f1:.3f} | {row.candidate_macro_f1:.3f} | "
            f"{row.delta_macro_f1:+.3f} [{row.delta_macro_f1_lower_95:+.3f}, {row.delta_macro_f1_upper_95:+.3f}] | "
            f"{row.baseline_macro_aupr:.3f} | {row.candidate_macro_aupr:.3f} | "
            f"{row.delta_macro_aupr:+.3f} [{row.delta_macro_aupr_lower_95:+.3f}, {row.delta_macro_aupr_upper_95:+.3f}] |")
    lines += ["", "| Support bucket | Categories | Baseline F1 | Candidate F1 | Mean delta F1 | Improved | Worse |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["support_bucket_summary"].itertuples():
        lines.append(f"| {row.support_bucket} | {row.categories} | {row.baseline_f1:.3f} | {row.candidate_f1:.3f} | "
                     f"{row.mean_delta_f1:+.3f} | {row.categories_improved} | {row.categories_worse} |")
    lines += ["", "Rare categories (<10 mean evaluation positives) are listed with status "
                  "\"insufficient support\" in per_category_summary.csv and are not ranked.", ""]
    return "\n".join(lines)


def write_outputs(result: dict, output: Path) -> None:
    """Write CSV tables, comparison.json and summary.md for one candidate/baseline pair."""
    output.mkdir(parents=True, exist_ok=True)
    result["per_seed"].to_csv(output / "per_seed.csv", index=False)
    result["per_category_by_seed"].to_csv(output / "per_category_by_seed.csv", index=False)
    result["per_category_summary"].to_csv(output / "per_category_summary.csv", index=False)
    result["support_bucket_summary"].to_csv(output / "support_bucket_summary.csv", index=False)
    serializable = {key: value for key, value in result.items() if not isinstance(value, pd.DataFrame)}
    serializable["per_seed"] = result["per_seed"].to_dict(orient="records")
    (output / "comparison.json").write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(render_markdown(result), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Compare against every requested baseline model, write outputs per baseline, print summaries."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--candidate-template", default=DEFAULT_TEMPLATE,
                        help="Candidate run folder with a {seed} placeholder, relative to the repo root or absolute")
    parser.add_argument("--candidate-name", default="distilroberta")
    parser.add_argument("--score-key", default="y_prob", help="Score array name in evaluation_predictions.npz")
    parser.add_argument("--threshold-key", default=None, help="Nested key in thresholds.json, if any")
    parser.add_argument("--baseline-models", nargs="+", default=list(BASELINE_MODELS), choices=BASELINE_MODELS)
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output folder; one vs_<baseline>/ subfolder per baseline model")
    parser.add_argument("--allow-missing-seeds", action="store_true",
                        help="Compare the seeds that exist (verdict becomes 'incomplete') instead of failing")
    parser.add_argument("--hf-cache-dir", type=Path, default=None,
                        help="Only used to re-chunk runs that saved no evaluation_chunks.csv")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)

    for baseline_model in args.baseline_models:
        result = compare(
            tuple(args.seeds), args.candidate_template, args.candidate_name, baseline_model,
            args.baseline_root, args.score_key, args.threshold_key, args.bootstrap_replicates,
            args.allow_missing_seeds, args.hf_cache_dir, args.offline,
        )
        write_outputs(result, args.output / f"vs_{baseline_model}")
        print(render_markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
