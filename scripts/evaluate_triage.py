"""
Rank the validation contracts for triage from a saved probability cache, check the ranking against
a perfect-detection reference built from CUAD's annotations, and run the sensitivity analysis.
Reads only train_separate_questions.json (test.json is never touched) and never reruns a model.

Run from the repo root:
    python -m scripts.evaluate_triage
    python -m scripts.evaluate_triage --probabilities <cache.npz> --thresholds <thresholds.json>

The cache must come from the shared split and chunker (split_by_contract seed 42 / 0.2,
chunk_sentence_windows 512 / 64). The script rebuilds those chunks and refuses a cache whose
saved y_true doesn't match them row for row, since a misaligned cache would score the wrong text.

Writes to --output:
    ranking.csv: one row per contract, predicted rank and score next to the gold reference
    report.json: inputs, agreement with the gold reference, and the sensitivity tables
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.evidence.cues import load_category_cues
from src.labels.loader import load_cuad
from src.labels.splits import split_by_contract
from src.pipeline import prepare_contracts, to_model_inputs
from src.risk.contract import aggregate_contracts, sensitivity
from src.risk.inputs import clauses_from_probabilities, gold_clauses
from src.risk.schema import DEFAULT_WEIGHTS_PATH, SIGNAL_NAMES, WeightSet, load_weight_set
from src.risk.scoring import score_contract_clauses

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "cuad" / "train_separate_questions.json"
CACHE_DIR = ROOT / "results" / "word_char_tfidf"
# Must match how the cache was produced (see the cache's report.json)
VAL_FRACTION, SEED, MAX_TOKENS, OVERLAP_TOKENS = 0.2, 42, 512, 64
TOP_N = 10  # "which contracts to open first": size of the top list compared between rankings
WEIGHT_STEP, THRESHOLD_STEP = 0.1, 0.05


def _reweighted(ws: WeightSet, signal: str, delta: float) -> WeightSet:
    """Move one signal weight by delta (kept in [0, 1]) and rescale the others so they still sum to 1."""
    old = ws.signal_weights[signal]
    # shortcut: divides by zero if this signal already has weight 1.0; guard it if a weight set ever does that
    new = min(max(old + delta, 0.0), 1.0)
    scale = (1 - new) / (1 - old)
    weights = {name: (new if name == signal else w * scale) for name, w in ws.signal_weights.items()}
    return ws.model_copy(update={"weight_version_id": f"{ws.weight_version_id}|{signal}{delta:+.2f}",
                                 "signal_weights": weights})


def _variants(ws: WeightSet) -> list[WeightSet]:
    """Every signal weight +/- WEIGHT_STEP, plus other triage settings."""
    variants = [_reweighted(ws, s, d) for s in SIGNAL_NAMES for d in (-WEIGHT_STEP, WEIGHT_STEP)]
    for field, value in [("triage_top_k", 1), ("triage_top_k", 5), ("triage_max_weight", 0.25),
                         ("triage_max_weight", 0.75), ("triage_max_weight", 1.0)]:
        if getattr(ws, field) != value:
            variants.append(ws.model_copy(update={"weight_version_id": f"{ws.weight_version_id}|{field}={value}",
                                                  field: value}))
    return variants


def _top(ranks: pd.Series) -> set[str]:
    """The first TOP_N contracts by rank, ties broken by contract_id (as aggregate_contracts lists them)."""
    return set(sorted(ranks.index, key=lambda cid: (ranks[cid], cid))[:TOP_N])


def _top_overlap(a: pd.Series, b: pd.Series) -> int:
    """How many of a's first TOP_N contracts are also in b's first TOP_N (exactly TOP_N each, so ties can't inflate it)."""
    return len(_top(a) & _top(b))


def _agreement(ranks: pd.Series, reference: pd.Series) -> dict:
    """Spearman correlation, largest rank move and top-N overlap of one ranking against another."""
    return {
        "spearman": round(float(pd.concat([ranks, reference], axis=1).corr(method="spearman").iloc[0, 1]), 4),
        "max_rank_move": int((ranks - reference).abs().max()),
        f"top_{TOP_N}_overlap": _top_overlap(ranks, reference),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--probabilities", type=Path, default=CACHE_DIR / "word_char_tfidf_validation.npz",
                        help=".npz with y_prob (and y_true, checked against the rebuilt chunks)")
    parser.add_argument("--thresholds", type=Path, default=CACHE_DIR / "word_char_tfidf_thresholds.json")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "contract_triage")
    args = parser.parse_args()

    # Rebuild the validation chunks exactly as the cache was produced, and check they line up
    _, val = split_by_contract(load_cuad(TRAIN_PATH), val_fraction=VAL_FRACTION, seed=SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION)
    labeled = prepare_contracts(val, tokenizer, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS)
    chunks = [lc.chunk for lc in labeled]
    cache = np.load(args.probabilities, allow_pickle=False)
    probabilities = cache["y_prob"]
    if "y_true" in cache.files and not np.array_equal(cache["y_true"], to_model_inputs(labeled)[1]):
        raise ValueError("Cache labels don't match the rebuilt chunks: wrong split, chunker or settings")

    # Predicted and gold-reference clauses, scored under the base weight set
    ws = load_weight_set(args.weights)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    contract_ids = [c.contract_id for c in val]
    predicted = clauses_from_probabilities({c.contract_id: c.text for c in val}, chunks, probabilities,
                                           thresholds, load_category_cues())
    gold = gold_clauses(val, chunks)
    ranked = aggregate_contracts(score_contract_clauses(predicted, ws), ws, contract_ids)
    gold_ranked = {c.contract_id: c for c in aggregate_contracts(score_contract_clauses(gold, ws), ws, contract_ids)}

    # ranking.csv: the reviewer-facing list, with the gold reference alongside
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "rank": c.rank, "contract_id": c.contract_id, "triage_score": c.triage_score,
        "high": c.high_count, "medium": c.medium_count, "low": c.low_count, "unknown": c.unknown_count,
        "top_clauses": "; ".join(f"{s.category} ({s.composite_score:.2f} {s.rating})" for s in c.top_clauses),
        "gold_rank": gold_ranked[c.contract_id].rank, "gold_triage_score": gold_ranked[c.contract_id].triage_score,
    } for c in ranked]).to_csv(args.output / "ranking.csv", index=False)

    # Agreement with the gold reference, for the triage score and two simpler rules
    pred_rank = pd.Series({c.contract_id: c.rank for c in ranked})
    gold_rank = pd.Series({cid: c.rank for cid, c in gold_ranked.items()})
    simple_rules = {
        "triage_score": pred_rank,
        "high_count": pd.Series({c.contract_id: -c.high_count for c in ranked}).rank(method="min"),
        "worst_clause": pd.Series({c.contract_id: -c.top_clauses[0].composite_score if c.top_clauses else 0.0
                                   for c in ranked}).rank(method="min"),
    }
    vs_gold = {name: _agreement(ranks, gold_rank) for name, ranks in simple_rules.items()}

    # Sensitivity: rerank predicted and gold clauses under each variant
    variants = _variants(ws)
    pred_ranks = sensitivity(predicted, [ws, *variants], contract_ids)
    gold_ranks = sensitivity(gold, [ws, *variants], contract_ids)
    weight_sensitivity = [
        {"variant": v.weight_version_id.split("|", 1)[1],
         **_agreement(pred_ranks[v.weight_version_id], pred_ranks[ws.weight_version_id]),
         "spearman_vs_gold": _agreement(pred_ranks[v.weight_version_id], gold_ranks[v.weight_version_id])["spearman"]}
        for v in variants
    ]

    # Rating boundaries: thresholds move clause buckets but never the contract ranking
    base_ratings = [s.rating for s in score_contract_clauses(predicted, ws)]
    threshold_sensitivity = []
    for field in ("medium_threshold", "high_threshold"):
        for delta in (-THRESHOLD_STEP, THRESHOLD_STEP):
            moved = ws.model_copy(update={"weight_version_id": f"{ws.weight_version_id}|{field}{delta:+.2f}",
                                          field: round(getattr(ws, field) + delta, 9)})
            ratings = [s.rating for s in score_contract_clauses(predicted, moved)]
            threshold_sensitivity.append({
                "variant": f"{field}{delta:+.2f}",
                "clause_ratings_changed": sum(a != b for a, b in zip(base_ratings, ratings)),
                "high_clauses": ratings.count("High"), "medium_clauses": ratings.count("Medium"),
            })

    report = {
        "inputs": {
            "probabilities": args.probabilities.relative_to(ROOT).as_posix() if args.probabilities.is_relative_to(ROOT)
            else str(args.probabilities),
            "thresholds": args.thresholds.name,
            "weight_version_id": ws.weight_version_id, "weight_set_sha256": ws.content_sha256,
            "triage_top_k": ws.triage_top_k, "triage_max_weight": ws.triage_max_weight,
            "split": {"method": "split_by_contract", "seed": SEED, "val_fraction": VAL_FRACTION},
            "chunking": {"method": "chunk_sentence_windows", "tokenizer": TOKENIZER_NAME,
                         "tokenizer_revision": TOKENIZER_REVISION, "max_tokens": MAX_TOKENS,
                         "overlap_tokens": OVERLAP_TOKENS},
            "test_data_used": False,
        },
        "counts": {
            "contracts": len(ranked), "chunks": len(chunks), "predicted_clauses": len(predicted),
            "gold_spans": len(gold), "distinct_triage_scores": len({c.triage_score for c in ranked}),
            "base_ratings": {r: base_ratings.count(r) for r in ("High", "Medium", "Low")},
        },
        "gold_reference_note": "Gold = CUAD annotations scored as if detected perfectly, with the full annotated "
                               "answer as evidence. It measures how much detection and evidence errors move the "
                               "ranking, not whether the ranking matches a reviewer's judgement.",
        "agreement_with_gold": vs_gold,
        "weight_sensitivity": weight_sensitivity,
        "threshold_sensitivity": threshold_sensitivity,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("counts", "agreement_with_gold")}, indent=2))
    print(pd.DataFrame(weight_sensitivity).to_string(index=False))
    print(pd.DataFrame(threshold_sensitivity).to_string(index=False))


if __name__ == "__main__":
    main()
