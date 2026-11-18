"""
One-time sealed evaluation of the frozen word + character TF-IDF bundle, as pre-registered in
docs/final-test-evaluation.md. Every setting below is frozen there; nothing here is tuned.

Development dry run (reads only train_separate_questions.json; safe to run any time):
    python -m scripts.evaluate_final_test --write-development-holdout outputs/final_test_dry_run/development_holdout.json
    python -m scripts.evaluate_final_test --data outputs/final_test_dry_run/development_holdout.json \\
        --output outputs/final_test_dry_run/run --preflight
    python -m scripts.evaluate_final_test --data outputs/final_test_dry_run/development_holdout.json \\
        --output outputs/final_test_dry_run/run

Sealed run (phase 2 only; the same commands with the test file, the fixed output and the opt-in flag):
    python -m scripts.evaluate_final_test --data data/cuad/test.json --output results/final_test_evaluation \\
        --sealed-test-run --preflight
    git add results/final_test_evaluation/preflight.json && git commit ...
    python -m scripts.evaluate_final_test --data data/cuad/test.json --output results/final_test_evaluation \\
        --sealed-test-run

Guards: test.json is read only with --sealed-test-run; CUADv1.json and zip archives are always
refused; the sealed run needs committed code, a committed passing preflight for the same file, and
refuses to start if a result exists on disk or in git history. Nothing about a metric is printed
until every output file is written, and report.json is written last.
"""
import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from transformers import AutoTokenizer

from scripts.evaluate_triage import TOP_N, _agreement
from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.evidence.cues import load_category_cues
from src.holdout_metrics import (
    bootstrap_replicates, category_counts, compare_with_reference, contract_groups, contract_presence,
    bca_interval, detection_table, f1_from_counts, jackknife_replicates, macro_aupr, macro_f1, per_category_aupr,
    predict_at_thresholds, spearman, support_bucket,
)
from src.holdout_report import bucket_frame, jsonable, per_category_frame, render_summary
from src.inference import ContractInferencePipeline
from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.labels.loader import Contract, load_cuad
from src.labels.projection import spans_overlap
from src.labels.splits import split_by_contract
from src.models.naive.tfid_baseline import fit_tfidf_baseline
from src.pipeline import prepare_contracts, to_model_inputs
from src.registry.bundle import ClauseModelBundle
from src.registry.demo_model import BUNDLE_PATH, BUNDLE_SHA256, file_sha256
from src.risk.contract import aggregate_contracts
from src.risk.inputs import clauses_from_probabilities, gold_clauses
from src.risk.schema import DEFAULT_WEIGHTS_PATH, load_weight_set
from src.risk.scoring import score_contract_clauses
from src.sealed_run import (
    ATTEMPTS_FILE, PREFLIGHT_FILE, REPORT_FILE, SealedRunError, check_mode, check_sealed_output,
    check_sealed_preconditions, classify_data_path, environment, git_blob_id, git_history_has, git_state,
    git_tracked, ids_sha256, last_commit_of, preflight_failures, start_attempt, structure_report, text_file_sha256,
)

ROOT = Path(__file__).resolve().parents[1]

# ----- Frozen configuration (docs/final-test-evaluation.md, section 2) -----
PROTOCOL_PATH = ROOT / "docs" / "final-test-evaluation.md"
SEALED_DATA_PATH = ROOT / "data" / "cuad" / "test.json"
SEALED_DATA_GIT_BLOB = "16cb5c6aa4733bcaf46e24a1ee868b47a6731461"  # from the git index; file unchanged since d0a04de
SEALED_OUTPUT = ROOT / "results" / "final_test_evaluation"
EXPECTED_TEST_CONTRACTS = 102
TRAIN_PATH = ROOT / "data" / "cuad" / "train_separate_questions.json"
TRAIN_SHA256 = "7de21d2bb741ac939e2a839ef9640634a5e0f9f40ea2b82c6833baf7db33dab4"
SAVED_RUN = ROOT / "results" / "word_char_tfidf_nested"
DEVELOPMENT_SEEDS = (42, 7, 17, 23, 101)
SPLIT_SEED, EVALUATION_FRACTION, CALIBRATION_FRACTION = 42, 0.2, 0.2
PARTITION_DIGESTS = {
    "training": "44e5d6e9df85fa666896aca96e4638abce02371e5674316f1bad13bf079922fd",
    "calibration": "271e06814672af99ef807e67790290b82c41a1f9890f1402247cd6666ac95f69",
    "evaluation": "495488d5b49f0c1cd93f5a8bb416c110ed96122e8d03b3bcc4f82784068a87f9",
}
MAX_TOKENS, OVERLAP_TOKENS = 512, 64
TOKENIZER_CACHE = ROOT / ".venv" / "hf-cache"
WEIGHT_VERSION_ID = "v2-team-proposed"
WEIGHT_SET_SHA256 = "8d8e8e264b8c4ffe07c7e22cab7be70c4524ce7759e62c64432b83214e690995"
# Inputs that decide the numbers, by line-ending-normalised SHA-256; any change stops the run before the data is read
PINNED_FILES = {
    SAVED_RUN / "word_char_tfidf_thresholds.json": "8025756b8f12c6fe4dc8b6782d71e714450e6ac218c9b3a03d137154d2f7db28",
    SAVED_RUN / "word_tfidf_thresholds.json": "aa419d2cc53e6884813448ec2ad3c723fd17aa9711998ab1a2ff5a69571cbe3e",
    ROOT / "src" / "risk" / "weights" / "default_weights_v2.json":
        "770313649a6a51572fb2c9ad90662023ff6c158c4b7f54bca176aa281a387c6c",
    ROOT / "data" / "cuad" / "category_descriptions.csv":
        "f80c236906f171649aebe98db9cdbcf8595854cf54667c2e4274ba77dcfd6e7b",
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261105 + SPLIT_SEED  # the project's rule: 20261105 + split seed (the bundle's split)
REPRODUCTION_ATOL = 1e-5  # bundle and refit vs saved calibration probabilities
MODELS = ("word_char_tfidf", "word_tfidf")
DISCLOSURES = [
    "2026-11-17: a repository-wide text search for a word matched data/cuad/test.json and printed a "
    "match count for it. No content was displayed or used.",
]


class IntegrityError(RuntimeError):
    """Raised when the frozen artifact or a pre-data check does not match the protocol."""


def _shown(path: Path) -> str:
    """Repo-relative POSIX path when inside the repo, else the absolute path."""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def _seed_dir(seed: int) -> Path:
    """Saved nested TF-IDF run for one development seed."""
    return SAVED_RUN if seed == SPLIT_SEED else SAVED_RUN.parent / f"{SAVED_RUN.name}_seed{seed}"


def _thresholds(path: Path) -> dict[str, float]:
    """Read a saved per-category threshold file."""
    return {key: float(value) for key, value in json.loads(path.read_text(encoding="utf-8")).items()}


def _vector(thresholds: dict[str, float]) -> np.ndarray:
    """Thresholds as a float64 vector in CATEGORIES order."""
    return np.asarray([thresholds[category] for category in CATEGORIES], dtype=np.float64)


# ----- The frozen artifact and the training-file partitions -----
def load_frozen_bundle(path: Path = BUNDLE_PATH) -> ClauseModelBundle:
    """
    Load the released bundle only if its SHA-256 is the pinned one, it unpickles without a version
    warning, its thresholds equal the saved calibration thresholds, and its config matches the protocol.
    """
    actual = file_sha256(path)
    if actual != BUNDLE_SHA256:
        raise IntegrityError(f"{path} has SHA-256 {actual}, the protocol pins {BUNDLE_SHA256}")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        bundle = joblib.load(path)
    expected_config = {"split_seed": SPLIT_SEED, "max_tokens": MAX_TOKENS, "overlap_tokens": OVERLAP_TOKENS,
                       "tokenizer": TOKENIZER_NAME, "tokenizer_revision": TOKENIZER_REVISION, "c": 1.0}
    problems = [key for key, value in expected_config.items() if bundle.config.get(key) != value]
    if bundle.model_name != "word_char_tfidf" or bundle.categories != CATEGORIES or problems:
        raise IntegrityError(f"bundle metadata differs from the protocol: {problems or bundle.model_name}")
    if bundle.thresholds != _thresholds(SAVED_RUN / "word_char_tfidf_thresholds.json"):
        raise IntegrityError("bundle thresholds differ from results/word_char_tfidf_nested/word_char_tfidf_thresholds.json")
    return bundle


def check_pinned_files(pinned: dict[Path, str] = PINNED_FILES) -> dict[str, str]:
    """Raise IntegrityError unless every pinned input file has its pinned (CRLF-normalised) SHA-256; returns the hashes."""
    hashes = {_shown(path): text_file_sha256(path) for path in pinned}
    changed = [_shown(path) for path, expected in pinned.items() if hashes[_shown(path)] != expected]
    if changed:
        raise IntegrityError(f"pinned input files changed: {changed}")
    return hashes


def load_tokenizer():
    """The pinned DistilRoBERTa tokenizer from the local cache, never downloaded."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
                                         cache_dir=TOKENIZER_CACHE, local_files_only=True)


def training_partitions() -> dict[str, list[Contract]]:
    """The seed-42 training/calibration/evaluation partitions of the training file, checked against their digests."""
    if file_sha256(TRAIN_PATH) != TRAIN_SHA256:
        raise IntegrityError("train_separate_questions.json differs from the pinned SHA-256")
    development, evaluation = split_by_contract(load_cuad(TRAIN_PATH), val_fraction=EVALUATION_FRACTION,
                                                seed=SPLIT_SEED)
    training, calibration = split_by_contract(development, val_fraction=CALIBRATION_FRACTION, seed=SPLIT_SEED)
    partitions = {"training": training, "calibration": calibration, "evaluation": evaluation}
    for name, group in partitions.items():
        if ids_sha256([c.contract_id for c in group]) != PARTITION_DIGESTS[name]:
            raise IntegrityError(f"the {name} partition differs from the saved run's contract-ID digest")
    return partitions


def development_reference() -> dict:
    """
    The development numbers the test result is compared with, recomputed from the saved nested runs
    (each seed's own calibration-fitted thresholds): per-seed macro metrics for both models, their
    mean/min/max, and per-category mean support, bucket, F1 and AUPR over the five seeds.
    """
    per_seed, per_category = [], []
    for seed in DEVELOPMENT_SEEDS:
        row = {"seed": seed}
        for model in MODELS:
            saved = np.load(_seed_dir(seed) / f"{model}_evaluation.npz")
            table = detection_table(saved["y_true"], saved["y_prob"], _vector(_thresholds(_seed_dir(seed) / f"{model}_thresholds.json")))
            row[f"{model}_macro_f1"] = macro_f1(table["f1"].to_numpy())
            row[f"{model}_macro_aupr"] = macro_aupr(table["aupr"].to_numpy())
            per_category.append(table.assign(seed=seed, model=model))
        per_seed.append(row)
    frame = pd.DataFrame(per_seed)
    categories = pd.concat(per_category)
    word_char = categories[categories["model"] == "word_char_tfidf"].groupby("category", sort=False)
    summary = pd.DataFrame({
        "dev_mean_support": word_char["support"].mean(),
        "dev_mean_f1": word_char["f1"].mean(),
        "dev_mean_aupr": word_char["aupr"].mean(),  # pandas skips NaN (seeds with no positives)
    }).reindex(CATEGORIES)
    summary["dev_bucket"] = summary["dev_mean_support"].map(support_bucket)
    macro = {f"{model}_{metric}": {"mean": float(frame[f"{model}_{metric}"].mean()),
                                   "min": float(frame[f"{model}_{metric}"].min()),
                                   "max": float(frame[f"{model}_{metric}"].max()),
                                   "seed_42": float(frame.loc[frame["seed"] == SPLIT_SEED, f"{model}_{metric}"].iloc[0])}
             for model in MODELS for metric in ("macro_f1", "macro_aupr")}
    return {"per_seed": per_seed, "macro": macro, "per_category": summary.reset_index(names="category")}


def fit_word_reference(training: list[Contract], tokenizer):
    """
    The original word-only baseline refitted on the same 260 training contracts with its default
    settings (src/models/naive/tfid_baseline.py). Returns (predict function, training chunk count).
    """
    texts, labels = to_model_inputs(prepare_contracts(training, tokenizer, max_tokens=MAX_TOKENS,
                                                      overlap_tokens=OVERLAP_TOKENS))
    vectorizer, classifier = fit_tfidf_baseline(texts, labels)
    return (lambda chunk_texts: np.asarray(classifier.predict_proba(vectorizer.transform(chunk_texts))),
            len(texts))


def check_reproduction(predictors: dict, calibration: list[Contract], tokenizer) -> dict:
    """
    Before any data is read: both models must reproduce the saved seed-42 calibration probabilities
    (same labels, max difference within REPRODUCTION_ATOL). Returns the differences for the report.
    """
    texts, labels = to_model_inputs(prepare_contracts(calibration, tokenizer, max_tokens=MAX_TOKENS,
                                                      overlap_tokens=OVERLAP_TOKENS))
    checks = {}
    for model, predict in predictors.items():
        saved = np.load(SAVED_RUN / f"{model}_calibration.npz")
        probabilities = predict(texts)
        if not np.array_equal(saved["y_true"], labels):
            raise IntegrityError(f"{model}: rebuilt calibration labels differ from the saved run")
        difference = float(np.max(np.abs(probabilities.astype(np.float64) - saved["y_prob"].astype(np.float64))))
        vector = _vector(_thresholds(SAVED_RUN / f"{model}_thresholds.json"))
        flipped = int(np.sum(predict_at_thresholds(probabilities, vector) != predict_at_thresholds(saved["y_prob"], vector)))
        if difference > REPRODUCTION_ATOL:
            raise IntegrityError(f"{model}: calibration probabilities differ from the saved run by {difference}")
        checks[model] = {"calibration_chunks": len(texts), "max_abs_difference": difference,
                         "threshold_decisions_changed": flipped}
    return checks


# ----- The evaluation (identical for the dry run and the sealed run) -----
def _serving_parity(pipeline: ContractInferencePipeline, contracts: list[Contract], chunks, probabilities,
                    clauses) -> int:
    """
    Score every contract again through ContractInferencePipeline (what the API serves) and require
    the same chunks, probabilities and detected clauses. Returns the number of contracts checked.
    """
    def key(record):
        evidence = getattr(record, "evidence", None)
        return (record.chunk_index, record.category, None if evidence is None else (evidence.start, evidence.end))

    rows_of = defaultdict(list)
    for row, chunk in enumerate(chunks):
        rows_of[chunk.contract_id].append(row)
    expected = defaultdict(list)
    for clause in clauses:
        expected[clause.contract_id].append(key(clause))
    for contract in contracts:
        served = pipeline.predict_contract(contract.contract_id, contract.text)
        rows = rows_of[contract.contract_id]
        spans = [(c["start"], c["end"]) for c in served["chunks"]]
        served_probabilities = np.asarray([[c["probabilities"][name] for name in CATEGORIES] for c in served["chunks"]])
        if spans != [(chunks[r].start, chunks[r].end) for r in rows]:
            raise IntegrityError(f"serving chunks differ for contract #{contracts.index(contract)}")
        if not np.array_equal(served_probabilities, probabilities[rows].astype(np.float64)):
            raise IntegrityError(f"serving probabilities differ for contract #{contracts.index(contract)}")
        if sorted(map(key, [*served["grounded"], *served["unknown"]]), key=repr) != sorted(expected[contract.contract_id], key=repr):
            raise IntegrityError(f"serving clauses differ for contract #{contracts.index(contract)}")
    return len(contracts)


def _evidence_summary(clauses, contracts: list[Contract], chunks, y_true: np.ndarray, detections: int) -> dict:
    """Unknown rate, and how often a grounded quote overlaps a gold answer of the same category."""
    row_of = {(chunk.contract_id, chunk.chunk_index): row for row, chunk in enumerate(chunks)}
    gold = defaultdict(list)
    for contract in contracts:
        for span in contract.spans:
            gold[contract.contract_id, span.category].append((span.start, span.end))
    grounded = [c for c in clauses if c.evidence is not None]
    hits = [any(spans_overlap(c.evidence.start, c.evidence.end, s, e) for s, e in gold[c.contract_id, c.category])
            for c in grounded]
    on_true = [hit for c, hit in zip(grounded, hits)
               if y_true[row_of[c.contract_id, c.chunk_index], CATEGORY_TO_INDEX[c.category]] == 1]
    unknown = len(clauses) - len(grounded)
    return {
        "chunk_category_detections": int(detections),
        "clauses_after_deduplication": len(clauses),
        "grounded": len(grounded), "unknown": unknown,
        "unknown_rate": unknown / len(clauses) if clauses else float("nan"),
        "grounded_quote_overlaps_gold": float(np.mean(hits)) if hits else float("nan"),
        "grounded_on_true_positive_chunks": len(on_true),
        "grounded_quote_overlaps_gold_on_true_positive_chunks": float(np.mean(on_true)) if on_true else float("nan"),
    }


def _span_containment(contracts: list[Contract], chunks) -> dict:
    """How many gold answer spans fit inside a single chunk (the rest straddle chunk edges)."""
    by_contract = defaultdict(list)
    for chunk in chunks:
        by_contract[chunk.contract_id].append(chunk)
    spans = [(contract.contract_id, span) for contract in contracts for span in contract.spans]
    inside = sum(any(c.start <= span.start and span.end <= c.end for c in by_contract[cid]) for cid, span in spans)
    return {"gold_spans": len(spans), "inside_one_chunk": int(inside),
            "share_inside_one_chunk": inside / len(spans) if spans else float("nan")}


def evaluate(contracts: list[Contract], bundle: ClauseModelBundle, word_predict, tokenizer, *,
             replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> dict:
    """
    Run the registered analyses on `contracts` and return report sections and tables. Primary: chunk-level
    detection of the frozen bundle at its own thresholds. Secondary: the word-only reference, contract-level
    presence, triage ranking against the perfect-detection reference, and evidence grounding.
    """
    timings = {}
    # Chunk and label every contract exactly as training did
    start = time.perf_counter()
    labeled = prepare_contracts(contracts, tokenizer, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS)
    texts, y_true = to_model_inputs(labeled)
    chunks = [item.chunk for item in labeled]
    timings["chunking_seconds"] = time.perf_counter() - start

    # Raw probabilities from both models (the bundle's own predict_proba)
    start = time.perf_counter()
    probabilities = {"word_char_tfidf": bundle.predict_proba(texts)}
    timings["bundle_predict_seconds"] = time.perf_counter() - start
    probabilities["word_tfidf"] = word_predict(texts)
    thresholds = {"word_char_tfidf": _vector(bundle.thresholds),
                  "word_tfidf": _vector(_thresholds(SAVED_RUN / "word_tfidf_thresholds.json"))}

    # Detected clauses with evidence, then the same contracts through the serving pipeline
    weight_set = load_weight_set(DEFAULT_WEIGHTS_PATH)
    if (weight_set.weight_version_id, weight_set.content_sha256) != (WEIGHT_VERSION_ID, WEIGHT_SET_SHA256):
        raise IntegrityError("the default weight set is not the pinned v2-team-proposed version")
    cues = load_category_cues()
    contract_texts = {c.contract_id: c.text for c in contracts}
    clauses = clauses_from_probabilities(contract_texts, chunks, probabilities["word_char_tfidf"], bundle.thresholds, cues)
    start = time.perf_counter()
    parity = _serving_parity(ContractInferencePipeline(bundle, tokenizer, cues, max_tokens=MAX_TOKENS,
                                                       overlap_tokens=OVERLAP_TOKENS, weight_set=weight_set),
                             contracts, chunks, probabilities["word_char_tfidf"], clauses)
    timings["serving_parity_seconds"] = time.perf_counter() - start

    # Point estimates: chunk-level tables for both models, contract-level presence for the bundle
    tables = {model: detection_table(y_true, probabilities[model], thresholds[model]) for model in MODELS}
    order, groups = contract_groups([chunk.contract_id for chunk in chunks])
    if order != [c.contract_id for c in contracts]:
        raise IntegrityError("chunk order does not follow contract order")
    gold_presence, presence_score = contract_presence(y_true, probabilities["word_char_tfidf"], groups)
    contract_table = detection_table(gold_presence, presence_score, thresholds["word_char_tfidf"])

    # Triage ranking against perfect detection, under the pinned weights
    ranked = aggregate_contracts(score_contract_clauses(clauses, weight_set), weight_set, order)
    gold_ranked = {c.contract_id: c for c in aggregate_contracts(
        score_contract_clauses(gold_clauses(contracts, chunks), weight_set), weight_set, order)}
    predicted_rank = pd.Series({c.contract_id: c.rank for c in ranked})
    gold_rank = pd.Series({cid: c.rank for cid, c in gold_ranked.items()})
    rules = {
        "triage_score": predicted_rank,
        "high_count": pd.Series({c.contract_id: -c.high_count for c in ranked}).rank(method="min"),
        "worst_clause": pd.Series({c.contract_id: -c.top_clauses[0].composite_score if c.top_clauses else 0.0
                                   for c in ranked}).rank(method="min"),
    }
    agreement = {name: _agreement(ranks, gold_rank) for name, ranks in rules.items()}
    predicted_score = np.asarray([next(c.triage_score for c in ranked if c.contract_id == cid) for cid in order])
    gold_score = np.asarray([gold_ranked[cid].triage_score for cid in order])

    # Contract bootstrap: one set of draws for every statistic, so model differences are paired
    per_contract_counts = {model: [np.stack(parts) for parts in zip(*(
        category_counts(y_true[rows], predict_at_thresholds(probabilities[model][rows], thresholds[model]))
        for rows in groups))] for model in MODELS}

    def statistic(selected: np.ndarray) -> dict:
        """All bootstrapped statistics for one resample of contracts."""
        rows = np.concatenate([groups[index] for index in selected])
        values = {}
        for model in MODELS:
            f1 = f1_from_counts(*(counts[selected].sum(axis=0) for counts in per_contract_counts[model]))
            aupr = per_category_aupr(y_true[rows], probabilities[model][rows])
            values[f"{model}/macro_f1"], values[f"{model}/macro_aupr"] = macro_f1(f1), macro_aupr(aupr)
            if model == "word_char_tfidf":
                values["word_char_tfidf/f1"], values["word_char_tfidf/aupr"] = f1, aupr
        values["delta/macro_f1"] = values["word_char_tfidf/macro_f1"] - values["word_tfidf/macro_f1"]
        values["delta/macro_aupr"] = values["word_char_tfidf/macro_aupr"] - values["word_tfidf/macro_aupr"]
        gold, score = gold_presence[selected], presence_score[selected]
        contract_f1 = f1_from_counts(*category_counts(gold, predict_at_thresholds(score, thresholds["word_char_tfidf"])))
        contract_aupr = per_category_aupr(gold, score)
        values["contract/f1"], values["contract/aupr"] = contract_f1, contract_aupr
        values["contract/macro_f1"], values["contract/macro_aupr"] = macro_f1(contract_f1), macro_aupr(contract_aupr)
        values["triage/spearman"] = spearman(predicted_score[selected], gold_score[selected])
        return values

    start = time.perf_counter()
    draws = bootstrap_replicates(len(groups), replicates, seed, statistic)
    point = statistic(np.arange(len(groups)))
    jackknife = jackknife_replicates(len(groups), statistic)
    timings["bootstrap_seconds"] = time.perf_counter() - start
    if not np.isclose(point["word_char_tfidf/macro_f1"], macro_f1(tables["word_char_tfidf"]["f1"].to_numpy())):
        raise IntegrityError("the bootstrap statistic does not reproduce the point estimate")

    def interval(key: str, column: int | None = None) -> dict:
        """BCa interval (with percentile bounds) for one bootstrapped statistic, or one category's column of it."""
        if column is None:
            return bca_interval(draws[key], float(point[key]), jackknife[key])
        return bca_interval(draws[key][:, column], float(point[key][column]), jackknife[key][:, column])

    # Per-category tables with intervals
    for table, prefix in ((tables["word_char_tfidf"], "word_char_tfidf"), (contract_table, "contract")):
        for metric in ("f1", "aupr"):
            bounds = [interval(f"{prefix}/{metric}", i) for i in range(len(CATEGORIES))]
            table[f"{metric}_lower_95"] = [b["lower_95"] for b in bounds]
            table[f"{metric}_upper_95"] = [b["upper_95"] for b in bounds]
            table[f"{metric}_interval_method"] = [b["method"] for b in bounds]
            if metric == "aupr":
                table["aupr_replicates_defined"] = [b["replicates_defined"] for b in bounds]

    macro = {}
    for model in MODELS:
        table = tables[model]
        macro[model] = {
            "macro_f1": macro_f1(table["f1"].to_numpy()), "macro_f1_95": interval(f"{model}/macro_f1"),
            "macro_aupr": macro_aupr(table["aupr"].to_numpy()), "macro_aupr_95": interval(f"{model}/macro_aupr"),
            "categories_with_positives": int((table["support"] > 0).sum()),
            "macro_f1_over_categories_with_positives": float(table.loc[table["support"] > 0, "f1"].mean()),
        }
    macro["word_char_minus_word"] = {
        "macro_f1": macro["word_char_tfidf"]["macro_f1"] - macro["word_tfidf"]["macro_f1"],
        "macro_f1_95": interval("delta/macro_f1"),
        "macro_aupr": macro["word_char_tfidf"]["macro_aupr"] - macro["word_tfidf"]["macro_aupr"],
        "macro_aupr_95": interval("delta/macro_aupr"),
    }
    contract_level = {"macro_f1": macro_f1(contract_table["f1"].to_numpy()), "macro_f1_95": interval("contract/macro_f1"),
                      "macro_aupr": macro_aupr(contract_table["aupr"].to_numpy()),
                      "macro_aupr_95": interval("contract/macro_aupr"),
                      "categories_with_positives": int((contract_table["support"] > 0).sum())}
    triage = {"contracts": len(order), "weight_version_id": weight_set.weight_version_id,
              "weight_set_sha256": weight_set.content_sha256, "top_n": TOP_N,
              "agreement_with_perfect_detection": agreement,
              "triage_score_spearman_95": interval("triage/spearman"),
              "spearman_from_scores": spearman(predicted_score, gold_score),
              "distinct_triage_scores": len(set(predicted_score.tolist()))}
    ranking = pd.DataFrame([{
        "rank": c.rank, "contract_id": c.contract_id, "triage_score": c.triage_score,
        "high": c.high_count, "medium": c.medium_count, "low": c.low_count, "unknown": c.unknown_count,
        "top_clauses": "; ".join(f"{s.category} ({s.composite_score:.2f} {s.rating})" for s in c.top_clauses),
        "gold_rank": gold_ranked[c.contract_id].rank, "gold_triage_score": gold_ranked[c.contract_id].triage_score,
    } for c in ranked])
    detections = int(predict_at_thresholds(probabilities["word_char_tfidf"], thresholds["word_char_tfidf"]).sum())
    return {
        "contracts": contracts, "chunks": chunks, "y_true": y_true, "probabilities": probabilities,
        "thresholds": thresholds, "tables": tables, "contract_table": contract_table,
        "contract_presence": gold_presence, "ranking": ranking,
        "macro": macro, "contract_level": contract_level, "triage": triage,
        "evidence": _evidence_summary(clauses, contracts, chunks, y_true, detections),
        "data_description": {"contracts": len(contracts), "chunks": len(chunks),
                             "positive_chunk_labels": int(y_true.sum()),
                             "span_containment": _span_containment(contracts, chunks)},
        "serving_parity_contracts": parity, "timings": timings,
        "bootstrap": {"replicates": replicates, "seed": seed, "unit": "contract",
                      "method": "BCa 95% (jackknife over contracts for the acceleration; percentile bounds kept "
                                "alongside), numpy default_rng(seed).integers(0, n, size=n) per replicate, "
                                "all statistics on the same draws"},
    }


# ----- Output files -----
def write_outputs(output: Path, result: dict, report: dict, categories: pd.DataFrame, buckets: pd.DataFrame) -> None:
    """
    Render summary.md and report.json in memory first (so a formatting error can't leave partial
    results), then write every table and the raw probabilities, and report.json last: its presence
    marks a completed run.
    """
    summary = render_summary(report, categories, buckets)
    report_text = json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    categories.to_csv(output / "per_category_metrics.csv", index=False)
    buckets.to_csv(output / "bucket_summary.csv", index=False)
    result["contract_table"].to_csv(output / "contract_level_metrics.csv", index=False)
    result["ranking"].to_csv(output / "triage_ranking.csv", index=False)
    pd.DataFrame([{"contract_id": c.contract_id, "chunk_index": c.chunk_index, "start": c.start, "end": c.end}
                  for c in result["chunks"]]).to_csv(output / "chunks.csv", index=False)
    np.savez_compressed(output / "predictions.npz", y_true=result["y_true"],
                        **{model: result["probabilities"][model] for model in MODELS})
    (output / "summary.md").write_text(summary, encoding="utf-8")
    (output / REPORT_FILE).write_text(report_text, encoding="utf-8")


def development_reproduction(result: dict) -> dict | None:
    """
    Dry-run evidence only: when the data is exactly the seed-42 development holdout, compare labels,
    probabilities and macro metrics with the saved run, and triage agreement with results/contract_triage.
    Returns None for any other data (so it never applies to the sealed run).
    """
    ids = [c.contract_id for c in result["contracts"]]
    if ids_sha256(ids) != PARTITION_DIGESTS["evaluation"]:
        return None
    comparison = {}
    for model in MODELS:
        saved = np.load(SAVED_RUN / f"{model}_evaluation.npz")
        vector = _vector(_thresholds(SAVED_RUN / f"{model}_thresholds.json"))
        saved_table = detection_table(saved["y_true"], saved["y_prob"], vector)
        current = result["probabilities"][model]
        comparison[model] = {
            "labels_identical": bool(np.array_equal(saved["y_true"], result["y_true"])),
            "max_abs_probability_difference": float(np.max(np.abs(current.astype(np.float64) - saved["y_prob"]))),
            "threshold_decisions_changed": int(np.sum(predict_at_thresholds(current, vector)
                                                      != predict_at_thresholds(saved["y_prob"], vector))),
            "saved_macro_f1": macro_f1(saved_table["f1"].to_numpy()),
            "saved_macro_aupr": macro_aupr(saved_table["aupr"].to_numpy()),
            "this_run_macro_f1": result["macro"][model]["macro_f1"],
            "this_run_macro_aupr": result["macro"][model]["macro_aupr"],
        }
    saved_triage = json.loads((ROOT / "results" / "contract_triage" / "report.json").read_text(encoding="utf-8"))
    comparison["triage_agreement_saved"] = saved_triage["agreement_with_gold"]
    comparison["triage_agreement_this_run"] = result["triage"]["agreement_with_perfect_detection"]
    return comparison


def provenance_record(*, mode: str, bundle: ClauseModelBundle, git: dict, integrity: dict, reference: dict,
                      word_training_chunks: int) -> dict:
    """
    Everything about the run that doesn't depend on the data: protocol and model hashes, settings,
    environment, the development reference and the pre-data checks. Built before the attempt starts,
    so a failure here can't happen after the data has been read.
    """
    return {
        "experiment": "final_test_evaluation", "mode": mode, "test_data_used": mode == "sealed_test",
        "protocol": {"path": _shown(PROTOCOL_PATH), "sha256": text_file_sha256(PROTOCOL_PATH),
                     "last_commit": last_commit_of(ROOT, PROTOCOL_PATH)},
        "git": git, "environment": environment(),
        "model": {"path": _shown(BUNDLE_PATH), "sha256": file_sha256(BUNDLE_PATH), "pinned_sha256": BUNDLE_SHA256,
                  "model_name": bundle.model_name, "config": bundle.config, "seed": bundle.seed,
                  "thresholds": _shown(SAVED_RUN / "word_char_tfidf_thresholds.json"),
                  "thresholds_sha256": text_file_sha256(SAVED_RUN / "word_char_tfidf_thresholds.json")},
        "word_only_reference_model": {"fitted_on": "seed-42 training partition (260 contracts)",
                                      "training_chunks": word_training_chunks,
                                      "settings": "src/models/naive/tfid_baseline.py defaults",
                                      "thresholds": _shown(SAVED_RUN / "word_tfidf_thresholds.json"),
                                      "thresholds_sha256": text_file_sha256(SAVED_RUN / "word_tfidf_thresholds.json")},
        "chunking": {"method": "chunk_sentence_windows", "tokenizer": TOKENIZER_NAME,
                     "tokenizer_revision": TOKENIZER_REVISION, "max_tokens": MAX_TOKENS,
                     "overlap_tokens": OVERLAP_TOKENS, "label_projection": "any character overlap (spans_overlap)"},
        "weights": {"weight_version_id": WEIGHT_VERSION_ID, "weight_set_sha256": WEIGHT_SET_SHA256},
        "metric_definitions": {
            "prediction": "probability >= category threshold (float64), as ClauseModelBundle.predict",
            "macro_f1": "mean of per-category F1 over all 41 categories; F1 = 0 when undefined",
            "macro_aupr": "mean of per-category average precision over categories with at least one positive",
            "support_status": "unscorable (0 positives), insufficient support (<10), reported (10+)",
        },
        "development_reference": {"source": "results/word_char_tfidf_nested[_seed*] saved probabilities and thresholds",
                                  "per_seed": reference["per_seed"], "macro": reference["macro"]},
        "integrity": integrity,
        "disclosures": DISCLOSURES,
    }


def build_report(*, provenance: dict, data_path: Path, data_sha256: str, result: dict, reference: dict,
                 categories: pd.DataFrame, attempts: list[dict]) -> dict:
    """Assemble report.json: the provenance record plus primary and secondary results, attempts and deviations."""
    primary = dict(result["macro"]["word_char_tfidf"])
    primary["unscorable_categories"] = categories.loc[categories["status"] == "unscorable", "category"].tolist()
    primary["insufficient_support_categories"] = categories.loc[categories["status"] == "insufficient support",
                                                                "category"].tolist()
    primary["against_development"] = {
        metric: compare_with_reference(primary[metric], primary[f"{metric}_95"]["lower_95"],
                                       primary[f"{metric}_95"]["upper_95"],
                                       **{key: reference["macro"][f"word_char_tfidf_{metric}"][key2]
                                          for key, key2 in (("reference", "mean"), ("reference_min", "min"),
                                                            ("reference_max", "max"))})
        for metric in ("macro_f1", "macro_aupr")}
    deviations = [f"attempt {a['attempt']} resumed after a crash: {a['resume_reason']}"
                  for a in attempts if a.get("resume_reason")]
    return {
        **provenance, "status": "complete",
        "data": {"path": _shown(data_path), "sha256": data_sha256,
                 "contract_id_sha256": ids_sha256([c.contract_id for c in result["contracts"]])},
        "bootstrap": result["bootstrap"],
        "data_description": result["data_description"],
        "primary": primary,
        "secondary": {
            "word_only_reference": {**result["macro"]["word_tfidf"],
                                    "word_char_minus_word": result["macro"]["word_char_minus_word"]},
            "contract_level": result["contract_level"],
            "triage": result["triage"],
            "evidence": result["evidence"],
            "inference_timings": result["timings"],
        },
        "integrity": {**provenance["integrity"], "serving_parity_contracts": result["serving_parity_contracts"]},
        "development_reproduction": development_reproduction(result),
        "attempts": attempts, "deviations": deviations,
    }


# ----- Command line -----
def write_development_holdout(path: Path) -> int:
    """Write the seed-42 evaluation partition of the training file as CUAD JSON (original entries, split order)."""
    raw = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    evaluation = training_partitions()["evaluation"]
    by_title = {entry["title"]: entry for entry in raw["data"]}
    if len(by_title) != len(raw["data"]):
        raise IntegrityError("training-file titles are not unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": raw.get("version"), "data": [by_title[c.contract_id] for c in evaluation]}),
                    encoding="utf-8")
    return len(evaluation)


def run_preflight(data_path: Path, output: Path, sealed: bool) -> dict:
    """Structure-only checks of the data file; writes preflight.json. Computes no prediction."""
    partitions = training_partitions()
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    structure = structure_report(raw, partitions)
    try:
        load_cuad(data_path)
        loader_error = None
    except Exception as error:  # the message could quote contract text, so only the type is kept
        loader_error = type(error).__name__
    blob = git_blob_id(ROOT, data_path) if sealed else None
    failures = preflight_failures(structure, expected_contracts=EXPECTED_TEST_CONTRACTS if sealed else None,
                                  loader_error=loader_error, blob_matches=(blob == SEALED_DATA_GIT_BLOB) if sealed else None)
    record = {"passed": not failures, "failures": failures, "sealed": sealed,
              "data": {"path": _shown(data_path), "sha256": file_sha256(data_path),
                       "bytes": data_path.stat().st_size, "git_blob": blob,
                       "pinned_git_blob": SEALED_DATA_GIT_BLOB if sealed else None},
              "structure": structure, "git": git_state(ROOT), "disclosures": DISCLOSURES}
    output.mkdir(parents=True, exist_ok=True)
    (output / PREFLIGHT_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, help="CUAD-format JSON to evaluate")
    parser.add_argument("--output", type=Path, help="Folder for preflight.json, the attempt log and results")
    parser.add_argument("--preflight", action="store_true", help="Structure-only checks; no predictions")
    parser.add_argument("--sealed-test-run", action="store_true",
                        help="Opt in to reading data/cuad/test.json (the one registered evaluation)")
    parser.add_argument("--resume-after-crash", metavar="REASON",
                        help="Continue after an attempt that crashed before producing any metric; recorded as a deviation")
    parser.add_argument("--replicates", type=int, help="Bootstrap replicates (development runs and tests only)")
    parser.add_argument("--write-development-holdout", type=Path, metavar="PATH",
                        help="Write the seed-42 development holdout as CUAD JSON and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the preflight, the development dry run or the sealed evaluation; returns a process exit code."""
    args = _parser().parse_args(argv)
    if args.write_development_holdout:
        if classify_data_path(args.write_development_holdout) == "sealed":
            raise SealedRunError("the development holdout must not be named test.json")
        print(f"Wrote {write_development_holdout(args.write_development_holdout)} development contracts "
              f"to {args.write_development_holdout}")
        return 0
    if args.data is None or args.output is None:
        _parser().error("--data and --output are required")

    # Guards, all before the data file is opened
    data_kind = classify_data_path(args.data)
    sealed = data_kind == "sealed"
    check_mode(data_kind, args.sealed_test_run)
    output = args.output if args.output.is_absolute() else (Path.cwd() / args.output)
    git = git_state(ROOT)
    if sealed:
        if args.data.resolve() != SEALED_DATA_PATH.resolve():
            raise SealedRunError(f"the sealed run reads only {_shown(SEALED_DATA_PATH)}")
        check_sealed_output(output, SEALED_OUTPUT)
        if args.replicates is not None:
            raise SealedRunError("the sealed run uses the registered number of bootstrap replicates")
        check_sealed_preconditions(output=output, git=git, report_in_history=git_history_has(ROOT, output / REPORT_FILE),
                                   protocol_tracked=git_tracked(ROOT, PROTOCOL_PATH),
                                   preflight_tracked=git_tracked(ROOT, output / PREFLIGHT_FILE) if not args.preflight else False,
                                   data_sha256=None, preflight_only=args.preflight)
    else:
        if output.resolve() == SEALED_OUTPUT.resolve():
            raise SealedRunError("development runs never write to the sealed results folder")
        if (output / REPORT_FILE).exists():
            raise SealedRunError(f"{output / REPORT_FILE} exists; use a new output folder")
    if args.preflight:
        if (output / ATTEMPTS_FILE).exists():
            raise SealedRunError("the evaluation has already started; the preflight comes before it")
        record = run_preflight(args.data, output, sealed)
        print(json.dumps({"passed": record["passed"], "failures": record["failures"]}, indent=2))
        return 0 if record["passed"] else 1
    preflight_path = output / PREFLIGHT_FILE
    if not preflight_path.is_file() or not json.loads(preflight_path.read_text(encoding="utf-8")).get("passed"):
        raise SealedRunError("run a passing --preflight into this output folder first")

    # Pre-data integrity: pinned inputs, the frozen bundle, the training partitions, and reproduction of saved
    # calibration scores
    pinned_hashes = check_pinned_files()
    bundle = load_frozen_bundle()
    tokenizer = load_tokenizer()
    partitions = training_partitions()
    word_predict, word_training_chunks = fit_word_reference(partitions["training"], tokenizer)
    integrity = {"pinned_files_sha256": pinned_hashes, "bundle_sha256_matches": True,
                 "bundle_thresholds_match_saved": True,
                 "training_file_sha256_matches": True, "partition_digests_match": True,
                 "calibration_reproduction": check_reproduction(
                     {"word_char_tfidf": bundle.predict_proba, "word_tfidf": word_predict},
                     partitions["calibration"], tokenizer)}
    reference = development_reference()
    mode = "sealed_test" if sealed else "development_dry_run"
    provenance = provenance_record(mode=mode, bundle=bundle, git=git, integrity=integrity, reference=reference,
                                   word_training_chunks=word_training_chunks)

    # The run starts here: log the attempt, then read the data
    attempts = start_attempt(output, commit=git["commit"], mode=mode, resume_reason=args.resume_after_crash)
    data_sha256 = file_sha256(args.data)
    if data_sha256 != json.loads(preflight_path.read_text(encoding="utf-8"))["data"]["sha256"]:
        raise SealedRunError("the data file changed since the preflight")
    contracts = load_cuad(args.data)
    result = evaluate(contracts, bundle, word_predict, tokenizer,
                      replicates=args.replicates or BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)

    # Tables and report, written in full before anything about the metrics is printed
    categories = per_category_frame(result, reference)
    buckets = bucket_frame(categories)
    report = build_report(provenance=provenance, data_path=args.data, data_sha256=data_sha256, result=result,
                          reference=reference, categories=categories, attempts=attempts)
    write_outputs(output, result, report, categories, buckets)
    print((output / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SealedRunError as error:
        print(f"Refused: {error}", file=sys.stderr)
        sys.exit(2)
