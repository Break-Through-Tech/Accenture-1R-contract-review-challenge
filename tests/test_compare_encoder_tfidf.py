import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.compare_encoder_tfidf import (
    PairingError,
    compare,
    compare_seed,
    contract_rows_from_chunks,
    load_baseline,
    load_candidate,
    main,
    summarize_categories,
    support_bucket,
    verdict,
    verify_pairing,
    write_outputs,
)
from scripts.evaluate_encoder_nested import baseline_dir_for
from src.labels.categories import CATEGORIES

N_CATEGORIES = len(CATEGORIES)
DIGESTS = {"training": "a" * 64, "calibration": "b" * 64, "evaluation": "c" * 64}

# Synthetic seed: 4 contracts x 2 chunks; category 0 is positive in the first chunk of each contract.
# Candidate: perfect on category 0 -> F1 1, AUPR 1.
# Baseline: flags rows 0 (positive) and 1 (negative) -> TP 1, FP 1, FN 3 -> F1 2/6 = 1/3;
# AUPR = 0.25 * 0.5 (top pair) + 0.75 * 0.5 (all rows) = 0.5.
GOLDEN_DELTA_F1 = (1 - 1 / 3) / N_CATEGORIES
GOLDEN_DELTA_AUPR = 0.5


def synthetic_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (labels, candidate scores, baseline scores, chunk table) for the synthetic seed."""
    y_true = np.zeros((8, N_CATEGORIES), dtype=np.int8)
    y_true[[0, 2, 4, 6], 0] = 1
    candidate = np.where(y_true == 1, 0.9, 0.1).astype(np.float32)
    baseline = np.full((8, N_CATEGORIES), 0.1, dtype=np.float32)
    baseline[[0, 1], 0] = 0.9
    chunks = pd.DataFrame({"contract_id": [f"K{i // 2}" for i in range(8)], "chunk_index": [0, 1] * 4,
                           "start": [0, 10] * 4, "end": [10, 20] * 4})
    return y_true, candidate, baseline, chunks


def write_pair(root: Path, seed: int, candidate_scores: np.ndarray | None = None,
               baseline_scores: np.ndarray | None = None, candidate_report: dict | None = None) -> None:
    """Write one candidate run and its paired word_char_tfidf baseline run for a seed under root."""
    y_true, candidate, baseline, chunks = synthetic_arrays()
    candidate = candidate if candidate_scores is None else candidate_scores
    baseline = baseline if baseline_scores is None else baseline_scores
    report = {"data_sha256": "d" * 64, "test_data_used": False,
              "split": {"seed": seed, "contract_id_sha256": DIGESTS}}
    thresholds = {category: 0.5 for category in CATEGORIES}

    # Candidate in the encoder runner's layout
    run_dir = root / f"cand_seed{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(json.dumps(candidate_report or report))
    (run_dir / "thresholds.json").write_text(json.dumps(thresholds))
    np.savez_compressed(run_dir / "evaluation_predictions.npz", y_true=y_true, y_prob=candidate)
    chunks.to_csv(run_dir / "evaluation_chunks.csv", index=False)

    # Baseline in the saved TF-IDF layout
    baseline_dir = baseline_dir_for(root / "results", seed)
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "report.json").write_text(json.dumps(report))
    (baseline_dir / "word_char_tfidf_thresholds.json").write_text(json.dumps(thresholds))
    np.savez_compressed(baseline_dir / "word_char_tfidf_evaluation.npz", y_true=y_true, y_prob=baseline)


def run_compare(root: Path, seeds=(1, 2), **kwargs) -> dict:
    """Compare the synthetic candidate with its baseline for the given seeds under root."""
    return compare(seeds=seeds, candidate_template=str(root / "cand_seed{seed}"),
                   baseline_root=root / "results", replicates=200, expected_seeds=(1, 2), **kwargs)


# ----- support buckets -----
@pytest.mark.parametrize("support, bucket", [
    (0, "rare"), (9.9, "rare"), (10, "uncommon"), (29.9, "uncommon"), (30, "common"), (102, "common"),
])
def test_support_bucket_uses_error_analysis_cut_offs(support, bucket):
    """Checks the rare <10, uncommon 10-29, common 30+ buckets of scripts/analyze_errors.py."""
    assert support_bucket(support) == bucket


# ----- contract groups -----
def test_contract_rows_from_chunks_groups_rows_by_contract_in_first_seen_order():
    """Checks rows are grouped per contract, contracts ordered by first appearance."""
    chunks = pd.DataFrame({"contract_id": ["b", "b", "a", "a", "c"]})
    rows = contract_rows_from_chunks(chunks)

    assert [group.tolist() for group in rows] == [[0, 1], [2, 3], [4]]


# ----- per-seed comparison -----
def test_compare_seed_golden_macro_deltas():
    """Checks the hand-computed macro-F1 delta (2/3)/41 and macro-AUPR delta 0.5."""
    y_true, candidate, baseline, chunks = synthetic_arrays()
    thresholds = {category: 0.5 for category in CATEGORIES}
    seed_row, category_rows, _ = compare_seed(
        1, {"y_true": y_true, "scores": candidate, "thresholds": thresholds},
        {"y_true": y_true, "scores": baseline, "thresholds": thresholds},
        contract_rows_from_chunks(chunks), replicates=50)

    assert seed_row["candidate_macro_f1"] == pytest.approx(1 / N_CATEGORIES)
    assert seed_row["baseline_macro_f1"] == pytest.approx((1 / 3) / N_CATEGORIES)
    assert seed_row["delta_macro_f1"] == pytest.approx(GOLDEN_DELTA_F1)
    assert seed_row["delta_macro_aupr"] == pytest.approx(GOLDEN_DELTA_AUPR)
    # Per-category F1 comes from compute_base_metrics, which rounds to 5 decimals
    assert category_rows[0]["delta_f1"] == pytest.approx(2 / 3, abs=1e-5)
    assert len(category_rows) == N_CATEGORIES


def test_compare_seed_bootstrap_interval_is_positive_when_candidate_always_wins():
    """Checks every resample favours the perfect candidate, so the F1 interval sits above zero."""
    y_true, candidate, baseline, chunks = synthetic_arrays()
    thresholds = {category: 0.5 for category in CATEGORIES}
    seed_row, _, interval = compare_seed(
        1, {"y_true": y_true, "scores": candidate, "thresholds": thresholds},
        {"y_true": y_true, "scores": baseline, "thresholds": thresholds},
        contract_rows_from_chunks(chunks), replicates=200)

    assert 0 < interval["macro_f1_delta"]["lower_95"] <= interval["macro_f1_delta"]["upper_95"] <= 1 / N_CATEGORIES
    assert seed_row["delta_macro_f1_lower_95"] == interval["macro_f1_delta"]["lower_95"]


def test_compare_seed_identical_runs_give_zero_deltas_and_zero_interval():
    """Checks comparing a run with itself yields exactly zero everywhere."""
    y_true, candidate, _, chunks = synthetic_arrays()
    run = {"y_true": y_true, "scores": candidate, "thresholds": {category: 0.5 for category in CATEGORIES}}
    seed_row, _, interval = compare_seed(1, run, run, contract_rows_from_chunks(chunks), replicates=50)

    assert seed_row["delta_macro_f1"] == 0 and seed_row["delta_macro_aupr"] == 0
    assert interval["macro_f1_delta"] == {"lower_95": 0.0, "median": 0.0, "upper_95": 0.0}


# ----- pairing checks -----
def make_runs() -> tuple[dict, dict]:
    """Return a correctly paired (candidate, baseline) dict pair for seed 1."""
    y_true, candidate, baseline, _ = synthetic_arrays()
    report = {"data_sha256": "d" * 64, "test_data_used": False, "split": {"seed": 1, "contract_id_sha256": DIGESTS}}
    return ({"report": json.loads(json.dumps(report)), "y_true": y_true},
            {"report": json.loads(json.dumps(report)), "y_true": y_true.copy()})


def test_verify_pairing_matching_runs_pass():
    """Checks identical data hash, digests, seed and labels raise nothing."""
    verify_pairing(1, *make_runs())


@pytest.mark.parametrize("mutate, message", [
    (lambda c, b: c["report"].update(test_data_used=None), "test_data_used"),
    (lambda c, b: b["report"].update(data_sha256="e" * 64), "data hashes"),
    (lambda c, b: c["report"]["split"].update(seed=2), "report seeds"),
    (lambda c, b: c["report"]["split"]["contract_id_sha256"].update(evaluation="0" * 64), "evaluation contract-ID"),
    (lambda c, b: c["report"].pop("split"), "training contract-ID"),
    (lambda c, b: b["y_true"].__setitem__((1, 5), 1), "label matrices"),
])
def test_verify_pairing_any_mismatch_raises(mutate, message):
    """Checks each way two runs can be unpaired is refused and named."""
    candidate, baseline = make_runs()
    mutate(candidate, baseline)
    with pytest.raises(PairingError, match=message):
        verify_pairing(1, candidate, baseline)


# ----- category summary -----
def test_summarize_categories_ranks_supported_and_marks_rare_as_insufficient():
    """Checks B (delta 0.2) ranks 1, A (0.1) ranks 2, and rare C (0.5) is unranked despite the largest gain."""
    rows = []
    for seed in (1, 2):
        for category, support, delta in (("A", 40, 0.1), ("B", 15, 0.2), ("C", 3, 0.5)):
            rows.append({"seed": seed, "category": category, "support": support, "baseline_f1": 0.5,
                         "candidate_f1": 0.5 + delta, "delta_f1": delta, "delta_aupr": delta / 2})
    summary = summarize_categories(pd.DataFrame(rows)).set_index("category")

    assert summary.loc["A", "support_bucket"] == "common" and summary.loc["A", "rank_by_delta_f1"] == 2
    assert summary.loc["B", "support_bucket"] == "uncommon" and summary.loc["B", "rank_by_delta_f1"] == 1
    assert summary.loc["C", "status"] == "insufficient support"
    assert pd.isna(summary.loc["C", "rank_by_delta_f1"])
    assert summary.loc["A", "f1_wins"] == 2
    assert list(summary.index) == ["A", "B", "C"]


# ----- verdict -----
def seed_frame(deltas, lowers, aupr_deltas, seeds=(42, 7, 17, 23, 101)) -> pd.DataFrame:
    """Build a per-seed frame with the columns verdict reads."""
    return pd.DataFrame({"seed": list(seeds), "delta_macro_f1": deltas,
                         "delta_macro_f1_lower_95": lowers, "delta_macro_aupr": aupr_deltas})


@pytest.mark.parametrize("frame, expected", [
    # Wins all five, interval above zero on 4 of 5, AUPR up
    (seed_frame([.06, .05, .07, .04, .05], [.02, .01, .03, -.01, .01], [.03] * 5), "clearly_beats"),
    # Wins all five but interval above zero on only 3 of 5
    (seed_frame([.03, .02, .03, .02, .03], [.01, -.01, .01, -.02, .01], [.03] * 5), "not_clearly"),
    # Strong F1 but AUPR worse on average
    (seed_frame([.06, .05, .07, .04, .05], [.02] * 5, [-.01] * 5), "not_clearly"),
    # Loses one seed (the reranker pattern)
    (seed_frame([.024, .015, .026, -.012, .025], [-.01] * 5, [0] * 5), "not_clearly"),
    # Wins only two seeds
    (seed_frame([.05, .05, -.01, -.01, -.01], [0] * 5, [.01] * 5), "does_not_beat"),
    # Negative on average
    (seed_frame([-.02] * 5, [-.05] * 5, [-.01] * 5), "does_not_beat"),
    # Only three seeds run
    (seed_frame([.06, .05, .07], [.02] * 3, [.03] * 3, seeds=(42, 7, 17)), "incomplete"),
])
def test_verdict_applies_the_predeclared_rule(frame, expected):
    """Checks each verdict outcome on hand-built per-seed results."""
    assert verdict(frame)["verdict"] == expected


# ----- end to end on synthetic saved runs -----
def test_compare_synthetic_runs_reports_golden_deltas_and_clear_win(tmp_path):
    """Checks two synthetic seeds give the golden deltas, 2 wins and a clearly_beats verdict."""
    for seed in (1, 2):
        write_pair(tmp_path, seed)
    result = run_compare(tmp_path)

    assert result["per_seed"]["delta_macro_f1"].tolist() == pytest.approx([GOLDEN_DELTA_F1] * 2)
    assert result["aggregate"]["delta_macro_aupr"]["mean"] == pytest.approx(GOLDEN_DELTA_AUPR)
    assert result["aggregate"]["macro_f1_wins"] == 2
    assert result["verdict"]["verdict"] == "clearly_beats"


def test_compare_writes_tables_json_and_markdown(tmp_path):
    """Checks every output file is written and comparison.json parses with the verdict inside."""
    for seed in (1, 2):
        write_pair(tmp_path, seed)
    output = tmp_path / "out"

    write_outputs(run_compare(tmp_path), output)

    for name in ("per_seed.csv", "per_category_by_seed.csv", "per_category_summary.csv",
                 "support_bucket_summary.csv", "comparison.json", "summary.md"):
        assert (output / name).is_file()
    assert json.loads((output / "comparison.json").read_text())["verdict"]["verdict"] == "clearly_beats"
    assert "clearly_beats" in (output / "summary.md").read_text()


def test_compare_missing_seed_raises_unless_allowed(tmp_path):
    """Checks an unrun seed fails by default and gives an incomplete verdict with allow_missing."""
    write_pair(tmp_path, 1)

    with pytest.raises(FileNotFoundError):
        run_compare(tmp_path)
    result = run_compare(tmp_path, allow_missing=True)
    assert result["seeds_missing"] == [2]
    assert result["verdict"]["verdict"] == "incomplete"


def test_compare_unpaired_split_raises(tmp_path):
    """Checks a candidate whose evaluation digest differs from the baseline is refused."""
    bad = {"data_sha256": "d" * 64, "test_data_used": False,
           "split": {"seed": 1, "contract_id_sha256": {**DIGESTS, "evaluation": "f" * 64}}}
    write_pair(tmp_path, 1, candidate_report=bad)

    with pytest.raises(PairingError, match="evaluation contract-ID"):
        run_compare(tmp_path, seeds=(1,))


def test_load_candidate_threshold_key_reads_nested_thresholds(tmp_path):
    """Checks --threshold-key and --score-key adapt runs that store several models (reranker layout)."""
    y_true, candidate, _, _ = synthetic_arrays()
    (tmp_path / "report.json").write_text("{}")
    (tmp_path / "thresholds.json").write_text(json.dumps({"reranker": {c: 0.3 for c in CATEGORIES}}))
    np.savez_compressed(tmp_path / "evaluation_predictions.npz", y_true=y_true, reranker_score=candidate)

    loaded = load_candidate(tmp_path, score_key="reranker_score", threshold_key="reranker")

    assert loaded["thresholds"][CATEGORIES[0]] == 0.3
    assert loaded["chunks"] is None
    assert np.array_equal(loaded["scores"], candidate)


# ----- real saved runs -----
def test_load_baseline_reads_saved_seed42_word_char_run():
    """Checks the committed seed-42 TF-IDF run loads with 2,196 evaluation rows and 41 thresholds."""
    root = Path(__file__).resolve().parents[1] / "results"
    baseline = load_baseline(baseline_dir_for(root, 42), "word_char_tfidf")

    assert baseline["y_true"].shape == baseline["scores"].shape == (2196, N_CATEGORIES)
    assert list(baseline["thresholds"]) == CATEGORIES


def test_main_writes_one_folder_per_baseline_model(tmp_path, capsys):
    """Checks the command line runs and writes vs_word_char_tfidf/ for the synthetic pair."""
    for seed in (42, 7, 17, 23, 101):
        write_pair(tmp_path, seed)

    main(["--candidate-template", str(tmp_path / "cand_seed{seed}"), "--baseline-root", str(tmp_path / "results"),
          "--baseline-models", "word_char_tfidf", "--bootstrap-replicates", "20", "--output", str(tmp_path / "out")])

    assert (tmp_path / "out" / "vs_word_char_tfidf" / "comparison.json").is_file()
    assert "Verdict: clearly_beats" in capsys.readouterr().out
