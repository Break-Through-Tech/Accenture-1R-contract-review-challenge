"""Tests for scripts/estimate_deferred_reading.py on a hand-built results folder (no CUAD data)."""
import numpy as np
import pandas as pd
import pytest

from scripts import estimate_deferred_reading as estimate


@pytest.fixture
def results_dir(tmp_path):
    """Two contracts, two categories: contract a has one flagged window of two, contract b one of two."""
    # Windows: a covers characters 0 to 100 with an overlap, b covers 0 to 80
    pd.DataFrame({
        "contract_id": ["a", "a", "b", "b"],
        "chunk_index": [0, 1, 0, 1],
        "start": [0, 40, 0, 50],
        "end": [60, 100, 50, 80],
    }).to_csv(tmp_path / "chunks.csv", index=False)
    # a0 detects c1 (true), a1 misses c2, b1 detects c2 (false)
    y_true = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.int8)
    probabilities = np.array([[0.9, 0.1], [0.2, 0.3], [0.1, 0.1], [0.1, 0.7]], dtype=np.float32)
    np.savez(tmp_path / "predictions.npz", y_true=y_true, word_char_tfidf=probabilities)
    pd.DataFrame({"category": ["c1", "c2"], "threshold": [0.5, 0.5],
                  "tp": [1, 0], "fp": [0, 1], "fn": [0, 1]}).to_csv(tmp_path / "per_category_metrics.csv", index=False)
    pd.DataFrame({"category": ["c1", "c2"], "tp": [1, 0], "fp": [0, 1], "fn": [0, 1]}).to_csv(
        tmp_path / "contract_level_metrics.csv", index=False)
    return tmp_path


# Merging flagged windows
@pytest.mark.parametrize("intervals, expected", [
    ([], 0),
    ([(0, 10)], 10),
    ([(0, 10), (5, 20)], 20),
    ([(10, 20), (0, 5)], 15),
    ([(0, 10), (10, 20)], 20),
])
def test_merged_length_counts_overlap_once(intervals, expected):
    """Overlapping and touching windows are counted once; gaps are not counted."""
    assert estimate.merged_length(intervals) == expected


# Measured inputs on the synthetic folder
def test_deferred_reading_matches_hand_computed_values(results_dir):
    """a defers 40 of 100 characters and b 50 of 80; one of two labels lies in deferred text."""
    values = estimate.deferred_reading(results_dir)
    assert values["d_median"] == pytest.approx(0.5125)
    assert values["d_lower_quartile"] == pytest.approx(0.45625)
    assert values["d_upper_quartile"] == pytest.approx(0.56875)
    assert values["d_pooled"] == pytest.approx(0.5)
    assert (values["labels"], values["labels_deferred"], values["m"]) == (2, 1, 0.5)
    assert (values["detections"], values["false_detections"], values["contracts_without_flags"]) == (2, 1, 0)
    assert (values["contract_presences"], values["contract_presences_missed"]) == (2, 1)
    assert values["median_contract_characters"] == 90.0


def test_deferred_reading_mismatched_counts_exits(results_dir):
    """Saved TP counts that the probabilities do not reproduce stop the script."""
    metrics = pd.read_csv(results_dir / "per_category_metrics.csv")
    metrics.loc[0, "tp"] = 2
    metrics.to_csv(results_dir / "per_category_metrics.csv", index=False)
    with pytest.raises(SystemExit, match="recomputed tp"):
        estimate.deferred_reading(results_dir)


def test_deferred_reading_refuses_cuad_data_folder():
    """A results path inside data/cuad is refused before any file is opened."""
    with pytest.raises(SystemExit, match="refusing"):
        estimate.deferred_reading(estimate.FORBIDDEN_DATA_DIR / "anything")


# Arithmetic and output
def test_deferred_minutes_golden_value():
    """Half of 300 words at 150 words per minute is one minute."""
    assert estimate.deferred_minutes(0.5, 300.0, 150.0) == pytest.approx(1.0)


def test_main_prints_measured_inputs(results_dir, capsys):
    """The command line prints d, m and the grid for the synthetic folder."""
    estimate.main(["--results", str(results_dir)])
    output = capsys.readouterr().out
    assert "d median 51.2%, quartiles 45.6% to 56.9%, pooled 50.0%" in output
    assert "m 50.0% (1 of 2 labels)" in output
