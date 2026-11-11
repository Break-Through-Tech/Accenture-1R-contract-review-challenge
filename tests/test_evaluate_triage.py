import pandas as pd
import pytest

from scripts.evaluate_triage import _reweighted, _top_overlap, _variants
from src.risk.schema import load_weight_set

WEIGHTS = load_weight_set()


# ----- Weight variants -----
@pytest.mark.parametrize("delta", [-0.1, 0.1])
def test_reweighted_moves_one_signal_and_keeps_the_sum(delta):
    """One weight moves by delta, the rest keep their proportions, and the total stays 1."""
    moved = _reweighted(WEIGHTS, "magnitude", delta)
    assert moved.signal_weights["magnitude"] == pytest.approx(WEIGHTS.signal_weights["magnitude"] + delta)
    assert sum(moved.signal_weights.values()) == pytest.approx(1.0)
    ratio = moved.signal_weights["one_sidedness"] / moved.signal_weights["contract_context"]
    assert ratio == pytest.approx(WEIGHTS.signal_weights["one_sidedness"] / WEIGHTS.signal_weights["contract_context"])


def test_variants_have_unique_versions():
    """Every variant gets its own weight_version_id, so sensitivity() can tell them apart."""
    ids = [v.weight_version_id for v in _variants(WEIGHTS)]
    assert len(ids) == len(set(ids)) and WEIGHTS.weight_version_id not in ids


# ----- Top-N overlap -----
def test_top_overlap_ties_cannot_inflate_the_count():
    """A ranking with everything tied still contributes exactly 10 contracts, chosen by id."""
    ids = [f"c{i:02d}" for i in range(20)]
    all_tied = pd.Series(1, index=ids)
    reversed_order = pd.Series(range(20, 0, -1), index=ids)  # c19 first
    assert _top_overlap(all_tied, all_tied) == 10
    assert _top_overlap(all_tied, reversed_order) == 0  # c00-c09 vs c10-c19
