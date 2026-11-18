import json

import numpy as np
import pandas as pd
import pytest

from clausetriage.holdout_report import bucket_frame, fmt, jsonable


# ----- Number formatting -----
@pytest.mark.parametrize("value, interval, expected", [
    (0.5123, None, "0.512"),
    (0.5123, {"lower_95": 0.47, "upper_95": 0.5534}, "0.512 [0.470, 0.553]"),
    (float("nan"), None, "n/a"),
    (None, None, "n/a"),
    (np.float64(0.25), {"lower_95": float("nan"), "upper_95": 0.3}, "0.250 [n/a, 0.300]"),
])
def test_fmt_golden_strings(value, interval, expected):
    """Three decimals, optional bracketed interval, and n/a for missing values."""
    assert fmt(value, interval) == expected


# ----- Strict JSON -----
def test_jsonable_converts_numpy_values_and_nan_recursively():
    """numpy scalars become Python numbers, NaN becomes None, tuples become lists; the result is strict JSON."""
    value = {"a": np.int64(3), "b": [np.float32(0.5), float("nan")], "c": (np.bool_(True), {"d": np.nan})}
    converted = jsonable(value)
    assert converted == {"a": 3, "b": [0.5, None], "c": [True, {"d": None}]}
    json.dumps(converted, allow_nan=False)


# ----- Bucket summary -----
def test_bucket_frame_golden_means_and_order():
    """Buckets come out common, uncommon, rare; means skip NaN AUPR; flags are counted per bucket."""
    categories = pd.DataFrame({
        "category": ["A", "B", "C", "D"], "dev_bucket": ["rare", "common", "common", "uncommon"],
        "support": [0, 40, 30, 12], "status": ["unscorable", "reported", "reported", "reported"],
        "f1": [0.0, 0.8, 0.6, 0.5], "dev_mean_f1": [0.2, 0.7, 0.5, 0.4],
        "aupr": [np.nan, 0.9, 0.7, 0.6], "dev_mean_aupr": [0.3, 0.8, 0.6, 0.5],
    })
    buckets = bucket_frame(categories)
    assert buckets["dev_bucket"].tolist() == ["common", "uncommon", "rare"]
    assert buckets["test_mean_f1"].tolist() == pytest.approx([0.7, 0.5, 0.0])
    assert buckets["test_mean_aupr"].iloc[:2].tolist() == pytest.approx([0.8, 0.6])
    assert np.isnan(buckets["test_mean_aupr"].iloc[2])
    assert buckets["unscorable"].tolist() == [0, 0, 1]
    assert buckets["test_support"].tolist() == [70, 12, 0]
