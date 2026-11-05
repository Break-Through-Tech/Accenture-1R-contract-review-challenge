import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.labels.categories import CATEGORIES
from scripts.analyze_reranker_multiseed import _macro_f1
from scripts.train_retrieval_reranker import _metrics_at_thresholds, _resolve_contract_groups


def test_explicit_split_manifest_preserves_contract_order_and_hash(tmp_path):
    contracts = [SimpleNamespace(contract_id=f"C{index}") for index in range(4)]
    manifest_path = tmp_path / "split.json"
    manifest = {
        "training": ["C2"], "calibration": ["C0"], "evaluation": ["C3"],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    groups, ids, digest = _resolve_contract_groups(
        contracts, seed=42, evaluation_fraction=0.2, calibration_fraction=0.2,
        split_manifest=manifest_path,
    )

    assert ids == manifest
    assert {name: [item.contract_id for item in group] for name, group in groups.items()} == manifest
    assert digest == hashlib.sha256(manifest_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("manifest,match", [
    ({"training": ["C0"], "calibration": ["C0"], "evaluation": ["C1"]}, "overlap"),
    ({"training": ["C0"], "calibration": ["missing"], "evaluation": ["C1"]}, "unknown"),
    ({"training": ["C0", "C0"], "calibration": ["C1"], "evaluation": ["C2"]}, "unique"),
])
def test_explicit_split_manifest_rejects_invalid_contract_groups(tmp_path, manifest, match):
    contracts = [SimpleNamespace(contract_id=f"C{index}") for index in range(3)]
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _resolve_contract_groups(
            contracts, seed=42, evaluation_fraction=0.2, calibration_fraction=0.2,
            split_manifest=manifest_path,
        )


def test_fixed_category_macro_f1_counts_unsupported_classes_as_zero():
    y_true = np.zeros((1, len(CATEGORIES)), dtype=np.uint8)
    scores = np.zeros_like(y_true, dtype=np.float32)
    y_true[0, 0] = 1
    scores[0, 0] = 1.0

    assert _macro_f1(y_true, scores, np.full(len(CATEGORIES), 0.5)) == 1 / len(CATEGORIES)


def test_evaluation_metric_alignment_uses_calibration_thresholds_and_label_names():
    y_true = np.zeros((4, len(CATEGORIES)), dtype=np.uint8)
    y_score = np.zeros_like(y_true, dtype=np.float32)
    y_true[:, 0] = [1, 1, 0, 0]
    y_score[:, 0] = [0.9, 0.8, 0.2, 0.1]
    y_true[:, 1] = [0, 0, 1, 0]
    y_score[:, 1] = [0.1, 0.2, 0.9, 0.1]
    thresholds = {category: 0.5 for category in CATEGORIES}

    metrics = _metrics_at_thresholds(y_true, y_score, thresholds).set_index("category")

    assert metrics.loc[CATEGORIES[0], "support"] == 2
    assert metrics.loc[CATEGORIES[1], "support"] == 1
    assert metrics.loc[CATEGORIES[0], "aupr"] == 1.0
    assert metrics.loc[CATEGORIES[1], "aupr"] == 1.0
    assert np.isnan(metrics.loc[CATEGORIES[2], "aupr"])
    assert metrics.loc[CATEGORIES[2], "support"] == 0
    assert metrics.loc[CATEGORIES[2], "f1"] == 0.0
    assert metrics.loc[CATEGORIES[0], "f1"] == 1.0
    assert metrics.loc[CATEGORIES[1], "f1"] == 1.0
