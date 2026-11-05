import numpy as np

from src.labels.categories import CATEGORIES
from scripts.analyze_reranker_multiseed import _macro_f1
from scripts.train_retrieval_reranker import _metrics_at_thresholds


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
