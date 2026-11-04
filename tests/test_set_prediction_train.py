from src.labels.loader import AnswerSpan, Contract
from src.models.encoder.set_prediction import ContractSetScores, PredictedSpan
from scripts.train_set_prediction import calibrate_span_thresholds


def test_span_thresholds_are_fitted_to_calibration_iou_f1():
    contract = Contract(
        contract_id="doc", text="abcdefghij",
        spans=[AnswerSpan(category="Agreement Date", start=2, end=4, text="cd")],
    )
    predictions = {
        "doc": ContractSetScores(
            contract_presence=[0.9] * 41,
            chunk_scores=[],
            spans=(
                PredictedSpan(2, 0.6, 0, 0, 2, 4),
                PredictedSpan(2, 0.4, 1, 1, 6, 8),
            ),
        ),
    }

    thresholds = calibrate_span_thresholds([contract], predictions, iou_threshold=0.5)

    assert thresholds["Agreement Date"] == 0.6
    assert thresholds["Affiliate License-Licensee"] == 1.0
