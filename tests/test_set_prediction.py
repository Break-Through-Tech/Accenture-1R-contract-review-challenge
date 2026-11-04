import torch

from src.chunking.chunk import Chunk
from src.labels.loader import AnswerSpan, Contract
from src.models.encoder.set_prediction import (
    ContractSetPredictor,
    GoldChunkSpan,
    SetPredictionOutput,
    decode_contract_set,
    map_gold_spans_to_chunks,
    matched_set_loss,
    span_iou,
)


class FakeLabeledChunk:
    def __init__(self, contract_id, index, start, text):
        self.chunk = Chunk(
            contract_id=contract_id, start=start, end=start + len(text),
            text=text, chunk_index=index,
        )
        self.labels = [0] * 41


def test_gold_answers_map_to_overlapping_chunks_without_losing_source_offsets():
    text = "abcdefghij"
    contract = Contract(
        contract_id="doc", text=text,
        spans=[AnswerSpan(category="Effective Date", start=2, end=8, text=text[2:8])],
    )
    chunks = [
        FakeLabeledChunk("doc", 0, 0, text[0:4]),
        FakeLabeledChunk("doc", 1, 3, text[3:7]),
        FakeLabeledChunk("doc", 2, 6, text[6:10]),
    ]

    spans, uncovered = map_gold_spans_to_chunks(contract, chunks)

    assert uncovered == 0
    assert spans == [GoldChunkSpan(
        category_index=10, start_chunk=0, end_chunk=2, char_start=2, char_end=8,
    )]


def test_set_predictor_emits_one_slot_and_two_pointer_distributions_per_category():
    model = ContractSetPredictor(
        hidden_size=16, num_labels=3, max_slots=2, max_chunks=8,
        attention_heads=4, dropout=0,
    )

    output = model(torch.randn(5, 16))

    assert output.presence_logits.shape == (3, 2)
    assert output.start_logits.shape == output.end_logits.shape == (3, 2, 5)


def test_hungarian_matched_set_loss_trains_slots_and_counts_overflow():
    presence = torch.zeros((2, 2), requires_grad=True)
    start = torch.randn((2, 2, 4), requires_grad=True)
    end = torch.randn((2, 2, 4), requires_grad=True)
    output = SetPredictionOutput(presence, start, end)
    gold = [
        GoldChunkSpan(0, 0, 1, 1, 4),
        GoldChunkSpan(0, 1, 2, 3, 6),
        GoldChunkSpan(0, 2, 3, 5, 8),
    ]

    loss, stats = matched_set_loss(output, gold, max_slots=2)
    loss.backward()

    assert torch.isfinite(loss)
    assert stats == {"matched_spans": 2, "overflow_spans": 1}
    assert presence.grad is not None and presence.grad.abs().sum() > 0
    assert start.grad is not None and start.grad.abs().sum() > 0
    assert end.grad is not None and end.grad.abs().sum() > 0


def test_decode_maps_predicted_chunk_interval_to_source_offsets_and_chunk_scores():
    presence = torch.full((41, 2), -5.0)
    presence[0, 0] = 5.0
    start = torch.zeros((41, 2, 3))
    end = torch.zeros((41, 2, 3))
    start[0, 0, 2] = 5.0
    end[0, 0, 1] = 5.0
    chunks = [
        FakeLabeledChunk("doc", 0, 0, "abcd"),
        FakeLabeledChunk("doc", 1, 3, "defg"),
        FakeLabeledChunk("doc", 2, 6, "ghij"),
    ]

    result = decode_contract_set(SetPredictionOutput(presence, start, end), chunks)

    span = next(item for item in result.spans if item.category_index == 0 and item.score > 0.9)
    assert (span.start_chunk, span.end_chunk) == (1, 2)
    assert (span.char_start, span.char_end) == (3, 10)
    assert result.chunk_scores[1, 0] == result.chunk_scores[2, 0]
    assert result.chunk_scores[0, 0] < result.chunk_scores[1, 0]


def test_character_span_iou_uses_end_exclusive_offsets():
    assert span_iou(2, 8, 4, 10) == 4 / 8
    assert span_iou(0, 4, 4, 8) == 0.0
