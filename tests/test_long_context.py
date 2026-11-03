from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.chunking.chunk import Chunk
from src.labels.projection import LabeledChunk
from src.models.encoder.contract_dataset import ContractChunkDataset, collate_contract_sequences
from src.models.encoder.category_query import CategoryQueryClassifier
from src.models.encoder.long_context import LongContextClassifier
from src.registry.predictors import CategoryQueryPredictor, LongContextPredictor


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, truncation, max_length, padding):
        rows = [[1] + [min(len(word), 30) for word in text.split()][:max_length - 1] for text in texts]
        return {"input_ids": [row[:max_length] for row in rows]}


def _item(contract_id, index, text, positive=False):
    chunk = Chunk(contract_id=contract_id, start=0, end=len(text), text=text, chunk_index=index)
    labels = [0] * 41
    labels[0] = int(positive)
    return LabeledChunk(chunk=chunk, labels=labels)


def test_contract_dataset_groups_and_orders_chunks():
    chunks = [_item("b", 0, "other contract"), _item("a", 1, "second chunk", True),
              _item("a", 0, "first chunk")]
    dataset = ContractChunkDataset(chunks, FakeTokenizer(), max_length=8)

    assert len(dataset) == 2
    assert dataset[0]["contract_id"] == "a"
    assert dataset[0]["chunks"][0].chunk_index == 0
    assert dataset[0]["chunks"][1].chunk_index == 1
    assert dataset[0]["labels"][1][0] == 1


def test_contract_collator_pads_chunks_and_tokens_with_masks():
    dataset = ContractChunkDataset(
        [_item("a", 0, "one two three"), _item("a", 1, "four"), _item("b", 0, "one")],
        FakeTokenizer(), max_length=8,
    )
    batch = collate_contract_sequences([dataset[0], dataset[1]], pad_token_id=0)

    assert batch["input_ids"].shape == (2, 2, 4)
    assert batch["chunk_mask"].tolist() == [[True, True], [True, False]]
    assert batch["attention_mask"][1, 1].sum().item() == 0
    assert batch["labels"].shape == (2, 2, 41)


def test_long_context_classifier_returns_masked_chunk_logits_and_trainable_loss():
    model = LongContextClassifier(FakeEncoder(), hidden_size=8, num_labels=41, dropout=0)
    input_ids = torch.tensor([[[1, 2, 0], [3, 4, 5]], [[6, 7, 8], [0, 0, 0]]])
    attention = input_ids.ne(0).long()
    chunk_mask = torch.tensor([[True, True], [True, False]])
    labels = torch.zeros((2, 2, 41))
    labels[0, 1, 3] = 1
    output = model(input_ids, attention, chunk_mask, labels=labels,
                   pos_weight=torch.ones(41))

    assert output.logits.shape == (2, 2, 41)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.encoder.embedding.weight.grad is not None
    assert model.classifier.weight.grad is not None


def test_long_context_classifier_empty_predictions_have_expected_shape():
    model = LongContextClassifier(FakeEncoder(), hidden_size=8, num_labels=41)
    output = model(torch.tensor([[[1, 2]]]), torch.ones((1, 1, 2), dtype=torch.long),
                   torch.ones((1, 1), dtype=torch.bool))

    assert output.logits.shape == (1, 1, 41)
    assert output.loss is None


def test_inter_chunk_transformer_aggregates_context_and_masks_padded_chunk():
    model = LongContextClassifier(
        FakeEncoder(), hidden_size=8, num_labels=41, dropout=0,
        context_type="transformer", context_layers=2, attention_heads=2,
    )
    input_ids = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [0, 0]]])
    attention = input_ids.ne(0).long()
    chunk_mask = torch.tensor([[True, True], [True, False]])
    labels = torch.zeros((2, 2, 41))
    output = model(input_ids, attention, chunk_mask, labels=labels)

    assert output.logits.shape == (2, 2, 41)
    assert torch.isfinite(output.logits).all()
    assert output.loss is not None


def test_category_query_classifier_attends_to_contract_chunks_and_trains():
    model = CategoryQueryClassifier(
        FakeEncoder(), torch.randn(41, 8), dropout=0,
        context_layers=1, attention_heads=2,
    )
    input_ids = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [0, 0]]])
    attention_mask = input_ids.ne(0).long()
    chunk_mask = torch.tensor([[True, True], [True, False]])
    labels = torch.zeros((2, 2, 41))
    labels[0, 1, 3] = 1
    labels[1, 0, 5] = 1

    output = model(input_ids, attention_mask, chunk_mask, labels,
                   chunk_pos_weight=torch.ones(41),
                   contract_pos_weight=torch.ones(41))

    assert output.logits.shape == (2, 2, 41)
    assert output.contract_logits.shape == (2, 41)
    assert output.attention_weights.shape == (2, 41, 2)
    assert torch.allclose(output.attention_weights.sum(-1), torch.ones((2, 41)))
    assert torch.all(output.attention_weights[1, :, 1] == 0)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.label_queries.grad is not None
    assert model.encoder.embedding.weight.grad is not None


def test_category_query_predictor_returns_probabilities_and_evidence_attention():
    model = CategoryQueryClassifier(
        FakeEncoder(), torch.randn(41, 8), dropout=0,
        context_layers=1, attention_heads=2,
    )
    predictor = CategoryQueryPredictor(model, FakeTokenizer(), max_length=8)
    result = predictor.predict_with_evidence(["clause one", "second clause"])

    assert result["probabilities"].shape == (2, 41)
    assert result["attention_weights"].shape == (2, 41)
    assert result["top_evidence_chunk_indices"].shape == (41,)
    assert np.allclose(result["attention_weights"].sum(axis=0), 1)
    assert predictor.predict_proba([]).shape == (0, 41)


def test_long_context_registry_predictor_scores_one_contract_sequence():
    model = LongContextClassifier(FakeEncoder(), hidden_size=8, num_labels=41)
    predictor = LongContextPredictor(model, FakeTokenizer(), max_length=8)

    scores = predictor.predict_proba(["clause one", "second clause"])

    assert scores.shape == (2, 41)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert predictor.predict_proba([]).shape == (0, 41)
