from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.chunking.chunk import Chunk
from src.models.encoder.contract_graph import (
    ContractGraphClassifier,
    WeightedGraphSAGELayer,
    build_contract_graphs,
)


class FakeEncoder(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embedding = nn.Embedding(64, hidden)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _item(contract, index, text, label=0):
    return SimpleNamespace(
        chunk=Chunk(
            contract_id=contract, start=0, end=len(text), text=text, chunk_index=index,
        ),
        labels=[label] + [0] * 40,
    )


def test_graphs_have_bidirectional_adjacent_and_nonlocal_lexical_edges():
    chunks = [
        _item("a", 0, "common anchor phrase agreement"),
        _item("a", 1, "middle unrelated clause"),
        _item("a", 2, "common anchor phrase agreement"),
        _item("a", 3, "last separate condition"),
        _item("b", 0, "common anchor phrase agreement"),
    ]
    train_texts = [item.chunk.text for item in chunks] + ["common anchor phrase agreement"]

    _, graphs = build_contract_graphs(train_texts, chunks, semantic_neighbors=2)

    graph = graphs["a"]
    edges = {
        (int(source), int(target)): float(weight)
        for (source, target), weight in zip(graph.edge_index.T.tolist(), graph.edge_weight.tolist())
    }
    assert edges[0, 1] == edges[1, 0] == 1.0
    assert edges[1, 2] == edges[2, 1] == 1.0
    assert edges[0, 2] > 0
    assert graph.edge_index.shape[0] == 2
    assert graphs["b"].edge_index.shape == (2, 0)


def test_graph_construction_is_label_blind_and_deterministic():
    chunks = [
        _item("a", 0, "shared language here", 0),
        _item("a", 1, "other clause there", 1),
        _item("a", 2, "shared language here", 0),
    ]
    changed = [_item("a", i, item.chunk.text, 1 - item.labels[0]) for i, item in enumerate(chunks)]
    corpus = [item.chunk.text for item in chunks] * 2

    _, first = build_contract_graphs(corpus, chunks, semantic_neighbors=2)
    _, second = build_contract_graphs(corpus, changed, semantic_neighbors=2)

    assert torch.equal(first["a"].edge_index, second["a"].edge_index)
    torch.testing.assert_close(first["a"].edge_weight, second["a"].edge_weight)


def test_graph_classifier_produces_contract_chunk_logits_and_backpropagates():
    model = ContractGraphClassifier(
        FakeEncoder(), num_labels=3, graph_hidden_size=6,
        graph_layers=2, encoder_chunk_batch_size=2, dropout=0,
    )
    input_ids = torch.tensor([[[1, 2, 0], [3, 4, 5], [6, 7, 0]]])
    attention_mask = input_ids.ne(0).long()
    chunk_mask = torch.tensor([[True, True, True]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = torch.ones(4)
    labels = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

    logits, loss = model(
        input_ids, attention_mask, chunk_mask, edge_index, edge_weight,
        labels=labels, pos_weight=torch.ones(3),
    )
    loss.backward()

    assert logits.shape == (3, 3)
    assert torch.isfinite(loss)
    assert model.encoder.embedding.weight.grad is not None
    assert model.graph_layers[0].neighbor_projection.weight.grad is not None


def test_graph_classifier_rejects_multiple_contracts_in_one_graph():
    model = ContractGraphClassifier(FakeEncoder(), num_labels=2, graph_hidden_size=4)
    with pytest.raises(ValueError, match="exactly one contract"):
        model(
            torch.ones((2, 1, 3), dtype=torch.long),
            torch.ones((2, 1, 3), dtype=torch.long),
            torch.ones((2, 1), dtype=torch.bool),
            torch.empty((2, 0), dtype=torch.long), torch.empty((0,)),
        )


def test_weighted_graphsage_handles_isolated_nodes():
    layer = WeightedGraphSAGELayer(hidden_size=4, dropout=0)
    nodes = torch.randn(2, 4)

    output = layer(nodes, torch.empty((2, 0), dtype=torch.long), torch.empty((0,)))

    assert output.shape == nodes.shape
    assert torch.isfinite(output).all()
