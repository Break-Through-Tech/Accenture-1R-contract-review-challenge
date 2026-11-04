from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.chunking.chunk import Chunk
from src.models.encoder.sparse_long_context import (
    LocalGlobalSparseLayer,
    SparseLongContextClassifier,
    overlap_consensus_contract_scores,
    sparse_attention_edge_count,
)


def _chunk(contract, index, start, end):
    text = "x" * (end - start)
    return SimpleNamespace(chunk=Chunk(
        contract_id=contract, start=start, end=end, text=text, chunk_index=index,
    ))


def test_sparse_attention_layer_has_finite_shapes_and_gradients():
    layer = LocalGlobalSparseLayer(hidden_size=16, heads=4, radius=2, dropout=0)
    nodes = torch.randn(7, 16, requires_grad=True)
    global_state = torch.randn(16, requires_grad=True)

    contextual, updated_global = layer(nodes, global_state)
    (contextual.sum() + updated_global.sum()).backward()

    assert contextual.shape == nodes.shape
    assert updated_global.shape == global_state.shape
    assert torch.isfinite(contextual).all()
    assert nodes.grad is not None and global_state.grad is not None


def test_sparse_classifier_handles_complete_variable_length_contract_sequences():
    model = SparseLongContextClassifier(
        input_size=12, num_labels=5, hidden_size=16, layers=2,
        heads=4, local_radius=1, max_chunks=20, dropout=0,
    )

    short = model(torch.randn(1, 12))
    long = model(torch.randn(13, 12))

    assert short.shape == (1, 5)
    assert long.shape == (13, 5)
    assert torch.isfinite(long).all()
    with pytest.raises(ValueError, match="1..20 chunks"):
        model(torch.randn(21, 12))


def test_global_token_propagates_information_beyond_local_window():
    torch.manual_seed(4)
    model = SparseLongContextClassifier(
        input_size=8, num_labels=2, hidden_size=16, layers=2,
        heads=4, local_radius=0, max_chunks=10, dropout=0,
    ).eval()
    original = torch.zeros((4, 8))
    changed = original.clone()
    changed[0, 0] = 3.0

    with torch.no_grad():
        first = model(original)[-1]
        second = model(changed)[-1]

    assert not torch.allclose(first, second)


def test_sparse_attention_link_count_scales_linearly_with_sequence_length():
    assert sparse_attention_edge_count(3, local_radius=1) == 14
    assert sparse_attention_edge_count(100, local_radius=4) < 100 * 100
    with pytest.raises(ValueError):
        sparse_attention_edge_count(0, 1)


def test_overlap_consensus_averages_duplicate_window_scores():
    chunks = [_chunk("a", 0, 0, 10), _chunk("a", 1, 0, 10), _chunk("b", 0, 0, 5)]
    scores = np.zeros((3, 2), dtype=np.float32)
    scores[:, 0] = [0.9, 0.1, 0.6]
    scores[:, 1] = [0.2, 0.4, 0.8]

    aggregate = overlap_consensus_contract_scores(chunks, scores)

    np.testing.assert_allclose(aggregate["a"], [0.5, 0.3])
    np.testing.assert_allclose(aggregate["b"], [0.6, 0.8])
