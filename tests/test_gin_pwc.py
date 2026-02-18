"""Simple tests for GIN-PWC: pair helpers, model forward, selector save/load and algorithm_select."""

import tempfile
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data

from src.gin_ehm import NodeVocabulary
from src.gin_pwc import (
    GINPwc,
    GINPwcBackbone,
    GINPwcSelector,
    all_pairs,
    idx_to_pair,
    num_pairs,
    pair_to_idx,
)


def test_pair_index_helpers():
    """num_pairs, pair_to_idx, idx_to_pair, all_pairs roundtrip."""
    for K in [2, 3, 4, 5, 10]:
        n = num_pairs(K)
        assert n == K * (K - 1) // 2
        pairs = all_pairs(K)
        assert len(pairs) == n
        for idx, (i, j) in enumerate(pairs):
            assert 0 <= i < j < K
            assert pair_to_idx(i, j, K) == idx
            assert idx_to_pair(idx, K) == (i, j)


def test_gin_pwc_backbone_forward():
    """Backbone produces (batch_size, hidden_dim) from batched PyG Data."""
    num_node_types = 4
    hidden_dim = 8
    num_layers = 2
    backbone = GINPwcBackbone(
        num_node_types=num_node_types,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
    )
    # Two tiny graphs: 3 nodes and 2 nodes
    x1 = torch.tensor([[1], [2], [1]], dtype=torch.long)
    edge1 = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long).t()
    x2 = torch.tensor([[2], [1]], dtype=torch.long)
    edge2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
    d1 = Data(x=x1, edge_index=edge1)
    d2 = Data(x=x2, edge_index=edge2)
    batch = Batch.from_data_list([d1, d2])
    g = backbone.forward_data(batch)
    assert g.shape == (2, hidden_dim)


def test_gin_pwc_forward():
    """GINPwc full forward and forward_batch_for_loss."""
    num_node_types = 4
    K = 3
    hidden_dim = 8
    num_layers = 2
    model = GINPwc(
        num_node_types=num_node_types,
        num_solvers=K,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
    )
    # Batch of 2 graphs
    x1 = torch.tensor([[1], [2], [1]], dtype=torch.long)
    edge1 = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long).t()
    d1 = Data(x=x1, edge_index=edge1)
    d2 = Data(x=x1, edge_index=edge1)  # same structure
    batch = Batch.from_data_list([d1, d2])
    logits = model.forward_data(batch)
    assert logits.shape == (2, num_pairs(K))
    # forward_batch_for_loss: pair indices for each sample
    pair_idx = torch.tensor([0, 1], dtype=torch.long)  # first sample pair 0, second pair 1
    logits_selected = model.forward_batch_for_loss(batch, pair_idx)
    assert logits_selected.shape == (2,)
    assert logits_selected[0].item() == logits[0, 0].item()
    assert logits_selected[1].item() == logits[1, 1].item()


def test_gin_pwc_selector_save_load():
    """GINPwcSelector save and load roundtrip."""
    num_node_types = 4
    K = 3
    model = GINPwc(
        num_node_types=num_node_types,
        num_solvers=K,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
    )
    vocab = NodeVocabulary()
    for t in ["a", "b", "c", "d"]:
        vocab.add_type(t)
    vocab.freeze()
    selector = GINPwcSelector(
        model=model,
        vocabulary=vocab,
        fallback_solver_ids=[0, 1, 2],
        graph_timeout=5,
        random_seed=123,
    )
    with tempfile.TemporaryDirectory() as tmp:
        save_dir = Path(tmp) / "gin_pwc"
        selector.save(save_dir)
        assert (save_dir / "config.json").exists()
        assert (save_dir / "model.pt").exists()
        assert (save_dir / "vocab.json").exists()
        loaded = GINPwcSelector.load(save_dir)
        assert loaded.model.num_solvers == K
        assert loaded._K == K
        assert loaded.fallback_solver_ids == [0, 1, 2]
        assert loaded.random_seed == 123


def test_gin_pwc_selector_algorithm_select():
    """GINPwcSelector.algorithm_select on a minimal SMT file returns a solver id."""
    num_node_types = 10
    K = 2
    model = GINPwc(
        num_node_types=num_node_types,
        num_solvers=K,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
    )
    # Use real vocab from a minimal graph
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smt2", delete=False
    ) as f:
        f.write(
            "(set-logic QF_LIA)\n"
            "(declare-const x Int)\n"
            "(assert (>= x 0))\n"
            "(check-sat)\n"
        )
        smt_path = Path(f.name)
    try:
        from src.gin_ehm import build_vocabulary_from_graph_dicts
        from src.graph_rep import smt_to_graph

        graph_dict = smt_to_graph(smt_path)
        vocab = build_vocabulary_from_graph_dicts([graph_dict])
        selector = GINPwcSelector(
            model=model,
            vocabulary=vocab,
            fallback_solver_ids=[0, 1],
            graph_timeout=5,
            random_seed=42,
        )
        solver_id = selector.algorithm_select(smt_path)
        assert solver_id in (0, 1)
        # Deterministic: same file same seed -> same solver
        assert selector.algorithm_select(smt_path) == solver_id
    finally:
        smt_path.unlink(missing_ok=True)
