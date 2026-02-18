"""Tests for GIN-ready graph conversion (vocab + graph_dict_to_gin_data in gin_model)."""

import tempfile
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_geometric")

from src.graph_rep import smt_to_graph, smt_graph_to_gin
from src.gin_ehm import (
    NodeVocabulary,
    build_vocabulary_from_graph_dicts,
    graph_dict_to_gin_data,
    UNK_TYPE_INDEX,
)


def test_smt_graph_to_gin():
    """smt_graph_to_gin returns node_types and undirected edges."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smt2", delete=False
    ) as f:
        f.write(
            "(set-logic QF_LIA)\n"
            "(declare-const x Int)\n"
            "(assert (>= x 0))\n"
            "(check-sat)\n"
        )
        path = Path(f.name)
    try:
        graph_dict = smt_to_graph(path)
        gin = smt_graph_to_gin(graph_dict)
        assert "node_types" in gin and "edges" in gin
        assert len(gin["node_types"]) == len(graph_dict["nodes"])
        assert all(isinstance(t, str) for t in gin["node_types"])
        assert all(len(e) == 2 for e in gin["edges"])
    finally:
        path.unlink(missing_ok=True)


def test_node_vocabulary_add_and_get():
    vocab = NodeVocabulary()
    assert vocab.get_index("bvmul") == UNK_TYPE_INDEX
    idx = vocab.add_type("bvmul")
    assert idx == 1
    assert vocab.get_index("bvmul") == 1
    vocab.add_type("forall")
    assert vocab.get_index("forall") == 2
    vocab.freeze()
    assert vocab.get_index("unknown_type") == UNK_TYPE_INDEX
    assert vocab.num_types() == 2
    assert set(vocab.type_names()) == {"bvmul", "forall"}


def test_graph_dict_to_gin_data():
    """Convert a small graph dict to PyG Data and check shapes."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smt2", delete=False
    ) as f:
        f.write(
            "(set-logic QF_LIA)\n"
            "(declare-const x Int)\n"
            "(assert (>= x 0))\n"
            "(check-sat)\n"
        )
        path = Path(f.name)
    try:
        graph_dict = smt_to_graph(path)
        vocab = NodeVocabulary()
        vocab.add_graph_dict(graph_dict)
        vocab.freeze()

        data = graph_dict_to_gin_data(graph_dict, vocab)
        assert isinstance(data.x, torch.Tensor)
        assert data.x.dtype == torch.long
        assert data.x.dim() == 2 and data.x.shape[1] == 1
        assert data.x.shape[0] == len(graph_dict["nodes"])
        assert isinstance(data.edge_index, torch.Tensor)
        assert data.edge_index.shape[0] == 2
        # Undirected: each edge (u,v) and (v,u)
        assert data.edge_index.shape[1] % 2 == 0
    finally:
        path.unlink(missing_ok=True)


def test_build_vocabulary_from_graph_dicts():
    """Vocabulary built from multiple graphs includes all types and freezes."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smt2", delete=False
    ) as f1:
        f1.write(
            "(set-logic QF_LIA)\n"
            "(declare-const x Int)\n"
            "(assert (>= x 0))\n"
            "(check-sat)\n"
        )
        p1 = Path(f1.name)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smt2", delete=False
    ) as f2:
        f2.write(
            "(set-logic QF_BV)\n"
            "(declare-const a (_ BitVec 8))\n"
            "(assert (bvuge a a))\n"
            "(check-sat)\n"
        )
        p2 = Path(f2.name)
    try:
        d1 = smt_to_graph(p1)
        d2 = smt_to_graph(p2)
        vocab = build_vocabulary_from_graph_dicts([d1, d2])
        assert vocab.num_types() >= 1
        assert (
            vocab.get_index("bvuge") != UNK_TYPE_INDEX
            or "bvuge" in vocab.type_names()
        )
        data = graph_dict_to_gin_data(d2, vocab)
        assert data.x.shape[0] == len(d2["nodes"])
    finally:
        p1.unlink(missing_ok=True)
        p2.unlink(missing_ok=True)
