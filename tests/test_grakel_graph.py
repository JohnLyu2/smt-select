"""Tests that the GraKel graph conversion works with the WL kernel."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("grakel")
from grakel.kernels import WeisfeilerLehman

from src.graph_rep import smt_to_graph, smt_graph_to_grakel


def test_smt_graph_to_grakel_single_graph():
    """Convert one SMT instance to GraKel and compute WL kernel on it."""
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
        g = smt_graph_to_grakel(graph_dict)
        assert g is not None
        wl = WeisfeilerLehman(n_iter=2, normalize=True)
        K = wl.fit_transform([g])
        assert K.shape == (1, 1)
        assert K[0, 0] >= 0
    finally:
        path.unlink(missing_ok=True)


def test_smt_graph_to_grakel_two_graphs():
    """Two SMT instances -> two GraKel graphs -> WL kernel matrix 2x2."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smt2", delete=False
    ) as f1:
        f1.write(
            "(set-logic QF_LIA)\n"
            "(declare-const x Int)\n"
            "(assert (>= x 0))\n"
            "(check-sat)\n"
        )
        path1 = Path(f1.name)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smt2", delete=False
    ) as f2:
        f2.write(
            "(set-logic QF_LIA)\n"n
    try:
        d1 = smt_to_graph(path1)
        d2 = smt_to_graph(path2)
        g1 = smt_graph_to_grakel(d1)
        g2 = smt_graph_to_grakel(d2)
        wl = WeisfeilerLehman(n_iter=2, normalize=True)
        K = wl.fit_transform([g1, g2])
        assert K.shape == (2, 2)
        assert K[0, 1] == K[1, 0]
        assert K[0, 0] >= 0 and K[1, 1] >= 0
    finally:
        path1.unlink(missing_ok=True)
        path2.unlink(missing_ok=True)
