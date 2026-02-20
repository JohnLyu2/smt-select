"""Small tests for performance module (datasets and CSV/JSON parsers)."""

import csv
import json
from pathlib import Path

import pytest

from src.performance import (
    MultiSolverDataset,
    SingleSolverDataset,
    filter_training_instances,
    parse_as_perf_csv,
    parse_performance_csv,
    parse_performance_json,
)


def test_parse_performance_csv(tmp_path: Path) -> None:
    """Parse multi-solver performance CSV and check dataset."""
    csv_file = tmp_path / "perf.csv"
    # Header: path, SolverA,, SolverB,,  then ,solved,runtime,solved,runtime
    with csv_file.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "SolverA", "", "SolverB", ""])
        w.writerow(["", "solved", "runtime", "solved", "runtime"])
        w.writerow(["bench/a.smt2", 1, 10.5, 0, 0.0])
        w.writerow(["bench/b.smt2", 0, 0.0, 1, 2.3])

    timeout = 1200.0
    data = parse_performance_csv(str(csv_file), timeout)

    assert len(data) == 2
    assert data.get_timeout() == timeout
    assert data.get_solver_name(0) == "SolverA"
    assert data.get_solver_name(1) == "SolverB"

    perf_a = data.get_performance("bench/a.smt2", 0)
    assert perf_a == (1, 10.5)
    perf_b_s1 = data.get_performance("bench/b.smt2", 1)
    assert perf_b_s1 == (1, 2.3)


def test_parse_as_perf_csv(tmp_path: Path) -> None:
    """Parse algorithm-selection result CSV and check SingleSolverDataset."""
    csv_file = tmp_path / "as_results.csv"
    with csv_file.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "selected", "solved", "runtime"])
        w.writerow(["p1.smt2", "cvc5", 1, 5.0])
        w.writerow(["p2.smt2", "Bitwuzla", 0, 0.0])

    timeout = 600.0
    data = parse_as_perf_csv(str(csv_file), timeout)

    assert len(data) == 2
    assert data.get_timeout() == timeout
    assert data.get_solver_name() == "as_results"
    assert data["p1.smt2"] == (1, 5.0)
    assert data["p2.smt2"] == (0, 0.0)


def test_parse_performance_json(tmp_path: Path) -> None:
    """Parse smtcomp24-style performance JSON; use wallclock_time; sat/unsat = solved."""
    json_file = tmp_path / "BV.json"
    obj = {
        "b1.smt2": {
            "Bitwuzla": {"result": "unsat", "cpu_time": 1.0, "wallclock_time": 1.1, "memory_usage": 1e6},
            "cvc5": {"result": "Timeout", "cpu_time": 1200.0, "wallclock_time": 1200.5, "memory_usage": 2e6},
        },
        "b2.smt2": {
            "Bitwuzla": {"result": "unknown", "cpu_time": 2.0, "wallclock_time": 2.2, "memory_usage": 1e6},
            "cvc5": {"result": "sat", "cpu_time": 0.5, "wallclock_time": 0.55, "memory_usage": 1e6},
        },
    }
    json_file.write_text(json.dumps(obj), encoding="utf-8")

    timeout = 1200.0
    data = parse_performance_json(str(json_file), timeout)

    assert len(data) == 2
    assert data.get_timeout() == timeout
    # Solver order is sorted: Bitwuzla=0, cvc5=1
    assert data.get_solver_name(0) == "Bitwuzla"
    assert data.get_solver_name(1) == "cvc5"

    # b1: Bitwuzla unsat 1.1s (solved), cvc5 Timeout (not solved)
    assert data.get_performance("b1.smt2", 0) == (1, 1.1)
    assert data.get_performance("b1.smt2", 1) == (0, 1200.5)
    # b2: Bitwuzla unknown (not solved), cvc5 sat 0.55s (solved)
    assert data.get_performance("b2.smt2", 0) == (0, 2.2)
    assert data.get_performance("b2.smt2", 1) == (1, 0.55)


def test_multi_solver_dataset_basic() -> None:
    """MultiSolverDataset construction and get_par2."""
    perf_dict = {
        "a": [(1, 10.0), (0, 0.0)],   # solver0 solved in 10s, solver1 unsolved
        "b": [(0, 0.0), (1, 5.0)],
    }
    solver_dict = {0: "S1", 1: "S2"}
    timeout = 20.0
    ds = MultiSolverDataset(perf_dict, solver_dict, timeout)

    assert len(ds) == 2
    assert ds.get_par2("a", 0) == 10.0
    assert ds.get_par2("a", 1) == 2 * timeout  # PAR-2 for unsolved
    assert ds.get_par2("b", 1) == 5.0


def test_is_none_solved_and_is_trivial() -> None:
    """is_none_solved and is_trivial (strict: all solvers, all solved, runtime <= max)."""
    # 2 solvers; (is_solved, wc_time) per solver
    perf_dict = {
        "unsolvable": [(0, 100.0), (0, 100.0)],   # no solver solved
        "trivial": [(1, 10.0), (1, 24.0)],         # both solved, both <= 24
        "trivial_strict_under_5": [(1, 2.0), (1, 4.0)],
        "not_trivial_one_timeout": [(1, 10.0), (0, 1200.0)],
        "not_trivial_one_slow": [(1, 10.0), (1, 100.0)],  # one > 24
    }
    solver_dict = {0: "A", 1: "B"}
    timeout = 1200.0
    ds = MultiSolverDataset(perf_dict, solver_dict, timeout)

    assert ds.is_none_solved("unsolvable") is True
    assert ds.is_none_solved("trivial") is False
    assert ds.is_none_solved("not_trivial_one_timeout") is False

    assert ds.is_trivial("trivial", 24.0) is True
    assert ds.is_trivial("trivial_strict_under_5", 5.0) is True
    assert ds.is_trivial("trivial_strict_under_5", 3.0) is False   # 4 > 3
    assert ds.is_trivial("not_trivial_one_timeout", 24.0) is False
    assert ds.is_trivial("not_trivial_one_slow", 24.0) is False
    assert ds.is_trivial("unsolvable", 24.0) is False


def test_filter_training_instances() -> None:
    """filter_training_instances drops unsolvable and trivial (strict)."""
    perf_dict = {
        "unsolvable": [(0, 100.0), (0, 100.0)],
        "trivial": [(1, 10.0), (1, 24.0)],
        "keep": [(1, 10.0), (0, 1200.0)],
    }
    solver_dict = {0: "A", 1: "B"}
    timeout = 1200.0
    ds = MultiSolverDataset(perf_dict, solver_dict, timeout)

    paths, stats = filter_training_instances(
        ds, skip_unsolvable=True, skip_trivial_under=24.0
    )
    assert set(paths) == {"keep"}
    assert stats["n_kept"] == 1
    assert stats["n_unsolvable"] == 1
    assert stats["n_trivial"] == 1
    assert stats["skipped_unsolvable"] == ["unsolvable"]
    assert stats["skipped_trivial"] == ["trivial"]

    # Only skip unsolvable
    paths2, stats2 = filter_training_instances(
        ds, skip_unsolvable=True, skip_trivial_under=None
    )
    assert set(paths2) == {"trivial", "keep"}
    assert stats2["n_trivial"] == 0

    # No filtering
    paths3, stats3 = filter_training_instances(
        ds, skip_unsolvable=False, skip_trivial_under=None
    )
    assert set(paths3) == {"unsolvable", "trivial", "keep"}
    assert stats3["n_unsolvable"] == 0
