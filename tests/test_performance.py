"""Small tests for performance module (datasets and CSV/JSON parsers)."""

import csv
import json
from pathlib import Path

import pytest

from src.performance import (
    MultiSolverDataset,
    SingleSolverDataset,
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
