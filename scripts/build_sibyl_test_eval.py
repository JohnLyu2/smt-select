#!/usr/bin/env python3
"""
Build test_eval.csv for Sibyl selections in the same format as GNN (gin_pwc) test_eval.

Every test instance (from test.json) has a row. Overhead from graph_log, capped at 10s.
Case A (in selection CSV): use selection, feature_fail=0.
Case B (missing from selection): use SBS from train, feature_fail=1.

Reads: data/cp26/results/sibyl/selection/<logic>/seed{N}.csv (benchmark, selected)
       data/cp26/performance_splits/smtcomp24/<logic>/seed{N}/train.json, test.json
       data/cp26/results/sibyl/graph_log/graph_build_<logic>.csv (path, time_sec, failed)
Writes: data/cp26/results/sibyl/evaluation/<logic>/seed{N}/test_eval.csv
        data/cp26/results/sibyl/evaluation/<logic>/summary.json (test metrics only)
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from src.evaluate import _apply_overhead_to_perf
from src.evaluate import compute_metrics
from src.performance import parse_as_perf_csv
from src.performance import parse_performance_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_DIR = PROJECT_ROOT / "data" / "cp26" / "results" / "sibyl" / "selection"
SPLITS_BASE = PROJECT_ROOT / "data" / "cp26" / "performance_splits" / "smtcomp24"
GRAPH_LOG_DIR = PROJECT_ROOT / "data" / "cp26" / "results" / "sibyl" / "graph_log"
OUT_ROOT = PROJECT_ROOT / "data" / "cp26" / "results" / "sibyl" / "evaluation"
OVERHEAD_CAP = 10.0

HEADER = ["benchmark", "selected", "solved", "runtime", "solver_runtime", "overhead", "feature_fail"]


def _seed_from_filename(name: str) -> int | None:
    m = re.match(r"seed(\d+)\.csv$", name)
    return int(m.group(1)) if m else None


def load_graph_log(graph_log_csv: Path) -> tuple[dict[str, float], dict[str, int]]:
    """
    Load graph_build_<logic>.csv (path, time_sec, failed).
    Returns (path -> time_sec overhead, path -> feature_fail 0/1).
    Raises if the file is missing.
    """
    if not graph_log_csv.exists():
        raise FileNotFoundError(f"Graph log not found: {graph_log_csv}")
    overhead_by_path: dict[str, float] = {}
    feature_fail_by_path: dict[str, int] = {}
    with graph_log_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "path" not in reader.fieldnames or "time_sec" not in reader.fieldnames:
            raise ValueError(
                f"Graph log expected columns 'path','time_sec','failed'; got {reader.fieldnames}"
            )
        for row in reader:
            path = (row.get("path") or "").strip()
            if not path:
                continue
            try:
                time_sec = float(row["time_sec"])
            except (ValueError, TypeError):
                time_sec = 0.0
            failed = 1 if (row.get("failed", "0").strip() in ("1", "true", "True")) else 0
            overhead_by_path[path] = time_sec
            feature_fail_by_path[path] = failed
    return overhead_by_path, feature_fail_by_path


def load_selection_lookup(selection_csv: Path) -> dict[str, str]:
    """Load selection CSV into benchmark -> selected solver name. Empty if file missing."""
    if not selection_csv.exists():
        return {}
    lookup: dict[str, str] = {}
    with selection_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["benchmark", "selected"]:
            raise ValueError(
                f"Expected columns 'benchmark','selected' in {selection_csv}, got {reader.fieldnames}"
            )
        for row in reader:
            bench = (row.get("benchmark") or "").strip()
            sel = (row.get("selected") or "").strip()
            if bench and sel:
                lookup[bench] = sel
    return lookup


def compute_sbs_from_train(train_json: Path, timeout: float) -> str:
    """Single best solver (lowest average PAR-2) on train. Raises if train missing or empty."""
    if not train_json.exists():
        raise FileNotFoundError(f"Train JSON not found: {train_json}")
    train_perf = parse_performance_json(str(train_json), timeout)
    best_id = train_perf.get_best_solver_id()
    name = train_perf.get_solver_name(best_id)
    if name is None:
        raise ValueError("SBS solver name is None")
    return name


def build_test_eval(
    selection_csv: Path,
    train_json: Path,
    test_json: Path,
    out_csv: Path,
    overhead_by_path: dict[str, float],
    timeout: float = 1200.0,
    overhead_cap: float = OVERHEAD_CAP,
) -> None:
    """
    One row per test instance. Overhead from graph_log, required for all test instances, capped.
    Case A (in selection): use selection, feature_fail=0. Case B (missing): SBS from train, feature_fail=1.
    """
    if not test_json.exists():
        raise FileNotFoundError(f"Test JSON not found: {test_json}")

    multi_perf = parse_performance_json(str(test_json), timeout)
    solver_id_by_name = {name: sid for sid, name in multi_perf.get_solver_id_dict().items()}
    test_benchmarks = sorted(multi_perf.keys())

    missing_in_graph_log = [b for b in test_benchmarks if b not in overhead_by_path]
    if missing_in_graph_log:
        raise ValueError(
            f"Every test instance must be in graph_log. Missing: {missing_in_graph_log[:5]}{'...' if len(missing_in_graph_log) > 5 else ''}"
        )

    selection_lookup = load_selection_lookup(selection_csv)
    sbs_name = compute_sbs_from_train(train_json, timeout)
    if sbs_name not in solver_id_by_name:
        raise ValueError(
            f"SBS solver {sbs_name!r} from train not in test performance data (solver list may differ)"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list] = []

    for benchmark in test_benchmarks:
        raw_overhead = overhead_by_path[benchmark]
        total_overhead = min(raw_overhead, overhead_cap)

        if benchmark in selection_lookup:
            selected_name = selection_lookup[benchmark]
            feature_fail = 0
        else:
            selected_name = sbs_name
            feature_fail = 1

        if selected_name not in solver_id_by_name:
            raise ValueError(
                f"Selected solver {selected_name!r} not in test data for benchmark {benchmark!r}"
            )
        solver_id = solver_id_by_name[selected_name]
        raw_perf = multi_perf.get_performance(benchmark, solver_id)
        if raw_perf is None:
            raise ValueError(
                f"No performance for benchmark={benchmark!r} solver={selected_name!r}"
            )
        raw_solved, raw_runtime = raw_perf

        solved, runtime = _apply_overhead_to_perf(
            raw_solved, raw_runtime, total_overhead, timeout
        )
        rows.append([
            benchmark,
            selected_name,
            solved,
            runtime,
            raw_runtime,
            f"{total_overhead:.6f}" if total_overhead else "",
            feature_fail,
        ])

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Sibyl test_eval.csv from selection CSVs and performance splits."
    )
    parser.add_argument(
        "--logic",
        type=str,
        required=True,
        help="Logic/division (e.g. ABV).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Seeds to process (default: all seed* CSV files in selection/<logic>/).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Timeout in seconds (default: 1200).",
    )
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=SELECTION_DIR,
        help=f"Base directory for selection CSVs (default: {SELECTION_DIR}).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_BASE,
        help=f"Base directory for performance splits (default: {SPLITS_BASE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_ROOT,
        help=f"Output base directory (default: {OUT_ROOT}).",
    )
    parser.add_argument(
        "--graph-log-dir",
        type=Path,
        default=GRAPH_LOG_DIR,
        help=f"Directory containing graph_build_<logic>.csv (default: {GRAPH_LOG_DIR}).",
    )
    args = parser.parse_args()

    logic_dir = args.selection_dir / args.logic
    if not logic_dir.is_dir():
        raise FileNotFoundError(f"Selection logic dir not found: {logic_dir}")

    graph_log_csv = args.graph_log_dir / f"graph_build_{args.logic}.csv"
    overhead_by_path, feature_fail_by_path = load_graph_log(graph_log_csv)

    if args.seeds is not None:
        seeds = sorted(args.seeds)
    else:
        seeds = []
        for p in logic_dir.iterdir():
            if p.is_file() and p.suffix == ".csv":
                s = _seed_from_filename(p.name)
                if s is not None:
                    seeds.append(s)
        seeds = sorted(set(seeds))

    if not seeds:
        raise ValueError(f"No seed CSVs found in {logic_dir}")

    for seed in seeds:
        selection_csv = logic_dir / f"seed{seed}.csv"
        train_json = args.splits_dir / args.logic / f"seed{seed}" / "train.json"
        test_json = args.splits_dir / args.logic / f"seed{seed}" / "test.json"
        out_csv = args.output_dir / args.logic / f"seed{seed}" / "test_eval.csv"
        build_test_eval(
            selection_csv,
            train_json,
            test_json,
            out_csv,
            overhead_by_path,
            timeout=args.timeout,
            overhead_cap=OVERHEAD_CAP,
        )
        print(f"Wrote {out_csv}")

    # Summary (test metrics only), reusing compute_metrics from evaluate
    seed_results: list[dict] = []
    for seed in seeds:
        test_csv = args.output_dir / args.logic / f"seed{seed}" / "test_eval.csv"
        test_json_path = args.splits_dir / args.logic / f"seed{seed}" / "test.json"
        if not test_csv.exists():
            raise FileNotFoundError(f"Test eval CSV not found: {test_csv}")
        if not test_json_path.exists():
            raise FileNotFoundError(f"Test JSON not found: {test_json_path}")
        result_dataset = parse_as_perf_csv(str(test_csv), args.timeout)
        test_data = parse_performance_json(str(test_json_path), args.timeout)
        test_metrics = compute_metrics(result_dataset, test_data)
        seed_results.append({
            "seed": seed,
            "test_size": test_metrics["total_count"],
            "test_metrics": test_metrics,
        })
    test_metrics_list = [r["test_metrics"] for r in seed_results]
    aggregated = {
        "test": {
            "gap_cls_solved_mean": float(np.mean([m["gap_cls_solved"] for m in test_metrics_list])),
            "gap_cls_solved_std": float(np.std([m["gap_cls_solved"] for m in test_metrics_list])),
            "gap_cls_par2_mean": float(np.mean([m["gap_cls_par2"] for m in test_metrics_list])),
            "gap_cls_par2_std": float(np.std([m["gap_cls_par2"] for m in test_metrics_list])),
        }
    }
    results = {
        "division": args.logic,
        "n_seeds": len(seeds),
        "seed_values": seeds,
        "seeds": seed_results,
        "aggregated": aggregated,
    }
    summary_path = args.output_dir / args.logic / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
