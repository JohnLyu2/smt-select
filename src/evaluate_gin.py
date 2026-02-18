#!/usr/bin/env python3
"""Evaluate GIN selector (GIN-PWC or GIN EHM): load model, run as_evaluate, report metrics (solved, PAR2, gap_cls)."""

import json
import logging
from pathlib import Path

from .defaults import DEFAULT_BENCHMARK_ROOT
from .evaluate import as_evaluate
from .gin_ehm import GINSelector
from .gin_pwc import GINPwcSelector
from .performance import MultiSolverDataset, parse_performance_json
from .solver_selector import SolverSelector


def compute_metrics(result_dataset, multi_perf_data):
    """Same as evaluate_multi_splits_wl: solved, avg_par2, gap_cls vs SBS/VBS."""
    total_count = len(result_dataset)
    solved_count = result_dataset.get_solved_count()
    total_par2 = sum(result_dataset.get_par2(p) for p in result_dataset.keys())
    avg_par2 = total_par2 / total_count if total_count > 0 else 0.0

    sbs_dataset = multi_perf_data.get_best_solver_dataset()
    sbs_solved = sbs_dataset.get_solved_count()
    total_par2_sbs = sum(sbs_dataset.get_par2(p) for p in sbs_dataset.keys())
    avg_par2_sbs = total_par2_sbs / total_count if total_count > 0 else 0.0

    vbs_dataset = multi_perf_data.get_virtual_best_solver_dataset()
    vbs_solved = vbs_dataset.get_solved_count()
    total_par2_vbs = sum(vbs_dataset.get_par2(p) for p in vbs_dataset.keys())
    avg_par2_vbs = total_par2_vbs / total_count if total_count > 0 else 0.0

    solved_denom = vbs_solved - sbs_solved
    par2_denom = avg_par2_vbs - avg_par2_sbs
    gap_cls_solved = (
        (solved_count - sbs_solved) / solved_denom
        if solved_denom != 0
        else (1.0 if solved_count == vbs_solved else 0.0)
    )
    gap_cls_par2 = (
        (avg_par2 - avg_par2_sbs) / par2_denom
        if par2_denom != 0
        else (1.0 if avg_par2 == avg_par2_vbs else 0.0)
    )
    return {
        "solved": solved_count,
        "avg_par2": avg_par2,
        "sbs_solved": sbs_solved,
        "sbs_avg_par2": avg_par2_sbs,
        "vbs_solved": vbs_solved,
        "vbs_avg_par2": avg_par2_vbs,
        "gap_cls_solved": gap_cls_solved,
        "gap_cls_par2": gap_cls_par2,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate GIN algorithm selector")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with config.json, model.pt, vocab.json")
    parser.add_argument("--perf-json", type=str, required=True, help="Performance JSON (e.g. test.json)")
    parser.add_argument("--timeout", type=float, default=1200.0, help="PAR2 timeout in seconds")
    parser.add_argument("--benchmark-root", type=str, default=DEFAULT_BENCHMARK_ROOT, help="Root for relative instance paths")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV path for per-instance results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    if "num_heads" in config:
        selector: SolverSelector = GINSelector.load(args.model_dir)
        logging.info("Loaded GIN EHM (regression) selector from %s", args.model_dir)
    elif "num_solvers" in config:
        selector = GINPwcSelector.load(args.model_dir)
        logging.info("Loaded GIN-PWC selector from %s", args.model_dir)
    else:
        raise ValueError(
            f"Unknown GIN config in {args.model_dir}: expected 'num_heads' (EHM) or 'num_solvers' (PWC)"
        )
    multi_perf_data = parse_performance_json(args.perf_json, args.timeout)
    if args.benchmark_root:
        root = Path(args.benchmark_root).resolve()
        if not root.is_dir():
            raise ValueError(f"--benchmark-root is not a directory: {root}")
        rebased = {str(root / p): multi_perf_data[p] for p in multi_perf_data.keys()}
        multi_perf_data = MultiSolverDataset(
            rebased,
            multi_perf_data.get_solver_id_dict(),
            multi_perf_data.get_timeout(),
        )

    result = as_evaluate(
        selector,
        multi_perf_data,
        write_csv_path=args.output_csv,
        show_progress=True,
    )
    metrics = compute_metrics(result, multi_perf_data)
    n = len(multi_perf_data)
    sr = (metrics["solved"] / n * 100) if n else 0
    print(f"Instances: {n}, Solved: {metrics['solved']}, Solve rate: {sr:.2f}%")
    print(f"Avg PAR2: {metrics['avg_par2']:.2f}, gap_cls_par2: {metrics['gap_cls_par2']:.4f}")
    if args.output_csv:
        print(f"Wrote per-instance results to {args.output_csv}")


if __name__ == "__main__":
    main()
