#!/usr/bin/env python3
"""Evaluate GIN selector (GIN-PWC or GIN EHM): load model, run as_evaluate, report metrics (solved, PAR2, gap_cls)."""

import json
import logging
from pathlib import Path

from .defaults import DEFAULT_BENCHMARK_ROOT
from .evaluate import as_evaluate, compute_metrics, format_evaluation_short
from .gin_ehm import GINSelector
from .gin_pwc import GINPwcSelector
from .performance import MultiSolverDataset, parse_performance_json
from .solver_selector import SolverSelector


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
    print(format_evaluation_short(metrics))
    if args.output_csv:
        print(f"Wrote per-instance results to {args.output_csv}")


if __name__ == "__main__":
    main()
