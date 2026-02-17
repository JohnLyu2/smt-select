#!/usr/bin/env python3
"""
Run evaluate_multi_splits_wl for given logic and wl_iter values. Results are saved
under data/cp26/results/wl/{logic}/{iter}.

Usage (from project root):
  python scripts/run_wl_iter_sweep.py --logic BV --iter 0 1 2 3
  python scripts/run_wl_iter_sweep.py --logic BV --iter-le 3
  python scripts/run_wl_iter_sweep.py --logic BV --iter 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow importing from scripts/ when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_multi_splits_wl import evaluate_multi_splits_wl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run WL multi-splits evaluation for a logic and wl_iter values",
    )
    parser.add_argument(
        "--logic",
        type=str,
        default="BV",
        help="Logic name used for default paths, e.g. BV, QF_BV (default: BV)",
    )
    parser.add_argument(
        "--iter",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        metavar="N",
        help="WL iteration(s) to run; results under results-dir/N (default: 0 1 2 3)",
    )
    parser.add_argument(
        "--iter-le",
        type=int,
        default=None,
        metavar="N",
        help="Run for wl_iter 0..N inclusive; overrides --iter if set",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Splits directory (default: data/cp26/performance_splits/smtcomp24/{logic})",
    )
    parser.add_argument(
        "--wl-dir",
        type=str,
        default=None,
        help="WL feature directory (default: data/features/wl/{logic})",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Base results directory; outputs under results-dir/N (default: data/cp26/results/wl/{logic})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Timeout in seconds (default: 1200)",
    )
    parser.add_argument(
        "--svm-c",
        type=float,
        default=1.0,
        help="SVM C (default: 1.0)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save model per split in each output dir",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logic = args.logic
    splits_dir = Path(
        args.splits_dir or f"data/cp26/performance_splits/smtcomp24/{logic}"
    )
    wl_dir = Path(args.wl_dir or f"data/features/wl/{logic}")
    results_base = Path(args.results_dir or f"data/cp26/results/wl/{logic}")

    iterations = (
        list(range(0, args.iter_le + 1))
        if args.iter_le is not None
        else args.iter
    )
    for wl_iter in iterations:
        output_dir = results_base / str(wl_iter)
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Running wl_iter=%d -> %s", wl_iter, output_dir)
        evaluate_multi_splits_wl(
            splits_dir,
            wl_dir,
            wl_iter,
            save_models=args.save_models,
            output_dir=output_dir,
            timeout=args.timeout,
            svm_c=args.svm_c,
            random_seed=args.random_seed,
        )

    logging.info("Done. Results in %s/%s", results_base, ",".join(map(str, iterations)))


if __name__ == "__main__":
    main()
