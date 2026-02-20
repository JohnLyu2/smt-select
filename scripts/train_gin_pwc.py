#!/usr/bin/env python3
"""Train GIN-PWC: GIN backbone + pairwise classifier heads (weighted by |PAR2_i - PAR2_j|)."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.defaults import DEFAULT_BENCHMARK_ROOT
from src.gin_pwc import train_gin_pwc
from src.performance import (
    filter_training_instances,
    parse_performance_json,
    MultiSolverDataset,
)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Train GIN-PWC (GIN backbone + pairwise heads) for algorithm selection"
    )
    parser.add_argument("--perf-json", type=str, required=True, help="Path to performance JSON (e.g. train.json)")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save model and artifacts")
    parser.add_argument("--graph-timeout", type=int, default=5, help="Graph build timeout in seconds")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers for graph building; 1 = sequential")
    parser.add_argument("--timeout", type=float, default=1200.0, help="PAR2 timeout in seconds")
    parser.add_argument("--benchmark-root", type=str, default=DEFAULT_BENCHMARK_ROOT, help="Root for relative instance paths")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of data for validation (0 = no early stop)")
    parser.add_argument("--patience", type=int, default=50, help="Epochs without val improvement to stop (0 = disabled)")
    parser.add_argument("--val-split-seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--min-epochs", type=int, default=100, help="Minimum epochs before early stop can trigger")
    parser.add_argument(
        "--skip-easy-unsolvable",
        action="store_true",
        help="Exclude instances where no solver solves (VBS-unsolvable) or all solvers solve in <= N seconds (trivial); saves graph/train time",
    )
    parser.add_argument(
        "--skip-trivial-under",
        type=float,
        default=24.0,
        help="When --skip-easy-unsolvable: exclude instances where every solver solved with runtime <= this (seconds). Default 24",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    if args.skip_easy_unsolvable:
        paths_to_keep, filter_stats = filter_training_instances(
            multi_perf_data,
            skip_unsolvable=True,
            skip_trivial_under=args.skip_trivial_under,
        )
        multi_perf_data = MultiSolverDataset(
            {p: multi_perf_data[p] for p in paths_to_keep},
            multi_perf_data.get_solver_id_dict(),
            multi_perf_data.get_timeout(),
        )
        logging.info(
            "Filtered to %d instances (dropped %d unsolvable, %d trivial)",
            filter_stats["n_kept"],
            filter_stats["n_unsolvable"],
            filter_stats["n_trivial"],
        )
    else:
        filter_stats = None

    logging.info(
        "Training GIN-PWC: %d instances, %d solvers",
        len(multi_perf_data),
        multi_perf_data.num_solvers(),
    )

    train_gin_pwc(
        multi_perf_data,
        args.save_dir,
        graph_timeout=args.graph_timeout,
        jobs=args.jobs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        dropout=args.dropout,
        val_ratio=args.val_ratio,
        patience=args.patience,
        val_split_seed=args.val_split_seed,
        min_epochs=args.min_epochs,
    )

    if filter_stats is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "skipped_unsolvable.txt", "w") as f:
            for p in filter_stats["skipped_unsolvable"]:
                f.write(p + "\n")
        with open(save_dir / "skipped_trivial.txt", "w") as f:
            for p in filter_stats["skipped_trivial"]:
                f.write(p + "\n")
        logging.info("Wrote skipped_unsolvable.txt and skipped_trivial.txt to %s", save_dir)


if __name__ == "__main__":
    main()
