#!/usr/bin/env python3
"""
Evaluate GIN (EHM or PWC) over multiple train/test splits.

Expects --splits-dir to point at a division folder containing seed subdirs, e.g.:
  data/cp26/performance_splits/smtcomp24/BV/
    seed0/train.json, test.json
    seed10/train.json, test.json
    ...

For each split (seed):
  1. Load train.json and test.json; rebase paths with --benchmark-root.
  2. Train GIN EHM or GIN-PWC on the train set (--model-type).
  3. Evaluate on train and test sets.
  4. Compute metrics (solve rate, PAR2, gap closed vs SBS/VBS).

Results are aggregated across splits (mean ± std) and optionally saved to summary.json.
"""

import argparse
import gc
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.defaults import DEFAULT_BENCHMARK_ROOT
from src.evaluate import as_evaluate, as_evaluate_parallel, compute_metrics
from src.evaluate_gin import _load_gin_selector
from src.gin_ehm import train_gin_regression
from src.gin_pwc import train_gin_pwc
from src.performance import MultiSolverDataset, parse_performance_json


def discover_seed_dirs(splits_dir: Path) -> list[tuple[int, Path]]:
    """
    Find seed subdirectories under splits_dir that contain train.json and test.json.
    Returns list of (seed_value, seed_dir_path) sorted by seed value.
    """
    out: list[tuple[int, Path]] = []
    pattern = re.compile(r"^seed(\d+)$")
    for sub in splits_dir.iterdir():
        if not sub.is_dir():
            continue
        m = pattern.match(sub.name)
        if not m:
            continue
        if (sub / "train.json").exists() and (sub / "test.json").exists():
            out.append((int(m.group(1)), sub))
    return sorted(out, key=lambda x: x[0])


def _rebase_perf_data(multi_perf_data: MultiSolverDataset, benchmark_root: Path) -> MultiSolverDataset:
    """Rebase instance paths with benchmark_root."""
    rebased = {str(benchmark_root / p): multi_perf_data[p] for p in multi_perf_data.keys()}
    return MultiSolverDataset(
        rebased,
        multi_perf_data.get_solver_id_dict(),
        multi_perf_data.get_timeout(),
    )


def evaluate_multi_splits_gin(
    splits_dir: Path,
    *,
    model_type: str = "gin_ehm",
    benchmark_root: Path | str | None = None,
    save_models: bool = False,
    output_dir: Path | None = None,
    timeout: float = 1200.0,
    graph_timeout: int = 5,
    jobs: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 3,
    num_epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout: float = 0.1,
    val_ratio: float = 0.1,
    patience: int = 20,
    val_split_seed: int = 42,
    min_epochs: int = 50,
) -> dict:
    """
    Run train/test evaluation with GIN (EHM or PWC) for each split (seed) under splits_dir.
    Returns dict with division, n_seeds, per-split results, and aggregated metrics.
    """
    splits_dir = Path(splits_dir).resolve()
    if not splits_dir.is_dir():
        raise ValueError(f"Splits directory does not exist: {splits_dir}")

    root = Path(benchmark_root or DEFAULT_BENCHMARK_ROOT).resolve()
    if not root.is_dir():
        raise ValueError(f"Benchmark root is not a directory: {root}")

    division = splits_dir.name
    seed_entries = discover_seed_dirs(splits_dir)
    if not seed_entries:
        raise ValueError(
            f"No seed dirs (seedN with train.json and test.json) found in {splits_dir}"
        )

    n_seeds = len(seed_entries)
    logging.info(
        "Starting GIN (%s) evaluation over %d splits for division %s",
        model_type,
        n_seeds,
        division,
    )

    seed_results: list[dict] = []

    for seed_val, seed_dir in seed_entries:
        logging.info("\n%s", "=" * 60)
        logging.info("Seed %d (%s)", seed_val, seed_dir.name)
        logging.info("%s", "=" * 60)

        train_path = seed_dir / "train.json"
        test_path = seed_dir / "test.json"
        train_data = parse_performance_json(str(train_path), timeout)
        test_data = parse_performance_json(str(test_path), timeout)
        train_data = _rebase_perf_data(train_data, root)
        test_data = _rebase_perf_data(test_data, root)

        logging.info("Train instances: %d, Test instances: %d", len(train_data), len(test_data))

        if save_models and output_dir:
            model_save_dir = output_dir / "models" / f"seed{seed_val}"
            model_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            model_save_dir = Path(tempfile.mkdtemp())

        # Optionally capture training log to output_dir/train_log/seed{N}.log
        log_handler: logging.FileHandler | None = None
        if output_dir:
            train_log_dir = output_dir / "train_log"
            train_log_dir.mkdir(parents=True, exist_ok=True)
            log_file = train_log_dir / f"seed{seed_val}.log"
            log_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            log_handler.setLevel(logging.DEBUG)
            log_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logging.getLogger().addHandler(log_handler)

        try:
            if model_type == "gin_ehm":
                train_gin_regression(
                    train_data,
                    str(model_save_dir),
                    graph_timeout=graph_timeout,
                    jobs=jobs,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    dropout=dropout,
                    val_ratio=val_ratio,
                    patience=patience,
                    val_split_seed=val_split_seed,
                    min_epochs=min_epochs,
                )
            elif model_type == "gin_pwc":
                train_gin_pwc(
                    train_data,
                    str(model_save_dir),
                    graph_timeout=graph_timeout,
                    jobs=jobs,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    dropout=dropout,
                    val_ratio=val_ratio,
                    patience=patience,
                    val_split_seed=val_split_seed,
                    min_epochs=min_epochs,
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        finally:
            if log_handler is not None:
                logging.getLogger().removeHandler(log_handler)
                log_handler.close()

        train_output_csv = None
        test_output_csv = None
        if output_dir:
            out_train = output_dir / "train"
            out_test = output_dir / "test"
            out_train.mkdir(parents=True, exist_ok=True)
            out_test.mkdir(parents=True, exist_ok=True)
            train_output_csv = str(out_train / f"seed{seed_val}.csv")
            test_output_csv = str(out_test / f"seed{seed_val}.csv")

        if jobs > 1:
            train_instance_paths = list(train_data.keys())
            test_instance_paths = list(test_data.keys())
            train_result = as_evaluate_parallel(
                train_instance_paths,
                _load_gin_selector,
                (str(model_save_dir), "cpu"),
                train_data,
                n_workers=jobs,
                write_csv_path=train_output_csv,
                show_progress=True,
            )
            test_result = as_evaluate_parallel(
                test_instance_paths,
                _load_gin_selector,
                (str(model_save_dir), "cpu"),
                test_data,
                n_workers=jobs,
                write_csv_path=test_output_csv,
                show_progress=True,
            )
        else:
            selector = _load_gin_selector(str(model_save_dir))
            train_result = as_evaluate(
                selector, train_data, write_csv_path=train_output_csv, show_progress=True
            )
            test_result = as_evaluate(
                selector, test_data, write_csv_path=test_output_csv, show_progress=True
            )
        train_metrics = compute_metrics(train_result, train_data)
        test_metrics = compute_metrics(test_result, test_data)

        seed_results.append({
            "seed": seed_val,
            "train_size": len(train_data),
            "test_size": len(test_data),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        })

        n_train, n_test = len(train_data), len(test_data)
        train_sr = (train_metrics["solved"] / n_train * 100) if n_train else 0
        test_sr = (test_metrics["solved"] / n_test * 100) if n_test else 0
        logging.info(
            "  Train: solved %d/%d, solve_rate %.2f%%, gap_cls_par2 %.4f",
            train_metrics["solved"],
            n_train,
            train_sr,
            train_metrics["gap_cls_par2"],
        )
        logging.info(
            "  Test:  solved %d/%d, solve_rate %.2f%%, gap_cls_par2 %.4f",
            test_metrics["solved"],
            n_test,
            test_sr,
            test_metrics["gap_cls_par2"],
        )

        if not save_models:
            shutil.rmtree(model_save_dir, ignore_errors=True)

        # Free GPU memory so the next seed starts clean (no carry-over from this split)
        if jobs <= 1:
            del selector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_metrics_list = [r["test_metrics"] for r in seed_results]
    train_metrics_list = [r["train_metrics"] for r in seed_results]

    aggregated = {
        "train": {
            "gap_cls_solved_mean": float(np.mean([m["gap_cls_solved"] for m in train_metrics_list])),
            "gap_cls_solved_std": float(np.std([m["gap_cls_solved"] for m in train_metrics_list])),
            "gap_cls_par2_mean": float(np.mean([m["gap_cls_par2"] for m in train_metrics_list])),
            "gap_cls_par2_std": float(np.std([m["gap_cls_par2"] for m in train_metrics_list])),
        },
        "test": {
            "gap_cls_solved_mean": float(np.mean([m["gap_cls_solved"] for m in test_metrics_list])),
            "gap_cls_solved_std": float(np.std([m["gap_cls_solved"] for m in test_metrics_list])),
            "gap_cls_par2_mean": float(np.mean([m["gap_cls_par2"] for m in test_metrics_list])),
            "gap_cls_par2_std": float(np.std([m["gap_cls_par2"] for m in test_metrics_list])),
        },
    }

    results = {
        "division": division,
        "n_seeds": n_seeds,
        "seed_values": [s for s, _ in seed_entries],
        "model_type": model_type,
        "splits_dir": str(splits_dir),
        "benchmark_root": str(root),
        "seeds": seed_results,
        "aggregated": aggregated,
    }

    if output_dir:
        summary_path = output_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        def to_python(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: to_python(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_python(x) for x in obj]
            return obj

        with open(summary_path, "w") as f:
            json.dump(to_python(results), f, indent=2)
        logging.info("Saved summary to %s", summary_path)

    agg = results["aggregated"]
    logging.info("\n%s", "=" * 60)
    logging.info("Multi-splits summary — %s (GIN %s)", results["division"], model_type.upper())
    logging.info("%s", "=" * 60)
    logging.info("Seeds: %s", results["seed_values"])
    tr = agg["train"]
    logging.info(
        "Train: gap_cls_solved %.4f ± %.4f, gap_cls_par2 %.4f ± %.4f",
        tr["gap_cls_solved_mean"],
        tr["gap_cls_solved_std"],
        tr["gap_cls_par2_mean"],
        tr["gap_cls_par2_std"],
    )
    t = agg["test"]
    logging.info(
        "Test:  gap_cls_solved %.4f ± %.4f, gap_cls_par2 %.4f ± %.4f",
        t["gap_cls_solved_mean"],
        t["gap_cls_solved_std"],
        t["gap_cls_par2_mean"],
        t["gap_cls_par2_std"],
    )
    logging.info("%s", "=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GIN (EHM or PWC) over multiple train/test splits (per seed)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gin_ehm", "gin_pwc"],
        default="gin_ehm",
        help="GIN model to train and evaluate (default: gin_ehm)",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        required=True,
        help="Directory containing seed subdirs (e.g. data/cp26/performance_splits/smtcomp24/BV)",
    )
    parser.add_argument(
        "--benchmark-root",
        type=str,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Root for relative instance paths",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for summary.json and optional train/test CSVs and models",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save GIN model per split (requires --output-dir)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="PAR2 timeout in seconds (default: 1200)",
    )
    parser.add_argument(
        "--graph-timeout",
        type=int,
        default=5,
        help="Graph build timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Parallel workers for graph building and for evaluation (default: 4)",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of train data for validation (0 = no early stop)")
    parser.add_argument("--patience", type=int, default=20, help="Epochs without val improvement to stop (0 = disabled)")
    parser.add_argument("--val-split-seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--min-epochs", type=int, default=50, help="Minimum epochs before early stop can trigger")
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

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    evaluate_multi_splits_gin(
        Path(args.splits_dir),
        model_type=args.model_type,
        benchmark_root=args.benchmark_root,
        save_models=args.save_models,
        output_dir=output_dir,
        timeout=args.timeout,
        graph_timeout=args.graph_timeout,
        jobs=args.jobs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        val_ratio=args.val_ratio,
        patience=args.patience,
        val_split_seed=args.val_split_seed,
        min_epochs=args.min_epochs,
    )


if __name__ == "__main__":
    main()
