#!/usr/bin/env python3
"""
Evaluate algorithm selection over multiple train/test splits using WL (Weisfeiler-Lehman) kernel.

Expects --splits-dir to point at a division folder containing seed subdirs, e.g.:
  data/cp26/performance_splits/smtcomp24/ABV/
    seed0/train.json, test.json
    seed10/train.json, test.json
    ...

For each split (seed):
  1. Load train.json and test.json (performance JSON format).
  2. Train a WL-based pairwise model on the train set (graphs from SMT instances).
  3. Evaluate on train and test sets.
  4. Compute metrics (solve rate, PAR-2, gap closed vs SBS/VBS).

Results are aggregated across splits (mean ± std) and optionally saved to summary.json.
No feature CSV is required; WL builds graphs from instance paths in the performance data.
"""

import argparse
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from src.evaluate import as_evaluate
from src.performance import parse_performance_json, MultiSolverDataset
from src.pwc_wl import PwcWlSelector, train_pwc_wl


def compute_metrics(result_dataset, multi_perf_data):
    """
    Compute evaluation metrics for a result dataset.
    Handles edge case where SBS and VBS are identical (no division by zero).

    Args:
        result_dataset: SingleSolverDataset with algorithm selection results
        multi_perf_data: MultiSolverDataset for comparison metrics

    Returns:
        Dictionary of metrics (no solve_rate fields; gap_cls_* used for aggregation).
    """
    total_count = len(result_dataset)
    solved_count = result_dataset.get_solved_count()

    total_par2 = sum(result_dataset.get_par2(path) for path in result_dataset.keys())
    avg_par2 = total_par2 / total_count if total_count > 0 else 0.0

    sbs_dataset = multi_perf_data.get_best_solver_dataset()
    sbs_solved = sbs_dataset.get_solved_count()
    total_par2_sbs = sum(sbs_dataset.get_par2(path) for path in sbs_dataset.keys())
    avg_par2_sbs = total_par2_sbs / total_count if total_count > 0 else 0.0

    vbs_dataset = multi_perf_data.get_virtual_best_solver_dataset()
    vbs_solved = vbs_dataset.get_solved_count()
    total_par2_vbs = sum(vbs_dataset.get_par2(path) for path in vbs_dataset.keys())
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
        "sbs_name": sbs_dataset.get_solver_name(),
        "sbs_solved": sbs_solved,
        "sbs_avg_par2": avg_par2_sbs,
        "vbs_solved": vbs_solved,
        "vbs_avg_par2": avg_par2_vbs,
        "gap_cls_solved": gap_cls_solved,
        "gap_cls_par2": gap_cls_par2,
    }


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


def evaluate_multi_splits_wl(
    splits_dir: Path,
    *,
    wl_iter: int = 2,
    graph_timeout: int = 10,
    save_models: bool = False,
    output_dir: Path | None = None,
    timeout: float = 1200.0,
    benchmark_root: Path | str | None = None,
) -> dict:
    """
    Run train/test evaluation for each split (seed) using WL kernel, then aggregate metrics.

    Args:
        splits_dir: Directory containing seed subdirs (e.g. data/cp26/performance_splits/smtcomp24/ABV)
        wl_iter: Weisfeiler-Lehman iteration count
        graph_timeout: Timeout in seconds for building graph per instance
        save_models: Save model per split (requires output_dir)
        output_dir: Where to write summary.json and optional per-split outputs
        timeout: PAR-2 timeout in seconds for performance data
        benchmark_root: Root directory for instance paths; use when JSON paths are relative (e.g. ABV/...).

    Returns:
        Dict with division, n_seeds, per-split results, and aggregated metrics.
    """
    splits_dir = Path(splits_dir).resolve()
    if not splits_dir.is_dir():
        raise ValueError(f"Splits directory does not exist: {splits_dir}")

    division = splits_dir.name
    seed_entries = discover_seed_dirs(splits_dir)
    if not seed_entries:
        raise ValueError(
            f"No seed dirs (seedN with train.json and test.json) found in {splits_dir}"
        )

    n_seeds = len(seed_entries)
    logging.info(
        f"Starting WL evaluation over {n_seeds} splits for division {division} "
        f"(wl_iter={wl_iter}, graph_timeout={graph_timeout}s)"
    )

    seed_results: list[dict] = []

    for seed_val, seed_dir in seed_entries:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Seed {seed_val} ({seed_dir.name})")
        logging.info(f"{'=' * 60}")

        train_path = seed_dir / "train.json"
        test_path = seed_dir / "test.json"
        train_data = parse_performance_json(str(train_path), timeout)
        test_data = parse_performance_json(str(test_path), timeout)
        if benchmark_root is not None:
            root = Path(benchmark_root).resolve()
            if not root.is_dir():
                raise ValueError(f"--benchmark-root is not a directory: {root}")
            train_data = MultiSolverDataset(
                {str(root / p): train_data[p] for p in train_data.keys()},
                train_data.get_solver_id_dict(),
                train_data.get_timeout(),
            )
            test_data = MultiSolverDataset(
                {str(root / p): test_data[p] for p in test_data.keys()},
                test_data.get_solver_id_dict(),
                test_data.get_timeout(),
            )
            if seed_val == seed_entries[0][0]:
                logging.info("Instance paths rebased under benchmark root: %s", root)

        logging.info(f"Train instances: {len(train_data)}, Test instances: {len(test_data)}")

        if save_models and output_dir:
            model_save_dir = output_dir / "models" / f"seed{seed_val}"
            model_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            model_save_dir = Path(tempfile.mkdtemp())

        train_pwc_wl(
            train_data,
            wl_iter=wl_iter,
            save_dir=str(model_save_dir),
            graph_timeout=graph_timeout,
        )

        as_model = PwcWlSelector.load(str(model_save_dir / "model.joblib"))

        train_output_csv = None
        test_output_csv = None
        if output_dir:
            out_train = output_dir / "train"
            out_test = output_dir / "test"
            out_train.mkdir(parents=True, exist_ok=True)
            out_test.mkdir(parents=True, exist_ok=True)
            train_output_csv = str(out_train / f"seed{seed_val}.csv")
            test_output_csv = str(out_test / f"seed{seed_val}.csv")

        train_result = as_evaluate(
            as_model, train_data, write_csv_path=train_output_csv
        )
        train_metrics = compute_metrics(train_result, train_data)

        test_result = as_evaluate(as_model, test_data, write_csv_path=test_output_csv)
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
            f"  Train: solved {train_metrics['solved']}/{n_train}, "
            f"solve_rate {train_sr:.2f}%, gap_cls_par2 {train_metrics['gap_cls_par2']:.4f}"
        )
        logging.info(
            f"  Test:  solved {test_metrics['solved']}/{n_test}, "
            f"solve_rate {test_sr:.2f}%, gap_cls_par2 {test_metrics['gap_cls_par2']:.4f}"
        )

        if not save_models:
            shutil.rmtree(model_save_dir, ignore_errors=True)

    test_metrics_list = [r["test_metrics"] for r in seed_results]
    train_metrics_list = [r["train_metrics"] for r in seed_results]

    def agg(key: str):
        return np.mean([m[key] for m in test_metrics_list]), np.std(
            [m[key] for m in test_metrics_list]
        )

    def agg_train(key: str):
        return np.mean([m[key] for m in train_metrics_list]), np.std(
            [m[key] for m in train_metrics_list]
        )

    aggregated = {
        "train": {
            "gap_cls_solved_mean": agg_train("gap_cls_solved")[0],
            "gap_cls_solved_std": agg_train("gap_cls_solved")[1],
            "gap_cls_par2_mean": agg_train("gap_cls_par2")[0],
            "gap_cls_par2_std": agg_train("gap_cls_par2")[1],
        },
        "test": {
            "gap_cls_solved_mean": agg("gap_cls_solved")[0],
            "gap_cls_solved_std": agg("gap_cls_solved")[1],
            "gap_cls_par2_mean": agg("gap_cls_par2")[0],
            "gap_cls_par2_std": agg("gap_cls_par2")[1],
        },
    }

    results = {
        "division": division,
        "n_seeds": n_seeds,
        "seed_values": [s for s, _ in seed_entries],
        "model_type": "WL",
        "wl_iter": wl_iter,
        "graph_timeout": graph_timeout,
        "benchmark_root": str(benchmark_root) if benchmark_root is not None else None,
        "splits_dir": str(splits_dir),
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
        logging.info(f"Saved summary to {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WL-based algorithm selection over multiple train/test splits (per seed)"
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        required=True,
        help="Directory containing seed subdirs (e.g. data/cp26/performance_splits/smtcomp24/ABV)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="PAR-2 timeout in seconds (default: 1200)",
    )
    parser.add_argument(
        "--wl-iter",
        type=int,
        default=2,
        help="Weisfeiler-Lehman iteration count (default: 2)",
    )
    parser.add_argument(
        "--graph-timeout",
        type=int,
        default=10,
        help="Timeout in seconds for graph build per instance (default: 10)",
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
        help="Save model per split (requires --output-dir)",
    )
    parser.add_argument(
        "--benchmark-root",
        type=str,
        default=None,
        help="Root directory for instance paths; required when paths are relative (e.g. ABV/...).",
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

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_multi_splits_wl(
        Path(args.splits_dir),
        wl_iter=args.wl_iter,
        graph_timeout=args.graph_timeout,
        save_models=args.save_models,
        output_dir=output_dir,
        timeout=args.timeout,
        benchmark_root=Path(args.benchmark_root) if args.benchmark_root else None,
    )

    agg = results["aggregated"]
    logging.info("\n" + "=" * 60)
    logging.info("Multi-splits summary (WL) — %s", results["division"])
    logging.info("=" * 60)
    logging.info("Model: %s (wl_iter=%s, graph_timeout=%ss)", results["model_type"], results["wl_iter"], results["graph_timeout"])
    logging.info("Splits (seeds): %s", results["seed_values"])
    logging.info("")
    tr = agg["train"]
    logging.info("Train: gap_cls_solved %.4f ± %.4f, gap_cls_par2 %.4f ± %.4f",
                 tr["gap_cls_solved_mean"], tr["gap_cls_solved_std"],
                 tr["gap_cls_par2_mean"], tr["gap_cls_par2_std"])
    t = agg["test"]
    logging.info("Test:  gap_cls_solved %.4f ± %.4f, gap_cls_par2 %.4f ± %.4f",
                 t["gap_cls_solved_mean"], t["gap_cls_solved_std"],
                 t["gap_cls_par2_mean"], t["gap_cls_par2_std"])
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
