#!/usr/bin/env python3
"""
Evaluate WL graph-kernel PWC (PwcWlSelector) over multiple train/test splits.

This script trains the WL-based selector from src.pwc_wl on each split and
evaluates it on train and test sets, aggregating metrics across seeds.
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from src.defaults import DEFAULT_BENCHMARK_ROOT
from src.evaluate import as_evaluate, as_evaluate_parallel, compute_metrics
from src.performance import (
    filter_training_instances,
    MultiSolverDataset,
    parse_as_perf_csv,
    parse_performance_json,
)
from src.pwc_wl import PwcWlSelector, train_pwc_wl


def _load_pwc_wl_selector(model_path: str):
    """Load PwcWlSelector from path (module-level for multiprocessing pickling)."""
    return PwcWlSelector.load(model_path)


def discover_seed_dirs(splits_dir: Path) -> list[tuple[int, Path]]:
    """Find seed subdirs under splits_dir that contain train.json and test.json."""
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


DEFAULT_LITE_DIR = Path("data/results/lite")

CSV_HEADER = [
    "benchmark",
    "selected",
    "solved",
    "runtime",
    "solver_runtime",
    "overhead",
    "feature_fail",
]


def _load_lite_lookup(path: Path) -> dict[str, dict]:
    """Load Lite eval CSV; return benchmark -> row dict."""
    out: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bench = (row.get("benchmark") or "").strip()
            if bench:
                out[bench] = row
    return out


def _load_eval_csv(path: Path) -> list[dict]:
    """Load eval CSV; return list of row dicts."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_eval_csv(path: Path, rows: list[dict]) -> None:
    """Write rows using CSV_HEADER order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow([row.get(k, "") for k in CSV_HEADER])


def _merge_with_lite(
    wl_rows: list[dict],
    lite_lookup: dict[str, dict],
    timeout: float,
) -> list[dict]:
    """
    Replace feature_fail rows with Lite results.

    For feature_fail rows:
      overhead_out = wl_overhead + lite_overhead
      runtime_out  = lite_solver_runtime + overhead_out  (capped at timeout)
    """
    out: list[dict] = []
    for row in wl_rows:
        bench = (row.get("benchmark") or "").strip()
        try:
            ff = int(row.get("feature_fail", 0) or 0)
        except (ValueError, TypeError):
            ff = 0

        if ff == 0:
            out.append(row)
            continue

        if bench not in lite_lookup:
            raise ValueError(
                f"Missing Lite row for feature-fail benchmark: {bench!r}"
            )
        lt_row = lite_lookup[bench]

        raw_wl_overhead = row.get("overhead", "")
        try:
            wl_overhead = float(raw_wl_overhead) if raw_wl_overhead not in ("", None) else 0.0
        except (TypeError, ValueError):
            wl_overhead = 0.0

        try:
            lt_solver_runtime = float(lt_row.get("solver_runtime") or 0.0)
        except (TypeError, ValueError):
            lt_solver_runtime = 0.0
        try:
            lt_solved = int(lt_row.get("solved") or 0)
        except (TypeError, ValueError):
            lt_solved = 0
        raw_lt_overhead = lt_row.get("overhead", "")
        try:
            lt_overhead = float(raw_lt_overhead) if raw_lt_overhead not in ("", None) else 0.0
        except (TypeError, ValueError):
            lt_overhead = 0.0

        overhead_out = wl_overhead + lt_overhead
        runtime_out = lt_solver_runtime + overhead_out

        if runtime_out > timeout:
            solved = 0
            runtime_out = timeout
            overhead_out = max(0.0, timeout - lt_solver_runtime)
        else:
            solved = lt_solved

        out.append(
            {
                "benchmark": bench,
                "selected": lt_row.get("selected", ""),
                "solved": str(solved),
                "runtime": str(runtime_out),
                "solver_runtime": str(lt_solver_runtime),
                "overhead": f"{overhead_out:.6f}",
                "feature_fail": "1",
            }
        )
    return out


def evaluate_multi_splits_wl(
    splits_dir: Path,
    *,
    benchmark_root: Path | str | None = None,
    save_models: bool = False,
    output_dir: Path | None = None,
    timeout: float = 1200.0,
    wl_iter: int = 2,
    graph_timeout: int = 5,
    lite_dir: Path | str | None = None,
    jobs: int = 1,
    skip_easy_unsolvable: bool = False,
    skip_trivial_under: float = 24.0,
    seeds: list[int] | None = None,
) -> dict:
    """
    Run WL graph-kernel PWC train/eval for each seed under splits_dir.

    When skip_easy_unsolvable is True, training uses only instances that are
    not VBS-unsolvable and not trivial (all solvers solved with runtime <= skip_trivial_under).
    Train and test evaluation still use the full splits.
    """
    splits_dir = Path(splits_dir).resolve()
    if not splits_dir.is_dir():
        raise ValueError(f"Splits directory does not exist: {splits_dir}")

    root = Path(benchmark_root or DEFAULT_BENCHMARK_ROOT).resolve()
    if not root.is_dir():
        raise ValueError(f"Benchmark root is not a directory: {root}")

    division = splits_dir.name

    lite_root = Path(lite_dir or DEFAULT_LITE_DIR).resolve()
    lite_division_dir = lite_root / division
    if not lite_division_dir.is_dir():
        raise FileNotFoundError(
            f"Lite results dir not found for division {division}: {lite_division_dir}"
        )

    seed_entries = discover_seed_dirs(splits_dir)
    if not seed_entries:
        raise ValueError(
            f"No seed dirs (seedN with train.json and test.json) found in {splits_dir}"
        )
    if seeds is not None:
        seed_set = set(seeds)
        seed_entries = [(s, d) for s, d in seed_entries if s in seed_set]
        if not seed_entries:
            raise ValueError(f"No matching seed dirs for seeds {sorted(seed_set)} in {splits_dir}")

    n_seeds = len(seed_entries)
    logging.info(
        "Starting WL graph-kernel evaluation over %d splits for division %s",
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

        n_train = len(train_data)
        n_test = len(test_data)
        logging.info("Train instances: %d, Test instances: %d", n_train, n_test)

        if skip_easy_unsolvable:
            paths_to_keep, filter_stats = filter_training_instances(
                train_data,
                skip_unsolvable=True,
                skip_trivial_under=skip_trivial_under,
            )
            train_data_for_training = MultiSolverDataset(
                {p: train_data[p] for p in paths_to_keep},
                train_data.get_solver_id_dict(),
                train_data.get_timeout(),
            )
            logging.info(
                "Training on %d instances (dropped %d unsolvable, %d trivial); train eval uses full %d",
                filter_stats["n_kept"],
                filter_stats["n_unsolvable"],
                filter_stats["n_trivial"],
                n_train,
            )
        else:
            train_data_for_training = train_data

        if save_models and output_dir:
            model_save_dir = output_dir / "models" / "pwc_wl" / division / f"seed{seed_val}"
            model_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            model_save_dir = Path(tempfile.mkdtemp())

        # Train WL graph-kernel selector on (possibly filtered) training data.
        train_pwc_wl(
            train_data_for_training,
            wl_iter=wl_iter,
            save_dir=str(model_save_dir),
            graph_timeout=graph_timeout,
            jobs=jobs,
        )

        model_path = model_save_dir / "model.joblib"
        model_path_str = str(model_path)

        # Evaluate on train and test (parallel when jobs > 1).
        if output_dir:
            seed_out_dir = output_dir / f"seed{seed_val}"
        else:
            seed_out_dir = Path(tempfile.mkdtemp())
        seed_out_dir.mkdir(parents=True, exist_ok=True)
        test_output_csv = seed_out_dir / "test_eval.csv"

        if jobs > 1:
            train_result = as_evaluate_parallel(
                list(train_data.keys()),
                _load_pwc_wl_selector,
                (model_path_str,),
                train_data,
                n_workers=jobs,
                write_csv_path=None,
                show_progress=True,
                csv_benchmark_root=root,
            )
            train_metrics = compute_metrics(train_result, train_data)

            test_result = as_evaluate_parallel(
                list(test_data.keys()),
                _load_pwc_wl_selector,
                (model_path_str,),
                test_data,
                n_workers=jobs,
                write_csv_path=str(test_output_csv),
                show_progress=True,
                csv_benchmark_root=root,
            )
        else:
            selector = PwcWlSelector.load(model_path)
            train_result = as_evaluate(
                selector,
                train_data,
                write_csv_path=None,
                show_progress=True,
                csv_benchmark_root=root,
            )
            train_metrics = compute_metrics(train_result, train_data)

            as_evaluate(
                selector,
                test_data,
                write_csv_path=str(test_output_csv),
                show_progress=True,
                csv_benchmark_root=root,
            )

        lt_seed_dir = lite_division_dir / f"seed{seed_val}"
        if not lt_seed_dir.is_dir():
            raise FileNotFoundError(f"Lite seed dir not found: {lt_seed_dir}")
        lt_test_csv = lt_seed_dir / "test_eval.csv"
        if not lt_test_csv.is_file():
            raise FileNotFoundError(f"Lite test CSV not found: {lt_test_csv}")

        fusion_test_rows = _load_eval_csv(test_output_csv)
        lite_test_lookup = _load_lite_lookup(lt_test_csv)
        merged_test = _merge_with_lite(fusion_test_rows, lite_test_lookup, timeout)
        _write_eval_csv(test_output_csv, merged_test)
        n_fb_test = sum(1 for r in merged_test if r.get("feature_fail") == "1")
        logging.info(
            "  Merged test: %d/%d rows from Lite fallback", n_fb_test, len(merged_test)
        )

        test_result = parse_as_perf_csv(str(test_output_csv), timeout)
        test_metrics = compute_metrics(test_result, test_data)

        seed_results.append(
            {
                "seed": seed_val,
                "train_size": n_train,
                "test_size": n_test,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }
        )

        train_gap_pct = (
            train_metrics["gap_cls_par2"] * 100
            if train_metrics.get("gap_cls_par2") is not None
            else 0.0
        )
        test_gap_pct = (
            test_metrics["gap_cls_par2"] * 100
            if test_metrics.get("gap_cls_par2") is not None
            else 0.0
        )
        logging.info(
            "  Train: solved %d/%d, gap closed (PAR-2): %.2f%%",
            train_metrics["solved"],
            n_train,
            train_gap_pct,
        )
        logging.info(
            "  Test:  solved %d/%d, gap closed (PAR-2): %.2f%%",
            test_metrics["solved"],
            n_test,
            test_gap_pct,
        )

        if not save_models:
            shutil.rmtree(model_save_dir, ignore_errors=True)

    # Aggregate metrics across seeds.
    test_metrics_list = [r["test_metrics"] for r in seed_results]
    train_metrics_list = [r["train_metrics"] for r in seed_results]

    def _agg(m_list: list[dict], key: str) -> tuple[float, float]:
        vals = [m[key] for m in m_list]
        return float(np.mean(vals)), float(np.std(vals))

    aggregated = {
        "train": {
            "gap_cls_solved_mean": _agg(train_metrics_list, "gap_cls_solved")[0],
            "gap_cls_solved_std": _agg(train_metrics_list, "gap_cls_solved")[1],
            "gap_cls_par2_mean": _agg(train_metrics_list, "gap_cls_par2")[0],
            "gap_cls_par2_std": _agg(train_metrics_list, "gap_cls_par2")[1],
        },
        "test": {
            "gap_cls_solved_mean": _agg(test_metrics_list, "gap_cls_solved")[0],
            "gap_cls_solved_std": _agg(test_metrics_list, "gap_cls_solved")[1],
            "gap_cls_par2_mean": _agg(test_metrics_list, "gap_cls_par2")[0],
            "gap_cls_par2_std": _agg(test_metrics_list, "gap_cls_par2")[1],
        },
    }

    results = {
        "division": division,
        "n_seeds": n_seeds,
        "seed_values": [s for s, _ in seed_entries],
        "model_type": "pwc_wl",
        "splits_dir": str(splits_dir),
        "benchmark_root": str(root),
        "wl_iter": wl_iter,
        "graph_timeout": graph_timeout,
        "seeds": seed_results,
        "aggregated": aggregated,
    }

    if output_dir:
        summary_path = Path(output_dir) / "summary.json"
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
    logging.info("Multi-splits WL summary — %s", results["division"])
    logging.info("%s", "=" * 60)
    logging.info(
        "Model: %s (wl_iter=%d, graph_timeout=%d)",
        results["model_type"],
        results["wl_iter"],
        results["graph_timeout"],
    )
    logging.info("Splits (seeds): %s", results["seed_values"])
    logging.info("")
    tr = agg["train"]
    logging.info(
        "Train: gap closed (solved) %.2f%% ± %.2f%%, gap closed (PAR-2) %.2f%% ± %.2f%%",
        tr["gap_cls_solved_mean"] * 100,
        tr["gap_cls_solved_std"] * 100,
        tr["gap_cls_par2_mean"] * 100,
        tr["gap_cls_par2_std"] * 100,
    )
    t = agg["test"]
    logging.info(
        "Test:  gap closed (solved) %.2f%% ± %.2f%%, gap closed (PAR-2) %.2f%% ± %.2f%%",
        t["gap_cls_solved_mean"] * 100,
        t["gap_cls_solved_std"] * 100,
        t["gap_cls_par2_mean"] * 100,
        t["gap_cls_par2_std"] * 100,
    )
    logging.info("%s", "=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate WL graph-kernel PWC (PwcWlSelector) over multiple splits"
    )
    parser.add_argument(
        "--logic",
        type=str,
        default=None,
        help="Division (e.g. BV). Sets splits and default output paths.",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Directory with seed subdirs (train.json, test.json). If omitted with --logic, auto-set.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=str,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Root directory for benchmark paths (default: project default).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument(
        "--wl-iter",
        type=int,
        default=1,
        help="Weisfeiler-Lehman iteration count (default: 1)",
    )
    parser.add_argument(
        "--graph-timeout",
        type=int,
        default=5,
        help="Graph build timeout in seconds (default: 5)",
    )
    _default_jobs = max(1, (os.cpu_count() or 4) - 1)
    parser.add_argument(
        "--jobs",
        type=int,
        default=_default_jobs,
        help=f"Parallel workers for graph building and evaluation; 1 = sequential (default: {_default_jobs})",
    )
    parser.add_argument(
        "--lite-dir",
        type=str,
        default=str(DEFAULT_LITE_DIR),
        help=f"Lite results root for fallback (default: {DEFAULT_LITE_DIR})",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Skip easy/unsolvable instances for training (alias for --skip-easy-unsolvable).",
    )
    parser.add_argument(
        "--skip-easy-unsolvable",
        action="store_true",
        help="Exclude VBS-unsolvable and trivial instances from training; train/test eval still use full splits.",
    )
    parser.add_argument(
        "--skip-trivial-under",
        type=float,
        default=24.0,
        help="When --filter/--skip-easy-unsolvable: exclude train instances where every solver solved with runtime <= this (default: 24).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Optional list of seed values to run (e.g. --seeds 0 10 20).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    if args.logic and not args.splits_dir:
        splits_base = Path("data/train_test_splits")
        args.splits_dir = str(splits_base / args.logic)
        if args.output_dir is None:
            # Default: save under data/results/wl_grakel/iter1/<logic>
            base = Path("data/results/wl_grakel") / f"iter{args.wl_iter}"
            args.output_dir = str(base / args.logic)
            args.save_models = True

    if not args.splits_dir:
        parser.error("Either --splits-dir or --logic must be provided")

    if getattr(args, "filter", False):
        args.skip_easy_unsolvable = True

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    evaluate_multi_splits_wl(
        Path(args.splits_dir),
        benchmark_root=args.benchmark_root,
        save_models=args.save_models,
        output_dir=output_dir,
        timeout=args.timeout,
        wl_iter=args.wl_iter,
        graph_timeout=args.graph_timeout,
        lite_dir=args.lite_dir,
        jobs=args.jobs,
        skip_easy_unsolvable=args.skip_easy_unsolvable,
        skip_trivial_under=args.skip_trivial_under,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()

