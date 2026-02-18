#!/usr/bin/env python3
"""
Evaluate WL-based PWC over multiple train/test splits (evaluate_multi_splits_wl).

Uses a WL feature directory (--wl-dir) containing level_0.csv, level_1.csv, ...
and failed_paths.txt. Expects --splits-dir to point at a division folder with
seed subdirs (seed0/train.json, test.json, ...).

For each split:
  1. Load train.json and test.json.
  2. Train PWC with train_wl_pwc(wl_dir=..., wl_iter=...).
  3. Evaluate on train and test sets.
  4. Compute metrics (solve rate, PAR-2, gap closed vs SBS/VBS).

Results are aggregated across splits and optionally saved to summary.json.
"""

import argparse
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from src.evaluate import as_evaluate, as_evaluate_parallel, _load_pwc_selector
from src.feature import validate_feature_coverage
from src.performance import parse_performance_json
from src.pwc import (
    PwcSelector,
    WL_FAILED_PATHS_FILENAME,
    train_wl_pwc,
    _load_path_list,
    _wl_level_csv_paths,
)


def compute_metrics(result_dataset, multi_perf_data):
    """
    Compute evaluation metrics for a result dataset.
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
    """Find seed subdirs under splits_dir with train.json and test.json."""
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
    wl_dir: str | Path,
    wl_iter: int,
    *,
    save_models: bool = False,
    output_dir: Path | None = None,
    timeout: float = 1200.0,
    svm_c: float = 1.0,
    random_seed: int = 42,
    jobs: int = 1,
) -> dict:
    """
    Run train/test evaluation for each split using WL features from wl_dir.
    jobs: parallel workers for evaluation; 1 = sequential.
    """
    splits_dir = Path(splits_dir).resolve()
    wl_dir = Path(wl_dir).resolve()
    if not splits_dir.is_dir():
        raise ValueError(f"Splits directory does not exist: {splits_dir}")
    if not wl_dir.is_dir():
        raise ValueError(f"WL directory does not exist: {wl_dir}")

    division = splits_dir.name
    seed_entries = discover_seed_dirs(splits_dir)
    if not seed_entries:
        raise ValueError(
            f"No seed dirs (seedN with train.json and test.json) found in {splits_dir}"
    )

    feature_csv_paths = _wl_level_csv_paths(wl_dir, wl_iter)
    first_seed_dir = seed_entries[0][1]
    train_data_0 = parse_performance_json(str(first_seed_dir / "train.json"), timeout)
    test_data_0 = parse_performance_json(str(first_seed_dir / "test.json"), timeout)
    all_instance_paths = set(train_data_0.keys()) | set(test_data_0.keys())

    failed_paths_file = wl_dir / WL_FAILED_PATHS_FILENAME
    failed_set = set(_load_path_list(failed_paths_file)) if failed_paths_file.exists() else set()
    instances_requiring_features = all_instance_paths - failed_set
    if failed_set:
        logging.info("%d instances in WL failed list (use fallback solver)", len(failed_set))

    logging.info("Validating feature coverage for %d instances...", len(instances_requiring_features))
    missing_instances, instance_missing_in_csvs = validate_feature_coverage(
        instances_requiring_features, feature_csv_paths
    )
    if missing_instances:
        error_msg = (
            f"ERROR: {len(missing_instances)} instance(s) missing in ALL feature CSV(s).\n"
            f"  Missing (first 10): {list(missing_instances)[:10]}"
        )
        logging.error(error_msg)
        raise ValueError(error_msg)
    if instance_missing_in_csvs:
        error_msg = (
            f"ERROR: {len(instance_missing_in_csvs)} instance(s) missing in some CSV(s)."
        )
        logging.error(error_msg)
        raise ValueError(error_msg)

    n_seeds = len(seed_entries)
    logging.info("Starting WL evaluation over %d splits for division %s", n_seeds, division)

    seed_results: list[dict] = []

    for seed_val, seed_dir in seed_entries:
        logging.info("\n%s", "=" * 60)
        logging.info("Seed %s (%s)", seed_val, seed_dir.name)
        logging.info("%s", "=" * 60)

        train_path = seed_dir / "train.json"
        test_path = seed_dir / "test.json"
        train_data = parse_performance_json(str(train_path), timeout)
        test_data = parse_performance_json(str(test_path), timeout)

        logging.info("Train instances: %d, Test instances: %d", len(train_data), len(test_data))

        if save_models and output_dir:
            model_save_dir = output_dir / "models" / f"seed{seed_val}"
            model_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            model_save_dir = Path(tempfile.mkdtemp())

        train_wl_pwc(
            train_data,
            save_dir=model_save_dir,
            wl_dir=wl_dir,
            wl_iter=wl_iter,
            svm_c=svm_c,
            random_seed=random_seed,
        )

        model_path = str(model_save_dir / "model.joblib")
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
            train_result = as_evaluate_parallel(
                list(train_data.keys()),
                _load_pwc_selector,
                (model_path,),
                train_data,
                n_workers=jobs,
                write_csv_path=train_output_csv,
                show_progress=False,
            )
            test_result = as_evaluate_parallel(
                list(test_data.keys()),
                _load_pwc_selector,
                (model_path,),
                test_data,
                n_workers=jobs,
                write_csv_path=test_output_csv,
                show_progress=False,
            )
        else:
            as_model = PwcSelector.load(model_path)
            train_result = as_evaluate(
                as_model, train_data, write_csv_path=train_output_csv
            )
            test_result = as_evaluate(as_model, test_data, write_csv_path=test_output_csv)
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
            train_metrics["solved"], n_train, train_sr, train_metrics["gap_cls_par2"],
        )
        logging.info(
            "  Test:  solved %d/%d, solve_rate %.2f%%, gap_cls_par2 %.4f",
            test_metrics["solved"], n_test, test_sr, test_metrics["gap_cls_par2"],
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
        "splits_dir": str(splits_dir),
        "wl_dir": str(wl_dir),
        "wl_iter": wl_iter,
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

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WL-based PWC over multiple train/test splits"
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        required=True,
        help="Directory containing seed subdirs (e.g. data/cp26/performance_splits/smtcomp24/BV)",
    )
    parser.add_argument(
        "--wl-dir",
        type=str,
        required=True,
        help="WL feature directory (level_0.csv, level_1.csv, ..., failed_paths.txt)",
    )
    parser.add_argument(
        "--wl-iter",
        type=int,
        default=2,
        help="WL iteration count; uses level_0.csv .. level_{wl_iter}.csv (default: 2)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Timeout in seconds (default: 1200)",
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
        "--svm-c",
        type=float,
        default=1.0,
        help="SVM C (default: 1.0)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for tie-breaking (default: 42)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers for evaluation; 1 = sequential (default: 1)",
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
        Path(args.wl_dir),
        args.wl_iter,
        save_models=args.save_models,
        output_dir=output_dir,
        timeout=args.timeout,
        svm_c=args.svm_c,
        random_seed=args.random_seed,
        jobs=args.jobs,
    )

    agg = results["aggregated"]
    logging.info("\n" + "=" * 60)
    logging.info("Multi-splits WL summary — %s", results["division"])
    logging.info("=" * 60)
    logging.info("Model: %s (wl_dir=%s, wl_iter=%s)", results["model_type"], results["wl_dir"], results["wl_iter"])
    logging.info("Splits (seeds): %s", results["seed_values"])
    logging.info("")
    tr = agg["train"]
    logging.info(
        "Train: gap_cls_solved %.4f ± %.4f, gap_cls_par2 %.4f ± %.4f",
        tr["gap_cls_solved_mean"], tr["gap_cls_solved_std"],
        tr["gap_cls_par2_mean"], tr["gap_cls_par2_std"],
    )
    t = agg["test"]
    logging.info(
        "Test:  gap_cls_solved %.4f ± %.4f, gap_cls_par2 %.4f ± %.4f",
        t["gap_cls_solved_mean"], t["gap_cls_solved_std"],
        t["gap_cls_par2_mean"], t["gap_cls_par2_std"],
    )
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
