#!/usr/bin/env python3
"""
Evaluate algorithm selection over multiple train/test splits (evaluate_multi_splits).

Use --logic (e.g. BV, QF_BV) to run one division; omit --logic to run all logics found in feature dir(s).
  Splits are read from data/cp26/performance_splits/smtcomp24/<logic>/ (seed0/, seed10/, ... with train.json, test.json).

Features: --features-dir DIR [DIR ...] can be given one or more directories. Each must have one
  logic folder per division (e.g. data/features/syntactic, data/features/desc/all-mpnet-base-v2).
  Each division folder must contain features.csv and extraction_times.csv. Feature vectors from
  multiple dirs are concatenated; every instance must appear in every features.csv (no missing).
  Extraction times from all dirs are summed per path; each feature source's time is capped at FEATURE_TIMEOUT (5s) before summing, then used as overhead for evaluation metrics.

For each split (seed):
  1. Load train.json and test.json (performance JSON format).
  2. Train a model on the train set.
  3. Evaluate on train and test sets.
  4. Compute metrics (solve rate, PAR-2, gap closed vs SBS/VBS).

Results are aggregated across splits (mean ± std) and saved to summary.json. Per-seed CSVs
are written under output_dir as seed0/train_eval.csv, seed0/test_eval.csv, seed10/..., etc.
When output_dir is set, training logs are saved to output_dir/train_log/seed{N}.log per seed.
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
from src.evaluate import compute_metrics
from src.evaluate import load_extraction_times_csv
from src.evaluate import load_failed_paths_from_extraction_times_csv
from src.feature import validate_feature_coverage
from src.performance import parse_performance_json
from src.pwc import PwcSelector, train_pwc
from src.utils import normalize_path

# Per-instance extraction-time overhead cap (seconds) when computing metrics
FEATURE_TIMEOUT = 5.0
# Default splits base when using --logic (relative to project root)
SPLITS_BASE = "data/cp26/performance_splits/smtcomp24"


def _to_python_for_json(obj) -> int | float | list | dict | str | bool | None:
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_python_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python_for_json(x) for x in obj]
    return obj


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


def evaluate_multi_splits(
    splits_dir: Path,
    feature_csv_path: str | list[str],
    extraction_time_by_path: dict[str, float],
    *,
    failed_paths_from_csv: set[str] | None = None,
    xg_flag: bool = False,
    save_models: bool = False,
    output_dir: Path | None = None,
    timeout: float = 1200.0,
    svm_c: float = 1.0,
    random_seed: int = 42,
) -> dict:
    """
    Run train/test evaluation for each split (seed) under splits_dir (seed0/, seed10/, ...),
    then aggregate metrics across splits. Division name for outputs is taken from splits_dir.name.
    Extraction times are added as overhead when computing metrics; each feature source's time
    is capped at FEATURE_TIMEOUT (5s) before summing. Only instances with failed=1 in
    extraction_times.csv are excluded from training and get SBS at evaluation.

    Args:
        splits_dir: Directory containing seed subdirs (e.g. data/cp26/performance_splits/smtcomp24/ABV)
        feature_csv_path: Path or list of paths to feature CSV(s)
        extraction_time_by_path: Map normalized instance path -> extraction time (sec) for overhead
        failed_paths_from_csv: Set of normalized paths with failed=1 in extraction_times CSV; used for SBS at eval
        xg_flag: Use XGBoost if True else SVM
        save_models: Save model per split (requires output_dir)
        output_dir: Where to write summary.json and optional per-split outputs
        timeout: Timeout in seconds
        svm_c: SVM C
        random_seed: Random seed for tie-breaking

    Returns:
        Dict with division (from splits_dir.name), n_seeds, per-split results, and aggregated metrics.
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

    # Validate feature coverage using first seed's train+test. Instances in
    # failed_paths_from_csv (failed=1 in extraction_times) are excluded from
    # training and get SBS at eval, so we do not require them in any feature CSV.
    first_seed_dir = seed_entries[0][1]
    train_data_0 = parse_performance_json(
        str(first_seed_dir / "train.json"), timeout
    )
    test_data_0 = parse_performance_json(str(first_seed_dir / "test.json"), timeout)
    all_instance_paths = set(train_data_0.keys()) | set(test_data_0.keys())
    failed_set = failed_paths_from_csv or set()
    instances_requiring_features = all_instance_paths - failed_set

    if isinstance(feature_csv_path, list):
        logging.info(
            f"Validating feature coverage for {len(instances_requiring_features)} instances "
            f"(every path must appear in all {len(feature_csv_path)} feature CSV(s)); "
            f"{len(failed_set)} failed paths excluded from check",
        )
    else:
        logging.info(
            f"Validating feature coverage for {len(instances_requiring_features)} instances "
            f"({len(failed_set)} failed paths excluded from check)",
        )
    missing_instances, instance_missing_in_csvs = validate_feature_coverage(
        instances_requiring_features, feature_csv_path
    )
    if missing_instances:
        error_msg = (
            f"ERROR: {len(missing_instances)} instance(s) missing in ALL feature CSV(s).\n"
            f"  Missing (first 10): {list(missing_instances)[:10]}"
        )
        logging.error(error_msg)
        raise ValueError(error_msg)
    if isinstance(feature_csv_path, list) and instance_missing_in_csvs:
        sample = list(instance_missing_in_csvs.items())[:3]
        details = "; ".join(f"{p!r} missing in {len(csvs)} CSV(s)" for p, csvs in sample)
        error_msg = (
            f"ERROR: {len(instance_missing_in_csvs)} instance(s) missing in at least one feature CSV. "
            f"Every path (except failed=1 in extraction_times) must have features in all {len(feature_csv_path)} source(s). Example: {details}"
        )
        logging.error(error_msg)
        raise ValueError(error_msg)

    missing_from_extraction_times = {
        p for p in instances_requiring_features
        if normalize_path(p) not in extraction_time_by_path
    }
    if missing_from_extraction_times:
        error_msg = (
            f"ERROR: {len(missing_from_extraction_times)} instance(s) missing from all extraction_times.csv. "
            f"Every path that has features must appear in extraction_times.csv from every feature dir. "
            f"Missing (first 10): {list(missing_from_extraction_times)[:10]}"
        )
        logging.error(error_msg)
        raise ValueError(error_msg)

    n_seeds = len(seed_entries)
    logging.info(
        f"Starting evaluation over {n_seeds} splits for division {division}"
    )

    # Paths to exclude from training and use SBS at eval (only CSV label failed=1)
    failed_paths_list = list(failed_paths_from_csv or set())
    if failed_paths_list:
        logging.info(
            "%d instances with failed=1 (excluded from training, SBS at eval)",
            len(failed_paths_list),
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

        logging.info(
            "Train instances: %d, Test instances: %d",
            len(train_data),
            len(test_data),
        )

        if save_models and output_dir:
            model_save_dir = output_dir / "models" / f"seed{seed_val}"
            model_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            model_save_dir = Path(tempfile.mkdtemp())

        log_handler: logging.FileHandler | None = None
        try:
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

            logging.info("Training started for seed %d", seed_val)
            timeout_file: Path | None = None
            if failed_paths_list:
                timeout_file = Path(
                    tempfile.mkstemp(suffix=".txt", prefix="timeout_paths_")[1]
                )
                timeout_file.write_text("\n".join(failed_paths_list), encoding="utf-8")
            try:
                train_pwc(
                    train_data,
                    save_dir=str(model_save_dir),
                    xg_flag=xg_flag,
                    feature_csv_path=feature_csv_path,
                    svm_c=svm_c,
                    random_seed=random_seed,
                    timeout_instance_paths=str(timeout_file) if timeout_file else None,
                    extraction_time_by_path=extraction_time_by_path,
                    feature_timeout=None,
                )
            finally:
                if timeout_file and timeout_file.is_file():
                    timeout_file.unlink(missing_ok=True)

            as_model = PwcSelector.load(str(model_save_dir / "model.joblib"))
            if as_model.feature_csv_path is None:
                as_model.feature_csv_path = feature_csv_path
            as_model.failed_paths_from_csv = failed_paths_from_csv or set()
            as_model.extraction_time_by_path = extraction_time_by_path
            as_model.sbs_solver_id = train_data.get_best_solver_id()

            train_output_csv = None
            test_output_csv = None
            if output_dir:
                seed_out_dir = output_dir / f"seed{seed_val}"
                seed_out_dir.mkdir(parents=True, exist_ok=True)
                train_output_csv = str(seed_out_dir / "train_eval.csv")
                test_output_csv = str(seed_out_dir / "test_eval.csv")

            train_result = as_evaluate(
                as_model,
                train_data,
                write_csv_path=train_output_csv,
                extra_overhead_by_path=extraction_time_by_path,
            )
            train_metrics = compute_metrics(train_result, train_data)

            test_result = as_evaluate(
                as_model,
                test_data,
                write_csv_path=test_output_csv,
                extra_overhead_by_path=extraction_time_by_path,
            )
            test_metrics = compute_metrics(test_result, test_data)

            seed_results.append({
                "seed": seed_val,
                "train_size": len(train_data),
                "test_size": len(test_data),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            })

            n_train, n_test = len(train_data), len(test_data)
            train_gap_pct = (
                (train_metrics["gap_cls_par2"] * 100)
                if train_metrics.get("gap_cls_par2") is not None
                else 0.0
            )
            test_gap_pct = (
                (test_metrics["gap_cls_par2"] * 100)
                if test_metrics.get("gap_cls_par2") is not None
                else 0.0
            )
            logging.info(
                f"  Train: solved {train_metrics['solved']}/{n_train}, gap closed (PAR-2): {train_gap_pct:.2f}%"
            )
            logging.info(
                f"  Test:  solved {test_metrics['solved']}/{n_test}, gap closed (PAR-2): {test_gap_pct:.2f}%"
            )
        finally:
            if log_handler is not None:
                logging.getLogger().removeHandler(log_handler)
                log_handler.close()
            if not save_models:
                shutil.rmtree(model_save_dir, ignore_errors=True)

    test_metrics_list = [r["test_metrics"] for r in seed_results]
    train_metrics_list = [r["train_metrics"] for r in seed_results]

    def _agg(metrics_list: list[dict], key: str) -> tuple[float, float]:
        vals = [m[key] for m in metrics_list]
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
        "model_type": "XGBoost" if xg_flag else "SVM",
        "splits_dir": str(splits_dir),
        "feature_csv_path": feature_csv_path,
        "seeds": seed_results,
        "aggregated": aggregated,
    }

    if output_dir:
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(_to_python_for_json(results), f, indent=2)
        logging.info(f"Saved summary to {summary_path}")

    agg = results["aggregated"]
    logging.info("\n" + "=" * 60)
    logging.info("Multi-splits summary — %s", results["division"])
    logging.info("=" * 60)
    logging.info("Model: %s", results["model_type"])
    logging.info("Splits (seeds): %s", results["seed_values"])
    logging.info("")
    tr = agg["train"]
    logging.info(
        "Train: gap closed (solved) %.2f%% ± %.2f%%, gap closed (PAR-2) %.2f%% ± %.2f%%",
        tr["gap_cls_solved_mean"] * 100, tr["gap_cls_solved_std"] * 100,
        tr["gap_cls_par2_mean"] * 100, tr["gap_cls_par2_std"] * 100,
    )
    t = agg["test"]
    logging.info(
        "Test:  gap closed (solved) %.2f%% ± %.2f%%, gap closed (PAR-2) %.2f%% ± %.2f%%",
        t["gap_cls_solved_mean"] * 100, t["gap_cls_solved_std"] * 100,
        t["gap_cls_par2_mean"] * 100, t["gap_cls_par2_std"] * 100,
    )
    logging.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate algorithm selection over multiple train/test splits (per seed)"
    )
    parser.add_argument(
        "--logic",
        type=str,
        default=None,
        metavar="LOGIC",
        help="Division name (e.g. BV, QF_BV). If not given, run all logics found in feature dir(s).",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        nargs="+",
        required=True,
        metavar="DIR",
        help="One or more base dirs (e.g. synt + desc); each must contain <logic>/features.csv and extraction_times.csv.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Timeout in seconds (default: 1200)",
    )
    parser.add_argument(
        "--xg",
        action="store_true",
        help="Use XGBoost instead of SVM",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for results. With --logic, output is written to <output-dir>/<logic>/.",
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

    project_root = Path(__file__).resolve().parent.parent
    features_dirs = [
        (project_root / d).resolve() if not Path(d).is_absolute() else Path(d).resolve()
        for d in args.features_dir
    ]
    splits_base = (project_root / SPLITS_BASE).resolve()

    def discover_logics_from_feature_dir() -> list[str]:
        """Logics that have features in every feature dir and have splits in SPLITS_BASE."""
        first_base = features_dirs[0]
        candidates = [
            sub.name for sub in first_base.iterdir()
            if sub.is_dir()
            and (sub / "features.csv").is_file()
            and (sub / "extraction_times.csv").is_file()
        ]
        logics = [
            name for name in sorted(candidates)
            if (splits_base / name).is_dir()
            and all(
                (base / name / "features.csv").is_file()
                and (base / name / "extraction_times.csv").is_file()
                for base in features_dirs
            )
        ]
        return logics

    if args.logic is not None:
        divisions_to_run = [args.logic]
        if not (splits_base / args.logic).is_dir():
            parser.error(f"Splits directory not found for logic {args.logic!r}: {splits_base / args.logic}")
    else:
        divisions_to_run = discover_logics_from_feature_dir()
        if not divisions_to_run:
            parser.error(
                f"No logics found: need subdirs with features.csv and extraction_times.csv in "
                f"{features_dirs[0]!s} and matching splits in {splits_base!s}"
            )
        logging.info("Running all %d logics: %s", len(divisions_to_run), divisions_to_run)

    base_output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    for division in divisions_to_run:
        splits_dir = splits_base / division

        feature_csv_paths = []
        extraction_time_by_path = {}
        failed_paths_set = set()
        for base in features_dirs:
            division_dir = base / division
            feature_csv = division_dir / "features.csv"
            extraction_times_csv = division_dir / "extraction_times.csv"
            if not feature_csv.is_file():
                raise FileNotFoundError(f"Missing feature CSV for {division}: {feature_csv}")
            if not extraction_times_csv.is_file():
                raise FileNotFoundError(f"Missing extraction_times CSV for {division}: {extraction_times_csv}")
            feature_csv_paths.append(str(feature_csv))
            times = load_extraction_times_csv(extraction_times_csv)
            for p, t in times.items():
                extraction_time_by_path[p] = extraction_time_by_path.get(p, 0.0) + min(
                    t, FEATURE_TIMEOUT
                )
            failed_paths_set.update(
                load_failed_paths_from_extraction_times_csv(extraction_times_csv)
            )
        feature_csv_path = feature_csv_paths[0] if len(feature_csv_paths) == 1 else feature_csv_paths
        output_dir = (base_output_dir / division).resolve() if base_output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        evaluate_multi_splits(
            splits_dir,
            feature_csv_path,
            extraction_time_by_path,
            failed_paths_from_csv=failed_paths_set,
            xg_flag=args.xg,
            save_models=args.save_models,
            output_dir=output_dir,
            timeout=args.timeout,
            svm_c=args.svm_c,
            random_seed=args.random_seed,
        )


if __name__ == "__main__":
    main()
