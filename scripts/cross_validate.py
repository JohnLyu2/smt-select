#!/usr/bin/env python3
"""
Cross-validation script for algorithm selection models.

Performs k-fold cross-validation using pre-split fold files by:
1. Loading fold files from a specified directory (e.g., data/perf_data/folds/ABV)
2. For each fold, using the fold file as the test set
3. Training a model on all other instances (train set = all instances - test set)
4. Evaluating on the held-out fold
5. Aggregating results across all folds
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np

from src.performance import MultiSolverDataset
from src.performance import parse_performance_json
from src.pwc import train_pwc, PwcSelector
from src.evaluate import as_evaluate
from src.feature import validate_feature_coverage


def create_subset_dataset(
    full_dataset: MultiSolverDataset, instance_paths: list[str]
) -> MultiSolverDataset:
    """
    Create a subset of MultiSolverDataset containing only the specified instance paths.

    Args:
        full_dataset: The full MultiSolverDataset
        instance_paths: List of instance paths to include in the subset

    Returns:
        A new MultiSolverDataset containing only the specified instances
    """
    subset_perf_dict = {}
    for path in instance_paths:
        if path in full_dataset:
            subset_perf_dict[path] = full_dataset[path]

    return MultiSolverDataset(
        subset_perf_dict,
        full_dataset.get_solver_id_dict(),
        full_dataset.get_timeout(),
    )


def compute_metrics(result_dataset, multi_perf_data):
    """
    Compute evaluation metrics for a result dataset.

    Args:
        result_dataset: SingleSolverDataset with algorithm selection results
        multi_perf_data: MultiSolverDataset for comparison metrics

    Returns:
        Dictionary of metrics
    """
    total_count = len(result_dataset)
    solved_count = result_dataset.get_solved_count()
    solve_rate = (solved_count / total_count * 100) if total_count > 0 else 0.0

    # Calculate average PAR-2
    total_par2 = sum(result_dataset.get_par2(path) for path in result_dataset.keys())
    avg_par2 = total_par2 / total_count if total_count > 0 else 0.0

    # Get best single solver for comparison
    sbs_dataset = multi_perf_data.get_best_solver_dataset()
    sbs_solved = sbs_dataset.get_solved_count()
    sbs_solve_rate = (sbs_solved / total_count * 100) if total_count > 0 else 0.0
    total_par2_sbs = sum(sbs_dataset.get_par2(path) for path in sbs_dataset.keys())
    avg_par2_sbs = total_par2_sbs / total_count if total_count > 0 else 0.0

    # Get virtual best solver for comparison
    vbs_dataset = multi_perf_data.get_virtual_best_solver_dataset()
    vbs_solved = vbs_dataset.get_solved_count()
    vbs_solve_rate = (vbs_solved / total_count * 100) if total_count > 0 else 0.0
    total_par2_vbs = sum(vbs_dataset.get_par2(path) for path in vbs_dataset.keys())
    avg_par2_vbs = total_par2_vbs / total_count if total_count > 0 else 0.0

    # Assert that SBS and VBS are not the same
    assert vbs_solved != sbs_solved, (
        f"SBS and VBS have the same solved count: {sbs_solved}"
    )
    assert avg_par2_vbs != avg_par2_sbs, (
        f"SBS and VBS have the same avg_par2: {avg_par2_sbs}"
    )

    # Calculate gap_cls (closed gap) metrics: (as - sbs) / (vbs - sbs)
    # For solved: gap_cls_solved = (solved_count - sbs_solved) / (vbs_solved - sbs_solved)
    # For par2: gap_cls_par2 = (avg_par2 - avg_par2_sbs) / (avg_par2_vbs - avg_par2_sbs)
    gap_cls_solved = (solved_count - sbs_solved) / (vbs_solved - sbs_solved)
    gap_cls_par2 = (avg_par2 - avg_par2_sbs) / (avg_par2_vbs - avg_par2_sbs)

    return {
        "total_instances": total_count,
        "solved": solved_count,
        "solve_rate": solve_rate,
        "avg_par2": avg_par2,
        "sbs_name": sbs_dataset.get_solver_name(),
        "sbs_solved": sbs_solved,
        "sbs_solve_rate": sbs_solve_rate,
        "sbs_avg_par2": avg_par2_sbs,
        "vbs_solved": vbs_solved,
        "vbs_solve_rate": vbs_solve_rate,
        "vbs_avg_par2": avg_par2_vbs,
        "gap_cls_solved": gap_cls_solved,
        "gap_cls_par2": gap_cls_par2,
    }


def load_fold_test_instances(
    fold_file_path: Path, timeout: float
) -> MultiSolverDataset:
    """
    Load test instances from a fold JSON file.

    Args:
        fold_file_path: Path to the fold JSON file
        timeout: Timeout value in seconds

    Returns:
        MultiSolverDataset containing test instances from the fold
    """
    return parse_performance_json(str(fold_file_path), timeout)


def load_all_instances_from_folds(
    folds_dir: Path, timeout: float
) -> MultiSolverDataset:
    """
    Load all instances by combining all fold files.

    Args:
        folds_dir: Directory containing fold JSON files
        timeout: Timeout value in seconds

    Returns:
        MultiSolverDataset containing all instances from all folds
    """
    fold_files = sorted(folds_dir.glob("*.json"), key=lambda x: int(x.stem))

    # Combine all fold datasets
    all_perf_dict = {}
    solver_id_dict = None

    for fold_file in fold_files:
        fold_data = parse_performance_json(str(fold_file), timeout)
        all_perf_dict.update(fold_data)

        # Get solver info from first fold (should be same for all)
        if solver_id_dict is None:
            solver_id_dict = fold_data.get_solver_id_dict()

    return MultiSolverDataset(
        all_perf_dict,
        solver_id_dict,
        timeout,
    )


def cross_validate(
    folds_dir: Path,
    feature_csv_path: str | list[str],
    xg_flag: bool = False,
    save_models: bool = False,
    output_dir: Path = None,
    timeout: float = 1200.0,
    svm_c: float = 1.0,
    random_seed: int = 42,
):
    """
    Perform k-fold cross-validation on algorithm selection model using pre-split fold files.

    Args:
        feature_csv_path: Path to features CSV file, or list of paths to multiple CSV files
        folds_dir: Directory containing fold JSON files (e.g., data/perf_data/folds/ABV)
        xg_flag: Whether to use XGBoost (default: False, uses SVM)
        save_models: Whether to save models for each fold
        output_dir: Directory to save results and models (optional)
        timeout: Timeout value in seconds

    Returns:
        Dictionary containing per-fold and aggregated results
    """
    folds_dir = Path(folds_dir)
    if not folds_dir.exists():
        raise ValueError(f"Folds directory does not exist: {folds_dir}")

    # Load all instances by combining all fold files
    logging.info(f"Loading all instances from fold files in {folds_dir}...")
    multi_perf_data = load_all_instances_from_folds(folds_dir, timeout)
    all_instance_paths = set(multi_perf_data.keys())
    n_instances = len(all_instance_paths)

    # Log feature CSV path(s)
    if isinstance(feature_csv_path, list):
        logging.info(
            f"Using {len(feature_csv_path)} feature CSV file(s): {feature_csv_path}"
        )
    else:
        logging.info(f"Using feature CSV file: {feature_csv_path}")

    # Validate that all instances have features
    logging.info("Validating feature coverage for all instances...")
    missing_instances, instance_missing_in_csvs = validate_feature_coverage(
        all_instance_paths, feature_csv_path
    )

    if missing_instances:
        error_msg = f"ERROR: {len(missing_instances)} instance(s) are missing features in ALL feature CSV(s).\n"
        if isinstance(feature_csv_path, list):
            error_msg += f"  Feature CSVs: {feature_csv_path}\n"
        else:
            error_msg += f"  Feature CSV: {feature_csv_path}\n"
        error_msg += "  Missing instances (showing first 10):\n"
        for instance in missing_instances[:10]:
            error_msg += f"    - {instance}\n"
        if len(missing_instances) > 10:
            error_msg += f"    ... and {len(missing_instances) - 10} more\n"
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Check for instances missing in some (but not all) CSVs when using multiple CSVs
    if isinstance(feature_csv_path, list) and instance_missing_in_csvs:
        error_msg = (
            f"ERROR: {len(instance_missing_in_csvs)} instance(s) are missing in some feature CSV(s).\n"
            f"  All instances must be present in ALL {len(feature_csv_path)} feature CSV(s).\n"
            f"  Instances with missing features (showing first 10):\n"
        )
        for instance, missing_csvs in list(instance_missing_in_csvs.items())[:10]:
            error_msg += f"    - {instance}\n"
            error_msg += f"      Missing in: {missing_csvs}\n"
        if len(instance_missing_in_csvs) > 10:
            error_msg += f"    ... and {len(instance_missing_in_csvs) - 10} more\n"
        logging.error(error_msg)
        raise ValueError(error_msg)

    logging.info(
        f"Feature coverage validated: all {n_instances} instances have features"
    )

    # Find all fold files (e.g., 0.json, 1.json, 2.json, ...)
    fold_files = sorted(folds_dir.glob("*.json"), key=lambda x: int(x.stem))
    n_splits = len(fold_files)

    if n_splits == 0:
        raise ValueError(f"No fold JSON files found in {folds_dir}")

    logging.info(
        f"Starting {n_splits}-fold cross-validation on {n_instances} instances"
    )
    logging.info("Loading folds from %s", folds_dir)

    # Storage for per-fold results
    fold_results = []

    # Iterate over folds
    for fold_num, fold_file in enumerate(fold_files):
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Fold {fold_num + 1}/{n_splits} ({fold_file.name})")
        logging.info(f"{'=' * 60}")

        # Load test instances from fold file
        logging.info(f"Loading test instances from {fold_file.name}...")
        test_data = load_fold_test_instances(fold_file, timeout)
        test_paths = set(test_data.keys())

        logging.info(f"Test instances: {len(test_paths)}")

        # Train set = all instances NOT in test set
        train_paths = list(all_instance_paths - test_paths)
        logging.info(f"Train instances: {len(train_paths)}")

        # Create train/test datasets
        train_data = create_subset_dataset(multi_perf_data, train_paths)

        # Determine model save location
        if save_models and output_dir:
            model_save_dir = output_dir / f"fold_{fold_num}"
        else:
            # Use temporary directory (models won't be saved)
            import tempfile

            model_save_dir = Path(tempfile.mkdtemp())

        # Train model
        logging.info("Training model...")
        train_pwc(
            train_data,
            save_dir=str(model_save_dir),
            xg_flag=xg_flag,
            feature_csv_path=feature_csv_path,
            svm_c=svm_c,
            random_seed=random_seed,
        )

        # Load trained model
        model_path = model_save_dir / "model.joblib"
        as_model = PwcSelector.load(str(model_path))

        # Set feature CSV path if not already set
        if as_model.feature_csv_path is None:
            as_model.feature_csv_path = feature_csv_path

        # Create train and test output directories if output_dir is specified
        train_output_dir = None
        test_output_dir = None
        if output_dir:
            train_output_dir = output_dir / "train"
            test_output_dir = output_dir / "test"
            train_output_dir.mkdir(parents=True, exist_ok=True)
            test_output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate model on train set
        logging.info("Evaluating model on train set...")
        train_output_csv_path = None
        if train_output_dir:
            train_output_csv_path = train_output_dir / f"{fold_num}.csv"
        train_result_dataset = as_evaluate(
            as_model,
            train_data,
            write_csv_path=str(train_output_csv_path)
            if train_output_csv_path
            else None,
        )

        # Compute train metrics
        train_metrics = compute_metrics(train_result_dataset, train_data)

        # Evaluate model on test set
        logging.info("Evaluating model on test set...")
        test_output_csv_path = None
        if test_output_dir:
            test_output_csv_path = test_output_dir / f"{fold_num}.csv"
        test_result_dataset = as_evaluate(
            as_model,
            test_data,
            write_csv_path=str(test_output_csv_path) if test_output_csv_path else None,
        )

        # Compute test metrics
        test_metrics = compute_metrics(test_result_dataset, test_data)

        # Store fold results
        fold_result = {
            "fold": fold_num + 1,
            "train_size": len(train_paths),
            "test_size": len(test_paths),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        fold_results.append(fold_result)

        # Log fold results (train first, then test)
        logging.info(f"\nFold {fold_num + 1} Results:")
        logging.info("  Train Set:")
        logging.info(
            f"    Solved: {train_metrics['solved']}/{train_metrics['total_instances']}"
        )
        logging.info(f"    Solve rate: {train_metrics['solve_rate']:.2f}%")
        logging.info(f"    Average PAR-2: {train_metrics['avg_par2']:.2f}")
        logging.info("  Test Set:")
        logging.info(
            f"    Solved: {test_metrics['solved']}/{test_metrics['total_instances']}"
        )
        logging.info(f"    Solve rate: {test_metrics['solve_rate']:.2f}%")
        logging.info(f"    Average PAR-2: {test_metrics['avg_par2']:.2f}")

        # Clean up temporary model if not saving
        if not save_models:
            import shutil

            shutil.rmtree(model_save_dir, ignore_errors=True)

    # Aggregate results across folds (test metrics)
    test_metrics_list = [fr["test_metrics"] for fr in fold_results]
    train_metrics_list = [fr["train_metrics"] for fr in fold_results]

    aggregated = {
        "test_solve_rate_mean": np.mean([m["solve_rate"] for m in test_metrics_list]),
        "test_solve_rate_std": np.std([m["solve_rate"] for m in test_metrics_list]),
        "test_avg_par2_mean": np.mean([m["avg_par2"] for m in test_metrics_list]),
        "test_avg_par2_std": np.std([m["avg_par2"] for m in test_metrics_list]),
        "test_sbs_solve_rate_mean": np.mean(
            [m["sbs_solve_rate"] for m in test_metrics_list]
        ),
        "test_sbs_solve_rate_std": np.std(
            [m["sbs_solve_rate"] for m in test_metrics_list]
        ),
        "test_sbs_avg_par2_mean": np.mean(
            [m["sbs_avg_par2"] for m in test_metrics_list]
        ),
        "test_sbs_avg_par2_std": np.std([m["sbs_avg_par2"] for m in test_metrics_list]),
        "test_vbs_solve_rate_mean": np.mean(
            [m["vbs_solve_rate"] for m in test_metrics_list]
        ),
        "test_vbs_solve_rate_std": np.std(
            [m["vbs_solve_rate"] for m in test_metrics_list]
        ),
        "test_vbs_avg_par2_mean": np.mean(
            [m["vbs_avg_par2"] for m in test_metrics_list]
        ),
        "test_vbs_avg_par2_std": np.std([m["vbs_avg_par2"] for m in test_metrics_list]),
        "test_gap_cls_solved_mean": np.mean(
            [m["gap_cls_solved"] for m in test_metrics_list]
        ),
        "test_gap_cls_solved_std": np.std(
            [m["gap_cls_solved"] for m in test_metrics_list]
        ),
        "test_gap_cls_par2_mean": np.mean(
            [m["gap_cls_par2"] for m in test_metrics_list]
        ),
        "test_gap_cls_par2_std": np.std([m["gap_cls_par2"] for m in test_metrics_list]),
        "train_solve_rate_mean": np.mean([m["solve_rate"] for m in train_metrics_list]),
        "train_solve_rate_std": np.std([m["solve_rate"] for m in train_metrics_list]),
        "train_avg_par2_mean": np.mean([m["avg_par2"] for m in train_metrics_list]),
        "train_avg_par2_std": np.std([m["avg_par2"] for m in train_metrics_list]),
        "train_sbs_solve_rate_mean": np.mean(
            [m["sbs_solve_rate"] for m in train_metrics_list]
        ),
        "train_sbs_solve_rate_std": np.std(
            [m["sbs_solve_rate"] for m in train_metrics_list]
        ),
        "train_sbs_avg_par2_mean": np.mean(
            [m["sbs_avg_par2"] for m in train_metrics_list]
        ),
        "train_sbs_avg_par2_std": np.std(
            [m["sbs_avg_par2"] for m in train_metrics_list]
        ),
        "train_vbs_solve_rate_mean": np.mean(
            [m["vbs_solve_rate"] for m in train_metrics_list]
        ),
        "train_vbs_solve_rate_std": np.std(
            [m["vbs_solve_rate"] for m in train_metrics_list]
        ),
        "train_vbs_avg_par2_mean": np.mean(
            [m["vbs_avg_par2"] for m in train_metrics_list]
        ),
        "train_vbs_avg_par2_std": np.std(
            [m["vbs_avg_par2"] for m in train_metrics_list]
        ),
        "train_gap_cls_solved_mean": np.mean(
            [m["gap_cls_solved"] for m in train_metrics_list]
        ),
        "train_gap_cls_solved_std": np.std(
            [m["gap_cls_solved"] for m in train_metrics_list]
        ),
        "train_gap_cls_par2_mean": np.mean(
            [m["gap_cls_par2"] for m in train_metrics_list]
        ),
        "train_gap_cls_par2_std": np.std(
            [m["gap_cls_par2"] for m in train_metrics_list]
        ),
    }

    results = {
        "n_splits": n_splits,
        "n_instances": n_instances,
        "model_type": "XGBoost" if xg_flag else "SVM",
        "folds_dir": str(folds_dir),
        "feature_csv_path": feature_csv_path,
        "svm_c": svm_c,
        "folds": fold_results,
        "aggregated": aggregated,
    }

    # Save summary.json if output_dir is provided
    if output_dir:
        summary_path = output_dir / "summary.json"

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {
                    key: convert_to_json_serializable(value)
                    for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj

        json_results = convert_to_json_serializable(results)

        with open(summary_path, "w") as f:
            json.dump(json_results, f, indent=2)

        logging.info(f"Saved summary to {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Perform k-fold cross-validation on algorithm selection model"
    )
    parser.add_argument(
        "--folds-dir",
        type=str,
        required=True,
        help="Directory containing fold CSV files (e.g., data/perf_data/folds/ABV)",
    )
    parser.add_argument(
        "--feature-csv",
        type=str,
        required=True,
        nargs="+",
        help="Path(s) to the features CSV file(s). Can specify multiple files to concatenate features.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Timeout value in seconds (default: 1200.0)",
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
        help="Directory to save results and models (optional)",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save models for each fold (requires --output-dir)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--svm-c",
        type=float,
        default=1.0,
        help="Regularization parameter C for SVM (default: 1.0)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for solver selection tie-breaking (default: 42)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create output directory if specified
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.save_models:
            logging.info(f"Models will be saved to {output_dir}")
        else:
            logging.info(f"Results will be saved to {output_dir}")

    # Perform cross-validation
    results = cross_validate(
        folds_dir=Path(args.folds_dir),
        feature_csv_path=args.feature_csv,
        xg_flag=args.xg,
        save_models=args.save_models,
        output_dir=output_dir,
        timeout=args.timeout,
        svm_c=args.svm_c,
        random_seed=args.random_seed,
    )

    # Print aggregated results
    logging.info("\n" + "=" * 60)
    logging.info("Cross-Validation Results Summary")
    logging.info("=" * 60)
    logging.info(f"Model type: {results['model_type']}")
    logging.info(f"Number of folds: {results['n_splits']}")
    logging.info(f"Total instances: {results['n_instances']}")
    logging.info("")
    agg = results["aggregated"]

    logging.info("Train Set Performance:")
    logging.info("  Algorithm Selection:")
    logging.info(
        f"    Solve rate: {agg['train_solve_rate_mean']:.2f}% ± {agg['train_solve_rate_std']:.2f}%"
    )
    logging.info(
        f"    Average PAR-2: {agg['train_avg_par2_mean']:.2f} ± {agg['train_avg_par2_std']:.2f}"
    )
    logging.info(
        f"    Gap closed (solved): {agg['train_gap_cls_solved_mean']:.4f} ± {agg['train_gap_cls_solved_std']:.4f}"
    )
    logging.info(
        f"    Gap closed (PAR-2): {agg['train_gap_cls_par2_mean']:.4f} ± {agg['train_gap_cls_par2_std']:.4f}"
    )
    logging.info("  SBS:")
    logging.info(
        f"    Solve rate: {agg['train_sbs_solve_rate_mean']:.2f}% ± {agg['train_sbs_solve_rate_std']:.2f}%"
    )
    logging.info(
        f"    Average PAR-2: {agg['train_sbs_avg_par2_mean']:.2f} ± {agg['train_sbs_avg_par2_std']:.2f}"
    )
    logging.info("  VBS:")
    logging.info(
        f"    Solve rate: {agg['train_vbs_solve_rate_mean']:.2f}% ± {agg['train_vbs_solve_rate_std']:.2f}%"
    )
    logging.info(
        f"    Average PAR-2: {agg['train_vbs_avg_par2_mean']:.2f} ± {agg['train_vbs_avg_par2_std']:.2f}"
    )
    logging.info("")
    logging.info("Test Set Performance:")
    logging.info("  Algorithm Selection:")
    logging.info(
        f"    Solve rate: {agg['test_solve_rate_mean']:.2f}% ± {agg['test_solve_rate_std']:.2f}%"
    )
    logging.info(
        f"    Average PAR-2: {agg['test_avg_par2_mean']:.2f} ± {agg['test_avg_par2_std']:.2f}"
    )
    logging.info(
        f"    Gap closed (solved): {agg['test_gap_cls_solved_mean']:.4f} ± {agg['test_gap_cls_solved_std']:.4f}"
    )
    logging.info(
        f"    Gap closed (PAR-2): {agg['test_gap_cls_par2_mean']:.4f} ± {agg['test_gap_cls_par2_std']:.4f}"
    )
    logging.info("  SBS:")
    logging.info(
        f"    Solve rate: {agg['test_sbs_solve_rate_mean']:.2f}% ± {agg['test_sbs_solve_rate_std']:.2f}%"
    )
    logging.info(
        f"    Average PAR-2: {agg['test_sbs_avg_par2_mean']:.2f} ± {agg['test_sbs_avg_par2_std']:.2f}"
    )
    logging.info("  VBS:")
    logging.info(
        f"    Solve rate: {agg['test_vbs_solve_rate_mean']:.2f}% ± {agg['test_vbs_solve_rate_std']:.2f}%"
    )
    logging.info(
        f"    Average PAR-2: {agg['test_vbs_avg_par2_mean']:.2f} ± {agg['test_vbs_avg_par2_std']:.2f}"
    )
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
