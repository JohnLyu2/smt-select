#!/usr/bin/env python3
"""
Train an algorithm selection model and evaluate on train and test sets.
Specify --train-file and --test-file (JSON).
"""

import argparse
import logging
import json
import tempfile
import shutil
import sys
from pathlib import Path
import numpy as np

from src.performance import parse_performance_json
from src.pwc import train_pwc, PwcSelector
from src.pwc_wl import train_pwc_wl, PwcWlSelector
from src.evaluate import as_evaluate
from src.feature import validate_feature_coverage


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

    # Calculate average PAR-2
    total_par2 = sum(result_dataset.get_par2(path) for path in result_dataset.keys())
    avg_par2 = total_par2 / total_count if total_count > 0 else 0.0

    # Get best single solver for comparison
    sbs_dataset = multi_perf_data.get_best_solver_dataset()
    sbs_solved = sbs_dataset.get_solved_count()
    total_par2_sbs = sum(sbs_dataset.get_par2(path) for path in sbs_dataset.keys())
    avg_par2_sbs = total_par2_sbs / total_count if total_count > 0 else 0.0

    # Get virtual best solver for comparison
    vbs_dataset = multi_perf_data.get_virtual_best_solver_dataset()
    vbs_solved = vbs_dataset.get_solved_count()
    total_par2_vbs = sum(vbs_dataset.get_par2(path) for path in vbs_dataset.keys())
    avg_par2_vbs = total_par2_vbs / total_count if total_count > 0 else 0.0

    # Calculate gap_cls (closed gap) metrics: (as - sbs) / (vbs - sbs)
    # Handle cases where SBS and VBS are identical to avoid division by zero
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
        "total_instances": total_count,
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


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate algorithm selection model on pre-split train/test data"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to training performance JSON file.",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test performance JSON file.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["pwc", "wl"],
        default="pwc",
        help="Model type: pwc (feature-based) or wl (graph-kernel). Default: pwc.",
    )
    parser.add_argument(
        "--feature-csv",
        type=str,
        default=None,
        nargs="+",
        help="Path(s) to the features CSV file(s). Required for --model-type pwc.",
    )
    parser.add_argument(
        "--wl-iter",
        type=int,
        default=2,
        help="Weisfeiler-Lehman iteration count for --model-type wl (default: 2)",
    )
    parser.add_argument(
        "--graph-timeout",
        type=int,
        default=10,
        help="Graph build timeout in seconds for --model-type wl (default: 10)",
    )
    parser.add_argument(
        "--benchmark-root",
        type=str,
        default=None,
        help="Root directory for instance paths; required for --model-type wl when paths are relative (e.g. ABV/...).",
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
        "--output-json",
        type=str,
        default=None,
        help="Path to save summary results JSON (optional)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory to save the trained model (optional)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained model (requires --model-dir)",
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
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    train_path = Path(args.train_file)
    test_path = Path(args.test_file)
    if not train_path.exists():
        logging.error(f"Training file not found: {train_path}")
        sys.exit(1)
    if not test_path.exists():
        logging.error(f"Test file not found: {test_path}")
        sys.exit(1)

    logging.info(f"Loading training data from {train_path}...")
    train_data = parse_performance_json(str(train_path), args.timeout)
    logging.info(f"Loading test data from {test_path}...")
    test_data = parse_performance_json(str(test_path), args.timeout)

    use_wl = args.model_type == "wl"
    if use_wl:
        if args.feature_csv is not None:
            logging.warning("--feature-csv is ignored for --model-type wl")
        if args.benchmark_root:
            root = Path(args.benchmark_root).resolve()
            if not root.is_dir():
                logging.error(f"--benchmark-root is not a directory: {root}")
                sys.exit(1)
            train_dict = {str(root / p): train_data[p] for p in train_data.keys()}
            test_dict = {str(root / p): test_data[p] for p in test_data.keys()}
            train_data = type(train_data)(
                train_dict,
                train_data.get_solver_id_dict(),
                train_data.get_timeout(),
            )
            test_data = type(test_data)(
                test_dict,
                test_data.get_solver_id_dict(),
                test_data.get_timeout(),
            )
            logging.info(f"Instance paths rebased under benchmark root: {root}")
    else:
        if args.feature_csv is None:
            logging.error("--feature-csv is required for --model-type pwc")
            sys.exit(1)
        all_instance_paths = set(train_data.keys()) | set(test_data.keys())
        logging.info("Validating feature coverage...")
        missing_instances, _ = validate_feature_coverage(
            all_instance_paths, args.feature_csv
        )
        if missing_instances:
            logging.error(
                f"ERROR: {len(missing_instances)} instance(s) are missing features."
            )
            sys.exit(1)

    # Setup save directory
    temp_dir = None
    if args.save_model and args.model_dir:
        model_save_dir = Path(args.model_dir)
        model_save_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp()
        model_save_dir = Path(temp_dir)

    # Train model
    if use_wl:
        logging.info(
            f"Training WL model (wl_iter={args.wl_iter}, graph_timeout={args.graph_timeout}s)..."
        )
        train_pwc_wl(
            train_data,
            wl_iter=args.wl_iter,
            save_dir=str(model_save_dir),
            graph_timeout=args.graph_timeout,
        )
        as_model = PwcWlSelector.load(str(model_save_dir / "model.joblib"))
    else:
        logging.info(f"Training {'XGBoost' if args.xg else 'SVM'} model...")
        train_pwc(
            train_data,
            save_dir=str(model_save_dir),
            xg_flag=args.xg,
            feature_csv_path=args.feature_csv,
            svm_c=args.svm_c,
            random_seed=args.random_seed,
        )
        model_path = model_save_dir / "model.joblib"
        as_model = PwcSelector.load(str(model_path))
        if as_model.feature_csv_path is None:
            as_model.feature_csv_path = args.feature_csv

    # Evaluate on Train Set
    logging.info("Evaluating on training set...")
    train_result_dataset = as_evaluate(as_model, train_data)
    train_metrics = compute_metrics(train_result_dataset, train_data)

    # Evaluate on Test Set
    logging.info("Evaluating on test set...")
    test_result_dataset = as_evaluate(as_model, test_data)
    test_metrics = compute_metrics(test_result_dataset, test_data)

    model_type_label = "WL" if use_wl else ("XGBoost" if args.xg else "SVM")
    results = {
        "model_type": model_type_label,
        "feature_csv": args.feature_csv,
        "train_file": str(train_path),
        "test_file": str(test_path),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("Algorithm Selection Results")
    print("=" * 60)
    print(f"Model: {results['model_type']}")
    print(f"Train file: {results['train_file']}")
    print(f"Test file:  {results['test_file']}")
    
    for name, metrics in [("Train", train_metrics), ("Test", test_metrics)]:
        print(f"\n{name} Set Performance:")
        print(f"  Total instances: {metrics['total_instances']}")
        print(f"  Solved:          {metrics['solved']}/{metrics['total_instances']}")
        print(f"  Average PAR-2:   {metrics['avg_par2']:.4f}")
        print(f"  Gap Closed (Solved): {metrics['gap_cls_solved']:.4f}")
        print(f"  Gap Closed (PAR-2):  {metrics['gap_cls_par2']:.4f}")
        print(f"  SBS (Solver: {metrics['sbs_name']}) Solved: {metrics['sbs_solved']}, PAR-2: {metrics['sbs_avg_par2']:.4f}")
        print(f"  VBS Solved:      {metrics['vbs_solved']}, PAR-2: {metrics['vbs_avg_par2']:.4f}")
    
    print("\n" + "=" * 60)

    # Save to file if output_json specified
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return round(float(obj), 4)
            elif isinstance(obj, float):
                return round(obj, 4)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(i) for i in obj]
            else:
                return obj

        with open(output_path, "w") as f:
            json.dump(convert_to_json_serializable(results), f, indent=2)
        logging.info(f"Saved summary results to {output_path}")

    # Cleanup temp directory
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
