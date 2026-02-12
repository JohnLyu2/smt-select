#!/usr/bin/env python3
"""
Train an algorithm selection model on a training CSV and evaluate on training and test CSVs.
Target logic is specified via --logic argument.
Performance files are expected at:
  - data/perf_data/train/{logic}.CSV
  - data/perf_data/test/{logic}.CSV
"""

import argparse
import logging
import json
import tempfile
import shutil
import sys
from pathlib import Path
import numpy as np

from src.perf_parser import parse_performance_csv
from src.pwc import train_pwc, PwcSelector
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
        "--logic",
        type=str,
        required=True,
        help="Logic to process (e.g., BV, ABV, QF_LIA)",
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

    # Resolve performance CSV paths
    logic = args.logic
    train_csv = Path(f"data/perf_data/train/{logic}.CSV")
    if not train_csv.exists():
        train_csv = Path(f"data/perf_data/train/{logic}.csv")

    test_csv = Path(f"data/perf_data/test/{logic}.CSV")
    if not test_csv.exists():
        test_csv = Path(f"data/perf_data/test/{logic}.csv")

    if not train_csv.exists():
        logging.error(f"Training CSV not found: {train_csv}")
        sys.exit(1)

    if not test_csv.exists():
        logging.error(f"Test CSV not found: {test_csv}")
        sys.exit(1)

    # Load performance data
    logging.info(f"Loading training data from {train_csv}...")
    train_data = parse_performance_csv(str(train_csv), args.timeout)
    logging.info(f"Loading test data from {test_csv}...")
    test_data = parse_performance_csv(str(test_csv), args.timeout)

    all_instance_paths = set(train_data.keys()) | set(test_data.keys())

    # Validate feature coverage
    logging.info("Validating feature coverage...")
    missing_instances, _ = validate_feature_coverage(all_instance_paths, args.feature_csv)
    if missing_instances:
        logging.error(
            f"ERROR: {len(missing_instances)} instance(s) are missing features in provided feature CSV(s)."
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
    logging.info(f"Training {'XGBoost' if args.xg else 'SVM'} model...")
    train_pwc(
        train_data,
        save_dir=str(model_save_dir),
        xg_flag=args.xg,
        feature_csv_path=args.feature_csv,
        svm_c=args.svm_c,
        random_seed=args.random_seed,
    )

    # Load trained model
    model_path = model_save_dir / "model.joblib"
    as_model = PwcSelector.load(str(model_path))

    # Ensure feature_csv_path is set in model (for inference)
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

    # Summary results
    results = {
        "logic": logic,
        "model_type": "XGBoost" if args.xg else "SVM",
        "feature_csv": args.feature_csv,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"Algorithm Selection Results for {logic}")
    print("=" * 60)
    print(f"Model: {results['model_type']}")
    print(f"Train CSV: {results['train_csv']}")
    print(f"Test CSV:  {results['test_csv']}")
    
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
