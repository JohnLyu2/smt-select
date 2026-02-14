#!/usr/bin/env python3
"""
Train syntactic and description models, fuse them with a fixed alpha,
and evaluate on training and test CSVs.

Performance files are expected at:
  - data/perf_data/train/{logic}.CSV
  - data/perf_data/test/{logic}.CSV
"""

import argparse
import json
import logging
import tempfile
import shutil
import sys
from pathlib import Path
import numpy as np

from src.parser import parse_performance_csv
from src.pwc import train_pwc, PwcSelector
from src.pwc_fusion import PwcModelFusion
from src.evaluate import as_evaluate
from src.feature import validate_feature_coverage
from ..train_evaluate import compute_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate fused model (syntactic + description) with fixed alpha"
    )
    parser.add_argument(
        "--logic",
        type=str,
        required=True,
        help="Logic to process (e.g., BV, ABV, QF_LIA)",
    )
    parser.add_argument(
        "--feature-csv-synt",
        type=str,
        required=True,
        help="Path to syntactic features CSV file",
    )
    parser.add_argument(
        "--feature-csv-desc",
        type=str,
        required=True,
        help="Path to description features CSV file",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Fusion alpha (0.0=desc only, 1.0=synt only)",
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
        help="Directory to save the trained model(s) (optional)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained model(s) (requires --model-dir)",
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

    # Validate feature coverage for both feature CSVs
    logging.info("Validating syntactic feature coverage...")
    missing_synt, _ = validate_feature_coverage(all_instance_paths, args.feature_csv_synt)
    if missing_synt:
        logging.error(
            f"ERROR: {len(missing_synt)} instance(s) missing in syntactic features CSV."
        )
        sys.exit(1)

    logging.info("Validating description feature coverage...")
    missing_desc, _ = validate_feature_coverage(all_instance_paths, args.feature_csv_desc)
    if missing_desc:
        logging.error(
            f"ERROR: {len(missing_desc)} instance(s) missing in description features CSV."
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

    # Train syntactic model
    logging.info(f"Training syntactic {'XGBoost' if args.xg else 'SVM'} model...")
    synt_dir = model_save_dir / "synt"
    train_pwc(
        train_data,
        save_dir=str(synt_dir),
        xg_flag=args.xg,
        feature_csv_path=args.feature_csv_synt,
        svm_c=args.svm_c,
        random_seed=args.random_seed,
    )

    # Train description model
    logging.info(f"Training description {'XGBoost' if args.xg else 'SVM'} model...")
    desc_dir = model_save_dir / "desc"
    train_pwc(
        train_data,
        save_dir=str(desc_dir),
        xg_flag=args.xg,
        feature_csv_path=args.feature_csv_desc,
        svm_c=args.svm_c,
        random_seed=args.random_seed,
    )

    # Load trained models
    model_synt = PwcSelector.load(str(synt_dir / "model.joblib"))
    if model_synt.feature_csv_path is None:
        model_synt.feature_csv_path = args.feature_csv_synt

    model_desc = PwcSelector.load(str(desc_dir / "model.joblib"))
    if model_desc.feature_csv_path is None:
        model_desc.feature_csv_path = args.feature_csv_desc

    # Build fused model
    fusion_model = PwcModelFusion(
        model_synt,
        model_desc,
        args.alpha,
        args.feature_csv_synt,
        args.feature_csv_desc,
    )

    # Evaluate on Train Set
    logging.info("Evaluating fused model on training set...")
    train_result_dataset = as_evaluate(fusion_model, train_data)
    train_metrics = compute_metrics(train_result_dataset, train_data)

    # Evaluate on Test Set
    logging.info("Evaluating fused model on test set...")
    test_result_dataset = as_evaluate(fusion_model, test_data)
    test_metrics = compute_metrics(test_result_dataset, test_data)

    # Summary results
    results = {
        "logic": logic,
        "model_type": f"Fusion_{'XGBoost' if args.xg else 'SVM'}",
        "alpha": args.alpha,
        "feature_csv_synt": args.feature_csv_synt,
        "feature_csv_desc": args.feature_csv_desc,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"Algorithm Selection Fusion Results for {logic}")
    print("=" * 60)
    print(f"Model: {results['model_type']}")
    print(f"Alpha: {results['alpha']}")
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
