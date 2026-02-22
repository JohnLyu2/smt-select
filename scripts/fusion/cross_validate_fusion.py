#!/usr/bin/env python3
"""
Cross-validation script for fixed-alpha decision-level fusion with automatic alpha tuning.

For each outer fold:
1. Train synt and desc models ONCE on the training set
2. Tune alpha using VALIDATION set (oracle mode for upper bound analysis)
3. Evaluate on test set with best alpha using those same models

Oracle mode: Uses test/validation set for alpha selection to find the upper bound
of what fusion could achieve with perfect alpha selection.
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import tempfile
import shutil

from src.pwc import train_pwc, PwcSelector
from src.pwc_fusion import PwcModelFusion
from src.evaluate import as_evaluate
from src.feature import validate_feature_coverage

# Import helper functions from cross_validate.py
from scripts.cross_validate import (
    create_subset_dataset,
    compute_metrics,
    load_fold_test_instances,
    load_all_instances_from_folds,
)


def tune_alpha_on_validation(
    model_synt,
    model_desc,
    val_data,
    feature_csv_synt,
    feature_csv_desc,
    alpha_candidates,
):
    """
    Tune alpha using validation set (ORACLE mode - gives upper bound).

    This uses the held-out validation/test set to select alpha, which provides
    an upper bound on what fusion could achieve with perfect alpha selection.
    NOT a realistic training scenario, but useful for analysis.

    Args:
        model_synt: Trained syntactic model
        model_desc: Trained description model
        val_data: Validation dataset (held-out fold)
        feature_csv_synt: Path to syntactic features CSV
        feature_csv_desc: Path to description features CSV
        alpha_candidates: List of alpha values to try

    Returns:
        Tuple of (best_alpha, tuning_results_dict)
    """
    logging.info(f"\n[ORACLE MODE] Tuning alpha on VALIDATION set ({len(val_data)} instances)")
    logging.info("  (This gives upper bound - not realistic for deployment)")

    # Evaluate each alpha on validation set
    alpha_val_perfs = {}
    alpha_details = {}

    for alpha in alpha_candidates:
        fusion_model = PwcModelFusion(
            model_synt,
            model_desc,
            alpha,
            feature_csv_synt,
            feature_csv_desc,
        )
        val_result = as_evaluate(fusion_model, val_data)
        val_metrics = compute_metrics(val_result, val_data)

        alpha_val_perfs[alpha] = val_metrics["gap_cls_par2"]
        alpha_details[alpha] = {
            "gap_cls_par2": val_metrics["gap_cls_par2"],
            "gap_cls_solved": val_metrics["gap_cls_solved"],
            "solve_rate": val_metrics["solve_rate"],
            "avg_par2": val_metrics["avg_par2"],
        }

    # Find best alpha
    best_alpha = max(alpha_val_perfs, key=alpha_val_perfs.get)
    best_perf = alpha_val_perfs[best_alpha]

    # Detailed logging of alpha tuning results
    logging.info(f"\n  Alpha tuning results (on validation set):")
    logging.info(f"  {'Alpha':>6} | {'Gap PAR2':>10} | {'Gap Solved':>10} | {'Solve%':>8} | {'Avg PAR2':>10}")
    logging.info(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*10}")

    for alpha in sorted(alpha_val_perfs.keys()):
        details = alpha_details[alpha]
        marker = " ← BEST" if alpha == best_alpha else ""
        logging.info(f"  {alpha:>6.2f} | {details['gap_cls_par2']:>10.4f} | {details['gap_cls_solved']:>10.4f} | "
                    f"{details['solve_rate']:>7.2f}% | {details['avg_par2']:>10.2f}{marker}")

    logging.info(f"\n  [ORACLE] Selected α={best_alpha:.2f} (gap_cls_par2={best_perf:.4f})")

    # Log interpretation
    if best_alpha == 0.0:
        logging.info("  → Best alpha=0.0 means description model alone is best")
    elif best_alpha == 1.0:
        logging.info("  → Best alpha=1.0 means syntactic model alone is best")
    else:
        logging.info(f"  → Best alpha={best_alpha:.2f} means fusion helps "
                    f"({best_alpha*100:.0f}% syntactic, {(1-best_alpha)*100:.0f}% description)")

    return best_alpha, alpha_val_perfs


def cross_validate_fusion(
    folds_dir: Path,
    feature_csv_synt: str,
    feature_csv_desc: str,
    alpha_candidates=None,
    xg_flag: bool = False,
    save_models: bool = False,
    output_dir: Path = None,
    timeout: float = 1200.0,
):
    """
    Perform k-fold cross-validation with fixed-alpha fusion.

    For each outer fold:
    1. Train synt and desc models on the training set
    2. Evaluate ALL alpha candidates on the test set

    After all folds:
    3. For each alpha, compute mean gap_cls_solved across all folds
    4. Select the alpha with the best mean gap_cls_solved

    Args:
        folds_dir: Directory containing fold CSV files (e.g., data/perf_data/folds/ABV)
        feature_csv_synt: Path to syntactic features CSV
        feature_csv_desc: Path to description features CSV
        alpha_candidates: List of alpha values to try (default: [0.0, 0.1, ..., 1.0])
        xg_flag: Whether to use XGBoost (default: False, uses SVM)
        save_models: Whether to save models for each fold
        output_dir: Directory to save results and models (optional)
        timeout: Timeout value in seconds

    Returns:
        Dictionary containing per-fold and aggregated results
    """
    if alpha_candidates is None:
        alpha_candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    folds_dir = Path(folds_dir)
    if not folds_dir.exists():
        raise ValueError(f"Folds directory does not exist: {folds_dir}")

    # Load all instances
    logging.info(f"Loading all instances from fold files in {folds_dir}...")
    multi_perf_data = load_all_instances_from_folds(folds_dir, timeout)
    all_instance_paths = set(multi_perf_data.keys())
    n_instances = len(all_instance_paths)

    logging.info(f"Using syntactic features: {feature_csv_synt}")
    logging.info(f"Using description features: {feature_csv_desc}")

    # Validate feature coverage for both CSVs
    logging.info("Validating feature coverage for syntactic features...")
    missing_synt, _ = validate_feature_coverage(all_instance_paths, feature_csv_synt)
    if missing_synt:
        raise ValueError(
            f"ERROR: {len(missing_synt)} instances missing in syntactic features CSV"
        )

    logging.info("Validating feature coverage for description features...")
    missing_desc, _ = validate_feature_coverage(all_instance_paths, feature_csv_desc)
    if missing_desc:
        raise ValueError(
            f"ERROR: {len(missing_desc)} instances missing in description features CSV"
        )

    logging.info(
        f"Feature coverage validated: all {n_instances} instances have features"
    )

    # Find all fold files
    fold_files = sorted(folds_dir.glob("*.csv"), key=lambda x: int(x.stem))
    n_splits = len(fold_files)

    if n_splits == 0:
        raise ValueError(f"No fold CSV files found in {folds_dir}")

    logging.info(
        f"Starting {n_splits}-fold cross-validation with fixed-alpha fusion on {n_instances} instances"
    )
    logging.info(f"Alpha candidates: {alpha_candidates}")

    # Storage for per-fold, per-alpha results
    # Structure: {alpha: [metrics_fold0, metrics_fold1, ...]}
    alpha_fold_metrics = {alpha: [] for alpha in alpha_candidates}
    fold_info = []  # Store fold metadata (train_size, test_size, etc.)

    # Iterate over outer folds
    for fold_num, fold_file in enumerate(fold_files):
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Outer Fold {fold_num + 1}/{n_splits} ({fold_file.name})")
        logging.info(f"{'=' * 60}")

        # Load test instances
        logging.info(f"Loading test instances from {fold_file.name}...")
        test_data = load_fold_test_instances(fold_file, timeout)
        test_paths = set(test_data.keys())
        logging.info(f"Test instances: {len(test_paths)}")

        # Train set = all instances NOT in test set
        train_paths = list(all_instance_paths - test_paths)
        logging.info(f"Train instances: {len(train_paths)}")
        train_data = create_subset_dataset(multi_perf_data, train_paths)

        # Store fold info
        fold_info.append({
            "fold": fold_num + 1,
            "train_size": len(train_paths),
            "test_size": len(test_paths),
        })

        # Determine model save location
        if save_models and output_dir:
            model_save_dir = output_dir / f"fold_{fold_num}"
        else:
            model_save_dir = Path(tempfile.mkdtemp())

        # Step 1: Train both models on full training set
        logging.info("Training syntactic model on training set...")
        synt_dir = model_save_dir / "synt"
        train_pwc(
            train_data,
            save_dir=str(synt_dir),
            xg_flag=xg_flag,
            feature_csv_path=feature_csv_synt,
        )
        model_synt = PwcSelector.load(str(synt_dir / "model.joblib"))
        if model_synt.feature_csv_path is None:
            model_synt.feature_csv_path = feature_csv_synt

        logging.info("Training description model on training set...")
        desc_dir = model_save_dir / "desc"
        train_pwc(
            train_data,
            save_dir=str(desc_dir),
            xg_flag=xg_flag,
            feature_csv_path=feature_csv_desc,
        )
        model_desc = PwcSelector.load(str(desc_dir / "model.joblib"))
        if model_desc.feature_csv_path is None:
            model_desc.feature_csv_path = feature_csv_desc

        # Step 2: Evaluate ALL alpha candidates on test set
        logging.info(f"\nEvaluating all alpha candidates on test set...")
        for alpha in alpha_candidates:
            fusion_model = PwcModelFusion(
                model_synt,
                model_desc,
                alpha,
                feature_csv_synt,
                feature_csv_desc,
            )
            test_result = as_evaluate(fusion_model, test_data)
            test_metrics = compute_metrics(test_result, test_data)
            alpha_fold_metrics[alpha].append(test_metrics)

        # Log results for this fold
        logging.info(f"\nFold {fold_num + 1} Results (all alphas):")
        logging.info(f"  {'Alpha':>6} | {'Gap Solved':>10} | {'Gap PAR2':>10} | {'Solve%':>8}")
        logging.info(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*8}")
        for alpha in alpha_candidates:
            m = alpha_fold_metrics[alpha][fold_num]
            logging.info(f"  {alpha:>6.2f} | {m['gap_cls_solved']:>10.4f} | {m['gap_cls_par2']:>10.4f} | {m['solve_rate']:>7.2f}%")

        # Clean up temporary model directory if not saving
        if not save_models:
            shutil.rmtree(model_save_dir, ignore_errors=True)

    # Step 3: Compute mean gap_cls_par2 for each alpha across all folds
    logging.info(f"\n{'=' * 60}")
    logging.info("Alpha Selection: Mean gap_cls_par2 across all folds")
    logging.info(f"{'=' * 60}")

    alpha_mean_gap_cls_par2 = {}
    for alpha in alpha_candidates:
        mean_gap = np.mean([m["gap_cls_par2"] for m in alpha_fold_metrics[alpha]])
        alpha_mean_gap_cls_par2[alpha] = mean_gap

    # Find best alpha (highest gap_cls_par2 is best)
    best_alpha = max(alpha_mean_gap_cls_par2, key=alpha_mean_gap_cls_par2.get)

    logging.info(f"\n  {'Alpha':>6} | {'Mean Gap PAR2':>15}")
    logging.info(f"  {'-'*6} | {'-'*15}")
    for alpha in alpha_candidates:
        marker = " ← BEST" if alpha == best_alpha else ""
        logging.info(f"  {alpha:>6.2f} | {alpha_mean_gap_cls_par2[alpha]:>15.4f}{marker}")

    logging.info(f"\nSelected α={best_alpha:.2f} (mean gap_cls_par2={alpha_mean_gap_cls_par2[best_alpha]:.4f})")

    # Use the best alpha's metrics for final results
    test_metrics_list = alpha_fold_metrics[best_alpha]

    # For train metrics, we need to re-evaluate with best alpha (or skip if not needed)
    # For simplicity, we'll compute aggregated stats from test metrics only
    # Train metrics would require re-running evaluation, which is expensive

    aggregated = {
        "best_alpha": best_alpha,
        "alpha_mean_gap_cls_par2": alpha_mean_gap_cls_par2,
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
    }

    # Build fold_results for output compatibility
    fold_results = []
    for fold_num in range(n_splits):
        fold_result = {
            "fold": fold_num + 1,
            "train_size": fold_info[fold_num]["train_size"],
            "test_size": fold_info[fold_num]["test_size"],
            "alpha_metrics": {alpha: alpha_fold_metrics[alpha][fold_num] for alpha in alpha_candidates},
            "test_metrics": alpha_fold_metrics[best_alpha][fold_num],
        }
        fold_results.append(fold_result)

    results = {
        "n_splits": n_splits,
        "n_instances": n_instances,
        "fusion_method": "fixed_alpha_cv",
        "model_type": f"Fusion_{'XGBoost' if xg_flag else 'SVM'}_fixed_alpha_cv",
        "best_alpha": best_alpha,
        "alpha_selection_note": "Alpha selected by best mean gap_cls_par2 across all folds",
        "folds_dir": str(folds_dir),
        "feature_csv_synt": feature_csv_synt,
        "feature_csv_desc": feature_csv_desc,
        "alpha_candidates": alpha_candidates,
        "folds": fold_results,
        "aggregated": aggregated,
    }

    # Save summary.json if output_dir is provided
    if output_dir:
        summary_path = output_dir / "summary.json"

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
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
        description="Perform k-fold CV with decision-level fusion and automatic alpha tuning"
    )
    parser.add_argument(
        "--folds-dir",
        type=str,
        required=True,
        help="Directory containing fold CSV files (e.g., data/perf_data/folds/ABV)",
    )
    parser.add_argument(
        "--feature-csv-synt",
        type=str,
        required=True,
        help="Path to syntactic features CSV",
    )
    parser.add_argument(
        "--feature-csv-desc",
        type=str,
        required=True,
        help="Path to description features CSV",
    )
    parser.add_argument(
        "--alpha-candidates",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Alpha values to try (default: 0.0 to 1.0 in steps of 0.1)",
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

    # Perform cross-validation with fusion
    results = cross_validate_fusion(
        folds_dir=Path(args.folds_dir),
        feature_csv_synt=args.feature_csv_synt,
        feature_csv_desc=args.feature_csv_desc,
        alpha_candidates=args.alpha_candidates,
        xg_flag=args.xg,
        save_models=args.save_models,
        output_dir=output_dir,
        timeout=args.timeout,
    )

    # Print aggregated results
    logging.info("\n" + "=" * 60)
    logging.info("Cross-Validation Results Summary")
    logging.info("=" * 60)
    logging.info(f"Model type: {results['model_type']}")
    logging.info(f"Number of folds: {results['n_splits']}")
    logging.info(f"Total instances: {results['n_instances']}")
    logging.info(f"Best alpha: {results['best_alpha']:.2f}")
    logging.info(f"Alpha selection: {results['alpha_selection_note']}")
    logging.info("")

    agg = results["aggregated"]

    logging.info("Test Set Performance:")
    logging.info("  Algorithm Selection (Fusion):")
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
    logging.info("  SBS (Single Best Solver):")
    logging.info(
        f"    Solve rate: {agg['test_sbs_solve_rate_mean']:.2f}% ± {agg['test_sbs_solve_rate_std']:.2f}%"
    )
    logging.info(
        f"    Average PAR-2: {agg['test_sbs_avg_par2_mean']:.2f} ± {agg['test_sbs_avg_par2_std']:.2f}"
    )
    logging.info("  VBS (Virtual Best Solver):")
    logging.info(
        f"    Solve rate: {agg['test_vbs_solve_rate_mean']:.2f}% ± {agg['test_vbs_solve_rate_std']:.2f}%"
    )
    logging.info(
        f"    Average PAR-2: {agg['test_vbs_avg_par2_mean']:.2f} ± {agg['test_vbs_avg_par2_std']:.2f}"
    )
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
