#!/usr/bin/env python3
"""
Run fixed-alpha decision-level fusion experiments for all logics.

For each logic found in the folds directory:
1. Finds the corresponding syntactic and description feature CSVs
2. Runs cross-validation with fixed-alpha fusion (α tuned via nested CV)
3. Saves results to the specified directory

Command-line arguments:
    --folds-dir: Base directory containing fold CSV files (default: data/perf_data/folds)
    --features-synt: Directory containing syntactic feature CSV files (required)
    --features-desc: Directory containing description feature CSV files (required)
    --alpha-candidates: Alpha values to try (default: 0.0 to 1.0)
    --results-dir: Base directory to save results (required)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from scripts.cross_validate_fusion import cross_validate_fusion

# Default directories
DEFAULT_FOLDS_BASE_DIR = Path("data/perf_data/folds")


def discover_logics(folds_dir: Path) -> list[str]:
    """
    Discover all logics that have fold directories.

    Args:
        folds_dir: Base directory containing logic folders

    Returns:
        List of logic names (directory names)
    """
    logics = []
    if not folds_dir.exists():
        print(f"Error: Folds directory does not exist: {folds_dir}")
        return logics

    for logic_dir in sorted(folds_dir.iterdir()):
        if logic_dir.is_dir():
            # Check if it contains CSV files (fold files)
            csv_files = list(logic_dir.glob("*.csv"))
            if csv_files:
                logics.append(logic_dir.name)

    return logics


def run_cv_for_logic(
    logic: str,
    folds_dir: Path,
    features_dir_synt: Path,
    features_dir_desc: Path,
    alpha_candidates: list[float],
    results_dir: Path,
):
    """
    Run fusion cross-validation for a single logic.

    Args:
        logic: Logic name (e.g., "ABV", "QF_LIA")
        folds_dir: Directory containing fold CSV files for this logic
        features_dir_synt: Directory containing syntactic feature CSV files
        features_dir_desc: Directory containing description feature CSV files
        alpha_candidates: List of alpha values to try
        results_dir: Directory to save results

    Returns:
        Results dict, or None if failed
    """
    # Construct paths
    logic_folds_dir = folds_dir / logic

    # Find syntactic feature CSV
    feature_csv_synt_upper = features_dir_synt / f"{logic}.CSV"
    feature_csv_synt_lower = features_dir_synt / f"{logic}.csv"
    if feature_csv_synt_upper.exists():
        feature_csv_synt = feature_csv_synt_upper
    elif feature_csv_synt_lower.exists():
        feature_csv_synt = feature_csv_synt_lower
    else:
        error_msg = f"ERROR: Syntactic feature CSV not found for {logic}\n"
        error_msg += f"  Searched in: {features_dir_synt}\n"
        error_msg += f"  Expected: {logic}.CSV or {logic}.csv\n"
        print(error_msg)
        raise FileNotFoundError(error_msg)

    # Find description feature CSV
    feature_csv_desc_upper = features_dir_desc / f"{logic}.CSV"
    feature_csv_desc_lower = features_dir_desc / f"{logic}.csv"
    if feature_csv_desc_upper.exists():
        feature_csv_desc = feature_csv_desc_upper
    elif feature_csv_desc_lower.exists():
        feature_csv_desc = feature_csv_desc_lower
    else:
        error_msg = f"ERROR: Description feature CSV not found for {logic}\n"
        error_msg += f"  Searched in: {features_dir_desc}\n"
        error_msg += f"  Expected: {logic}.CSV or {logic}.csv\n"
        print(error_msg)
        raise FileNotFoundError(error_msg)

    # Check if folds directory exists and has CSV files
    if not logic_folds_dir.exists():
        print(
            f"WARNING: Missing folds directory for {logic}: {logic_folds_dir} -> skipping"
        )
        return None

    csv_files = list(logic_folds_dir.glob("*.csv"))
    if not csv_files:
        print(f"WARNING: No fold CSV files found in {logic_folds_dir} -> skipping")
        return None

    print(f"\n{'=' * 70}")
    print(f"Running Fixed-Alpha Fusion for logic: {logic}")
    print(f"{'=' * 70}")
    print(f"  Folds dir:      {logic_folds_dir}")
    print(f"  Feature (synt): {feature_csv_synt}")
    print(f"  Feature (desc): {feature_csv_desc}")
    print(f"  Folds found:    {len(csv_files)}")
    print(f"  Alpha range:    {min(alpha_candidates):.1f} - {max(alpha_candidates):.1f}")

    # Create results directory for this logic
    logic_results_dir = results_dir / logic
    logic_results_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = cross_validate_fusion(
            folds_dir=logic_folds_dir,
            feature_csv_synt=str(feature_csv_synt),
            feature_csv_desc=str(feature_csv_desc),
            alpha_candidates=alpha_candidates,
            xg_flag=False,
            save_models=False,
            output_dir=logic_results_dir,
            timeout=1200.0,
        )
        print(f"  Completed fusion for {logic}")
        print(
            f"  Test gap_cls_par2: {results['aggregated']['test_gap_cls_par2_mean']:.4f} "
            f"± {results['aggregated']['test_gap_cls_par2_std']:.4f}"
        )
        print(
            f"  Best alpha (mean): {results['aggregated']['alpha_mean']:.2f} "
            f"± {results['aggregated']['alpha_std']:.2f}"
        )
        return results
    except (FileNotFoundError, ValueError) as e:
        print(f"  ERROR: Validation failed for {logic}: {e}")
        logging.exception("Validation failed")
        raise
    except Exception as e:
        print(f"  ERROR: Failed for {logic}: {e}")
        logging.exception("Cross-validation failed")
        return None


def generate_summary(all_results: dict, results_dir: Path):
    """
    Generate a summary across all logics.

    Args:
        all_results: Dict[logic -> results]
        results_dir: Directory to save summary
    """
    # Prepare summary data
    summary_data = []

    for logic, results in all_results.items():
        if results is None:
            continue

        agg = results["aggregated"]
        row = {
            "logic": logic,
            "gap_cls_par2_mean": agg["test_gap_cls_par2_mean"],
            "gap_cls_par2_std": agg["test_gap_cls_par2_std"],
            "solve_rate_mean": agg["test_solve_rate_mean"],
            "avg_par2_mean": agg["test_avg_par2_mean"],
            "alpha_mean": agg["alpha_mean"],
            "alpha_std": agg["alpha_std"],
        }
        summary_data.append(row)

    # Save summary JSON
    summary_path = results_dir / "all_logics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Fixed-Alpha Fusion Results - Test Gap Closed (PAR-2)")
    print("=" * 80)
    print(f"{'Logic':<12} {'Gap Closed':>16} {'Solve Rate':>14} {'Best Alpha':>14}")
    print("-" * 80)

    total_gap = 0
    count = 0
    for row in summary_data:
        logic = row["logic"]
        gap = row["gap_cls_par2_mean"]
        gap_std = row["gap_cls_par2_std"]
        solve = row["solve_rate_mean"]
        alpha = row["alpha_mean"]
        alpha_std = row["alpha_std"]

        print(
            f"{logic:<12} {gap:>7.2f} ± {gap_std:<6.2f} {solve:>13.1f}% {alpha:>7.2f} ± {alpha_std:<4.2f}"
        )
        total_gap += gap
        count += 1

    print("-" * 80)
    if count > 0:
        print(f"{'AVERAGE':<12} {total_gap/count:>16.2f}")
    print("=" * 80)

    return summary_data


def main():
    """Run fusion cross-validation for all logics."""
    parser = argparse.ArgumentParser(
        description="Run fixed-alpha fusion experiments for all logics"
    )
    parser.add_argument(
        "--folds-dir",
        type=str,
        default=str(DEFAULT_FOLDS_BASE_DIR),
        help=f"Base directory containing fold CSV files (default: {DEFAULT_FOLDS_BASE_DIR})",
    )
    parser.add_argument(
        "--features-synt",
        type=str,
        required=True,
        help="Directory containing syntactic feature CSV files",
    )
    parser.add_argument(
        "--features-desc",
        type=str,
        required=True,
        help="Directory containing description feature CSV files",
    )
    parser.add_argument(
        "--alpha-candidates",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Alpha values to try (default: 0.0 to 1.0 in steps of 0.1)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Base directory to save results",
    )

    args = parser.parse_args()

    # Convert to Path objects
    folds_base_dir = Path(args.folds_dir)
    features_dir_synt = Path(args.features_synt)
    features_dir_desc = Path(args.features_desc)
    results_base_dir = Path(args.results_dir)

    # Check base directories exist
    if not folds_base_dir.exists():
        print(f"Error: Folds base directory does not exist: {folds_base_dir}")
        sys.exit(1)

    if not features_dir_synt.exists():
        print(f"Error: Syntactic features directory does not exist: {features_dir_synt}")
        sys.exit(1)

    if not features_dir_desc.exists():
        print(
            f"Error: Description features directory does not exist: {features_dir_desc}"
        )
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Discover logics
    logics = discover_logics(folds_base_dir)

    if not logics:
        print("No logics found with fold files. Exiting.")
        sys.exit(1)

    print(f"Found {len(logics)} logics: {', '.join(logics)}")
    print("\nStarting fixed-alpha fusion experiments...")
    print(f"Folds directory:            {folds_base_dir}")
    print(f"Syntactic features:         {features_dir_synt}")
    print(f"Description features:       {features_dir_desc}")
    print(f"Alpha candidates:           {args.alpha_candidates}")
    print(f"Results directory:          {results_base_dir}")

    # Create results base directory
    results_base_dir.mkdir(parents=True, exist_ok=True)

    # Run CV for each logic
    all_results = {}
    successful = 0
    failed = 0

    for logic in logics:
        try:
            results = run_cv_for_logic(
                logic,
                folds_base_dir,
                features_dir_synt,
                features_dir_desc,
                args.alpha_candidates,
                results_base_dir,
            )
            all_results[logic] = results
            if results is not None:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"ERROR: Failed for {logic}: {e}")
            all_results[logic] = None
            failed += 1

    # Generate summary
    if all_results:
        generate_summary(all_results, results_base_dir)

    # Final summary
    print(f"\n{'=' * 70}")
    print("Fusion Cross-Validation Summary")
    print(f"{'=' * 70}")
    print(f"Total logics:     {len(logics)}")
    print(f"Successful:       {successful}")
    print(f"Failed:           {failed}")
    print(f"Results saved to: {results_base_dir}")
    print(f"{'=' * 70}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
