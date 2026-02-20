#!/usr/bin/env python3
"""
Run cross-validation experiments for all logics.

For each logic found in the folds directory:
1. Finds the corresponding feature CSV in the features directory
2. Runs cross-validation using the cross_validate function
3. Saves results to the results directory under {LOGIC}

Command-line arguments:
    --folds-dir: Base directory containing fold JSON files (default: data/perf_data/folds)
    --features-dir: Directory containing feature CSV files (required)
    --results-dir: Base directory to save results (required)
"""

import argparse
import logging
import sys
from pathlib import Path

from scripts.cross_validate import cross_validate

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
            # Check if it contains JSON files (fold files)
            json_files = list(logic_dir.glob("*.json"))
            if json_files:
                logics.append(logic_dir.name)

    return logics


def run_cv_for_logic(
    logic: str, folds_dir: Path, features_dirs: list[Path], results_dir: Path
):
    """
    Run cross-validation for a single logic.

    Args:
        logic: Logic name (e.g., "ABV", "QF_LIA")
        folds_dir: Directory containing fold CSV files for this logic
        features_dirs: List of directories containing feature CSV files
        results_dir: Directory to save results
    """
    # Construct paths
    logic_folds_dir = folds_dir / logic

    # Find feature CSV files in all feature directories
    feature_csvs = []
    missing_dirs = []
    for features_dir in features_dirs:
        # Try both .CSV and .csv extensions
        feature_csv_upper = features_dir / f"{logic}.CSV"
        feature_csv_lower = features_dir / f"{logic}.csv"

        if feature_csv_upper.exists():
            feature_csvs.append(feature_csv_upper)
        elif feature_csv_lower.exists():
            feature_csvs.append(feature_csv_lower)
        else:
            missing_dirs.append(features_dir)

    # If multiple feature directories are provided, ALL must contain the feature CSV
    if len(features_dirs) > 1:
        if missing_dirs:
            error_msg = f"ERROR: When multiple feature directories are provided, ALL must contain feature CSV for {logic}.\n"
            error_msg += f"  Missing in {len(missing_dirs)} directory(ies):\n"
            for features_dir in missing_dirs:
                error_msg += (
                    f"    - {features_dir} (expected: {logic}.CSV or {logic}.csv)\n"
                )
            error_msg += f"  Found in {len(feature_csvs)} directory(ies):\n"
            for csv_path in feature_csvs:
                error_msg += f"    - {csv_path.parent}\n"
            print(error_msg)
            raise FileNotFoundError(error_msg)

    # Check if we found at least one feature CSV (for single directory case)
    if not feature_csvs:
        error_msg = f"ERROR: No feature CSV found for {logic} in feature directory.\n"
        error_msg += f"  Searched in: {features_dirs[0]}\n"
        error_msg += f"  Expected: {logic}.CSV or {logic}.csv\n"
        print(error_msg)
        raise FileNotFoundError(error_msg)

    # Check if folds directory exists and has JSON files
    if not logic_folds_dir.exists():
        print(
            f"WARNING: Missing folds directory for {logic}: {logic_folds_dir} -> skipping"
        )
        return False

    json_files = list(logic_folds_dir.glob("*.json"))
    if not json_files:
        print(f"WARNING: No fold JSON files found in {logic_folds_dir} -> skipping")
        return False

    # Create results directory
    logic_results_dir = results_dir / logic
    logic_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running CV for logic: {logic}")
    print(f"{'=' * 60}")
    print(f"  Folds dir:    {logic_folds_dir}")
    if len(feature_csvs) == 1:
        print(f"  Feature CSV:  {feature_csvs[0]}")
    else:
        print(f"  Feature CSVs ({len(feature_csvs)}):")
        for csv_path in feature_csvs:
            print(f"    - {csv_path}")
    print(f"  Results dir:  {logic_results_dir}")
    print(f"  Folds found:  {len(json_files)}")

    try:
        # Convert Path objects to strings for cross_validate
        feature_csv_paths = [str(csv_path) for csv_path in feature_csvs]

        # Run cross-validation
        # If only one CSV, pass as string for backward compatibility
        # If multiple, pass as list
        feature_csv_arg = (
            feature_csv_paths[0] if len(feature_csv_paths) == 1 else feature_csv_paths
        )

        _ = cross_validate(
            folds_dir=logic_folds_dir,
            feature_csv_path=feature_csv_arg,
            xg_flag=False,
            save_models=False,
            output_dir=logic_results_dir,
            timeout=1200.0,
        )
        print(f"Completed CV for {logic}")
        return True
    except (FileNotFoundError, ValueError) as e:
        # These are validation errors that should stop execution
        print(f"ERROR: Validation failed for {logic}: {e}")
        logging.exception("Validation failed")
        raise  # Re-raise to stop execution
    except Exception as e:
        # Other errors (e.g., training failures) can be logged but allow continuation
        print(f"ERROR: Failed running CV for {logic}: {e}")
        logging.exception("Cross-validation failed")
        return False


def main():
    """Run cross-validation for all logics."""
    parser = argparse.ArgumentParser(
        description="Run cross-validation experiments for all logics"
    )
    parser.add_argument(
        "--folds-dir",
        type=str,
        default=str(DEFAULT_FOLDS_BASE_DIR),
        help=f"Base directory containing fold CSV files (default: {DEFAULT_FOLDS_BASE_DIR})",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        nargs="+",
        help="Directory(ies) containing feature CSV files. Can specify multiple directories to concatenate features.",
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
    features_dirs = [Path(d) for d in args.features_dir]
    results_base_dir = Path(args.results_dir)

    # Check base directories exist
    if not folds_base_dir.exists():
        print(f"Error: Folds base directory does not exist: {folds_base_dir}")
        sys.exit(1)

    # Check all feature directories exist
    for features_dir in features_dirs:
        if not features_dir.exists():
            print(f"Error: Features directory does not exist: {features_dir}")
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
    print("\nStarting cross-validation experiments...")
    print(f"Folds directory:  {folds_base_dir}")
    if len(features_dirs) == 1:
        print(f"Features directory: {features_dirs[0]}")
    else:
        print(f"Features directories ({len(features_dirs)}):")
        for features_dir in features_dirs:
            print(f"  - {features_dir}")
    print(f"Results directory: {results_base_dir}")

    # Create results base directory
    results_base_dir.mkdir(parents=True, exist_ok=True)

    # Run CV for each logic
    successful = 0
    failed = 0

    for logic in logics:
        success = run_cv_for_logic(
            logic,
            folds_base_dir,
            features_dirs,
            results_base_dir,
        )
        if success:
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("Cross-Validation Summary")
    print(f"{'=' * 60}")
    print(f"Total logics:     {len(logics)}")
    print(f"Successful:       {successful}")
    print(f"Failed:           {failed}")
    print(f"Results saved to: {results_base_dir}")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
