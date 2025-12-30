#!/usr/bin/env python3
"""
Run cross-validation experiments for all logics.

For each logic in data/perf_data/folds:
1. Finds the corresponding feature CSV in data/features/syntactic/catalog_all
2. Runs cross-validation using scripts/cross_validate.py
3. Saves results to data/cv_results/{LOGIC}
"""

import subprocess
import sys
from pathlib import Path

# Base directories
FOLDS_BASE_DIR = Path("data/perf_data/folds")
FEATURES_DIR = Path("data/features/syntactic/catalog_all")
RESULTS_BASE_DIR = Path("data/cv_results")
SCRIPT_PATH = Path("scripts/cross_validate.py")


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
    logic: str, folds_dir: Path, features_dir: Path, results_dir: Path
):
    """
    Run cross-validation for a single logic.

    Args:
        logic: Logic name (e.g., "ABV", "QF_LIA")
        folds_dir: Directory containing fold CSV files for this logic
        features_dir: Directory containing feature CSV files
        results_dir: Directory to save results
    """
    # Construct paths
    logic_folds_dir = folds_dir / logic
    feature_csv = features_dir / f"{logic}.CSV"

    # Check if feature CSV exists
    if not feature_csv.exists():
        print(f"WARNING: Missing feature CSV for {logic}: {feature_csv} -> skipping")
        return False

    # Check if folds directory exists and has CSV files
    if not logic_folds_dir.exists():
        print(
            f"WARNING: Missing folds directory for {logic}: {logic_folds_dir} -> skipping"
        )
        return False

    csv_files = list(logic_folds_dir.glob("*.csv"))
    if not csv_files:
        print(f"WARNING: No fold CSV files found in {logic_folds_dir} -> skipping")
        return False

    # Create results directory
    logic_results_dir = results_dir / logic
    logic_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running CV for logic: {logic}")
    print(f"{'=' * 60}")
    print(f"  Folds dir:    {logic_folds_dir}")
    print(f"  Feature CSV:  {feature_csv}")
    print(f"  Results dir:  {logic_results_dir}")
    print(f"  Folds found:  {len(csv_files)}")

    # Build command
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--folds-dir",
        str(logic_folds_dir),
        "--feature-csv",
        str(feature_csv),
        "--output-dir",
        str(logic_results_dir),
    ]

    try:
        # Run cross-validation
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"Completed CV for {logic}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed running CV for {logic}: {e}")
        return False


def main():
    """Run cross-validation for all logics."""
    # Check base directories exist
    if not FOLDS_BASE_DIR.exists():
        print(f"Error: Folds base directory does not exist: {FOLDS_BASE_DIR}")
        sys.exit(1)

    if not FEATURES_DIR.exists():
        print(f"Error: Features directory does not exist: {FEATURES_DIR}")
        sys.exit(1)

    if not SCRIPT_PATH.exists():
        print(f"Error: Cross-validation script does not exist: {SCRIPT_PATH}")
        sys.exit(1)

    # Discover logics
    logics = discover_logics(FOLDS_BASE_DIR)

    if not logics:
        print("No logics found with fold files. Exiting.")
        sys.exit(1)

    print(f"Found {len(logics)} logics: {', '.join(logics)}")
    print("\nStarting cross-validation experiments...")

    # Create results base directory
    RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Run CV for each logic
    successful = 0
    failed = 0

    for logic in logics:
        success = run_cv_for_logic(
            logic,
            FOLDS_BASE_DIR,
            FEATURES_DIR,
            RESULTS_BASE_DIR,
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
    print(f"Results saved to: {RESULTS_BASE_DIR}")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
