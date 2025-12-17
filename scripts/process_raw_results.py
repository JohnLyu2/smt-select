#!/usr/bin/env python3
"""
Process raw results from algorithm selector evaluations.

For each logic in data/folks26/selector_res/test, this script:
1. Loads the corresponding raw performance data (all solvers)
2. Loads each algorithm selector's results (machsmt, syn, des, des_exp, des_syn)
3. Computes SBS (Single Best Solver) and VBS (Virtual Best Solver)
4. Creates a table with each logic as a row and selectors as columns
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import parse_performance_csv, parse_as_perf_csv
from src.performance import MultiSolverDataset, SingleSolverDataset


def process_logic(
    logic: str,
    selector_res_dir: Path,
    raw_perf_dir: Path,
    timeout: float = 1200.0,
) -> Dict[str, int]:
    """
    Process a single logic and return solved counts for all required selectors.

    Args:
        logic: Logic name (e.g., "abv", "qf_slia")
        selector_res_dir: Base directory for selector results
        raw_perf_dir: Base directory for raw performance data
        timeout: Timeout value in seconds

    Returns:
        Dictionary mapping selector names to solved counts

    Raises:
        FileNotFoundError: If any required file is missing
        ValueError: If SBS or VBS cannot be computed
    """
    results: Dict[str, int] = {}

    # Paths
    logic_test_dir = selector_res_dir / logic / "test"
    raw_perf_csv = raw_perf_dir / f"{logic}_test.csv"

    # Check if directories/files exist - report errors
    if not logic_test_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {logic_test_dir}")

    if not raw_perf_csv.exists():
        raise FileNotFoundError(f"Raw performance file does not exist: {raw_perf_csv}")

    # Required selector files
    required_selectors = ["syn", "des", "des_exp", "des_syn"]
    for selector_name in required_selectors:
        selector_csv = logic_test_dir / f"{selector_name}.csv"
        if not selector_csv.exists():
            raise FileNotFoundError(f"Selector file does not exist: {selector_csv}")

    # Optional selector: machsmt (warn if missing)
    machsmt_csv = logic_test_dir / "machsmt.csv"
    if not machsmt_csv.exists():
        print(
            f"Warning: machsmt.csv does not exist for {logic}: {machsmt_csv}",
            file=sys.stderr,
        )

    # Load raw performance data (all solvers)
    print(f"Loading raw performance data: {raw_perf_csv}", file=sys.stderr)
    multi_perf = parse_performance_csv(str(raw_perf_csv), timeout)

    # Get SBS
    try:
        sbs_dataset = multi_perf.get_best_solver_dataset()
        results["sbs"] = sbs_dataset.get_solved_count()
    except ValueError as e:
        raise ValueError(f"Could not compute SBS for {logic}: {e}")

    # Get VBS
    try:
        vbs_dataset = multi_perf.get_virtual_best_solver_dataset()
        results["vbs"] = vbs_dataset.get_solved_count()
    except ValueError as e:
        raise ValueError(f"Could not compute VBS for {logic}: {e}")

    # Load each required algorithm selector's results
    for selector_name in required_selectors:
        selector_csv = logic_test_dir / f"{selector_name}.csv"
        print(f"Loading selector: {selector_csv.name}", file=sys.stderr)
        try:
            selector_dataset = parse_as_perf_csv(str(selector_csv), timeout)
            results[selector_name] = selector_dataset.get_solved_count()
        except Exception as e:
            raise RuntimeError(f"Could not load {selector_csv}: {e}")

    # Load machsmt if it exists
    if machsmt_csv.exists():
        print(f"Loading selector: {machsmt_csv.name}", file=sys.stderr)
        try:
            machsmt_dataset = parse_as_perf_csv(str(machsmt_csv), timeout)
            results["machsmt"] = machsmt_dataset.get_solved_count()
        except Exception as e:
            print(
                f"Warning: Could not load {machsmt_csv}: {e}",
                file=sys.stderr,
            )

    return results


def write_results_csv(output_path: Path, logic_results: Dict[str, Dict[str, int]]):
    """Write results to a CSV file with logic as rows and selectors as columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Column order: logic, machsmt, syn, des, des_exp, des_syn, sbs, vbs
    columns = ["logic", "machsmt", "syn", "des", "des_exp", "des_syn", "sbs", "vbs"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for logic in sorted(logic_results.keys()):
            row = [logic]
            for col in columns[1:]:  # Skip 'logic' column
                value = logic_results[logic].get(col, "")
                row.append(value if value != "" else "")
            writer.writerow(row)

    print(f"\nResults saved to: {output_path}", file=sys.stderr)


def main():
    """Main function to process all logics and create results table."""
    parser = argparse.ArgumentParser(
        description="Process raw results from algorithm selector evaluations"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV file path for storing results (default: no file output)",
    )

    args = parser.parse_args()

    # Base paths
    base_dir = Path(__file__).parent.parent
    selector_res_dir = base_dir / "data" / "folks26" / "selector_res"
    raw_perf_dir = base_dir / "data" / "folks26" / "raw_performance"

    # Default timeout (20 minutes)
    timeout = 1200.0

    # Discover all logics
    if not selector_res_dir.exists():
        print(f"Error: {selector_res_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    logics = sorted(
        [
            d.name
            for d in selector_res_dir.iterdir()
            if d.is_dir() and (d / "test").exists()
        ]
    )

    if not logics:
        print(f"Error: No logics found in {selector_res_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(logics)} logics: {', '.join(logics)}", file=sys.stderr)
    print(file=sys.stderr)

    # Process each logic
    logic_results: Dict[str, Dict[str, int]] = {}
    errors = []

    for logic in logics:
        print(f"Processing logic: {logic}", file=sys.stderr)
        try:
            results = process_logic(logic, selector_res_dir, raw_perf_dir, timeout)
            logic_results[logic] = results
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            error_msg = f"Error processing {logic}: {e}"
            print(error_msg, file=sys.stderr)
            errors.append(error_msg)
        print(file=sys.stderr)

    # Report errors and exit if any occurred
    if errors:
        print("\nErrors encountered:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        sys.exit(1)

    # Print results table
    print("=" * 110)
    print(
        f"{'Logic':12s} | {'machsmt':>8s} | {'syn':>8s} | {'des':>8s} | {'des_exp':>8s} | {'des_syn':>8s} | {'sbs':>8s} | {'vbs':>8s}"
    )
    print("=" * 110)

    for logic in sorted(logic_results.keys()):
        results = logic_results[logic]
        machsmt = results.get("machsmt", "")
        syn = results.get("syn", "")
        des = results.get("des", "")
        des_exp = results.get("des_exp", "")
        des_syn = results.get("des_syn", "")
        sbs = results.get("sbs", "")
        vbs = results.get("vbs", "")
        print(
            f"{logic:12s} | {machsmt:>8} | {syn:>8} | {des:>8} | {des_exp:>8} | {des_syn:>8} | {sbs:>8} | {vbs:>8}"
        )

    print("=" * 110)

    # Write results to CSV if output path is specified
    if args.output:
        output_path = Path(args.output)
        write_results_csv(output_path, logic_results)


if __name__ == "__main__":
    main()
