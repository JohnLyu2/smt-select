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


def compute_avg_par2(dataset) -> float:
    """Compute average PAR2 for a dataset."""
    total_par2 = sum(dataset.get_par2(path) for path in dataset.keys())
    count = len(list(dataset.keys()))
    return total_par2 / count if count > 0 else 0.0


def process_logic(
    logic: str,
    selector_res_dir: Path,
    raw_perf_dir: Path,
    timeout: float = 1200.0,
    use_par2: bool = False,
) -> Dict[str, float]:
    """
    Process a single logic and return solved counts or PAR2 for all required selectors.

    Args:
        logic: Logic name (e.g., "abv", "qf_slia")
        selector_res_dir: Base directory for selector results
        raw_perf_dir: Base directory for raw performance data
        timeout: Timeout value in seconds
        use_par2: If True, return PAR2 values instead of solved counts

    Returns:
        Dictionary mapping selector names to solved counts or PAR2 values

    Raises:
        FileNotFoundError: If any required file is missing
        ValueError: If SBS or VBS cannot be computed
    """
    results: Dict[str, float] = {}

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
        if use_par2:
            results["sbs"] = compute_avg_par2(sbs_dataset)
        else:
            results["sbs"] = sbs_dataset.get_solved_count()
    except ValueError as e:
        raise ValueError(f"Could not compute SBS for {logic}: {e}")

    # Get VBS
    try:
        vbs_dataset = multi_perf.get_virtual_best_solver_dataset()
        if use_par2:
            results["vbs"] = compute_avg_par2(vbs_dataset)
        else:
            results["vbs"] = vbs_dataset.get_solved_count()
    except ValueError as e:
        raise ValueError(f"Could not compute VBS for {logic}: {e}")

    # Load each required algorithm selector's results
    for selector_name in required_selectors:
        selector_csv = logic_test_dir / f"{selector_name}.csv"
        print(f"Loading selector: {selector_csv.name}", file=sys.stderr)
        try:
            selector_dataset = parse_as_perf_csv(str(selector_csv), timeout)
            if use_par2:
                results[selector_name] = compute_avg_par2(selector_dataset)
            else:
                results[selector_name] = selector_dataset.get_solved_count()
        except Exception as e:
            raise RuntimeError(f"Could not load {selector_csv}: {e}")

    # Load machsmt if it exists
    if machsmt_csv.exists():
        print(f"Loading selector: {machsmt_csv.name}", file=sys.stderr)
        try:
            machsmt_dataset = parse_as_perf_csv(str(machsmt_csv), timeout)
            if use_par2:
                results["machsmt"] = compute_avg_par2(machsmt_dataset)
            else:
                results["machsmt"] = machsmt_dataset.get_solved_count()
        except Exception as e:
            print(
                f"Warning: Could not load {machsmt_csv}: {e}",
                file=sys.stderr,
            )

    return results


def compute_gap_closed(
    as_value: float, sbs_value: float, vbs_value: float, use_par2: bool = False
) -> Optional[float]:
    """
    Compute the percentage of gap closed by an algorithm selector.

    For #solved (higher is better): (as - sbs) / (vbs - sbs)
    For PAR2 (lower is better): (sbs - as) / (sbs - vbs)

    Args:
        as_value: Value for algorithm selector (#solved or PAR2)
        sbs_value: Value for SBS (#solved or PAR2)
        vbs_value: Value for VBS (#solved or PAR2)
        use_par2: If True, use PAR2 formula (lower is better)

    Returns:
        Gap closed percentage (0-100), or None if gap is zero
    """
    if use_par2:
        # For PAR2: lower is better, so gap = sbs - vbs
        gap = sbs_value - vbs_value
        if gap == 0:
            return None  # No gap to close
        return ((sbs_value - as_value) / gap) * 100
    else:
        # For #solved: higher is better, so gap = vbs - sbs
        gap = vbs_value - sbs_value
        if gap == 0:
            return None  # No gap to close
        return ((as_value - sbs_value) / gap) * 100


def write_results_csv(
    output_path: Path,
    logic_results: Dict[str, Dict[str, float]],
    gap_cls: bool = False,
    use_par2: bool = False,
):
    """
    Write results to a CSV file with logic as rows and selectors as columns.

    Args:
        output_path: Path to output CSV file
        logic_results: Dictionary mapping logic names to their results
        gap_cls: If True, output gap-closed percentages instead of raw values for selectors
        use_par2: If True, values are PAR2 scores instead of solved counts
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Column order: logic, machsmt, syn, des, des_exp, des_syn, sbs, vbs
    # When gap_cls is True, exclude sbs and vbs columns
    if gap_cls:
        columns = ["logic", "machsmt", "syn", "des", "des_exp", "des_syn"]
    else:
        columns = ["logic", "machsmt", "syn", "des", "des_exp", "des_syn", "sbs", "vbs"]
    # Algorithm selector columns (exclude sbs and vbs)
    selector_columns = ["machsmt", "syn", "des", "des_exp", "des_syn"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for logic in sorted(logic_results.keys()):
            row = [logic]
            results = logic_results[logic]
            sbs = results.get("sbs", "")
            vbs = results.get("vbs", "")

            for col in columns[1:]:  # Skip 'logic' column
                if gap_cls and col in selector_columns:
                    # Compute gap-closed percentage for algorithm selectors
                    as_value = results.get(col, "")
                    if as_value == "" or sbs == "" or vbs == "":
                        row.append("")
                    else:
                        gap_pct = compute_gap_closed(as_value, sbs, vbs, use_par2)
                        if gap_pct is None:
                            row.append("")  # No gap to close
                        else:
                            row.append(f"{gap_pct:.1f}")
                else:
                    # Output raw values for sbs, vbs, or when gap_cls is False
                    value = results.get(col, "")
                    if value == "":
                        row.append("")
                    elif use_par2:
                        row.append(f"{value:.1f}")
                    else:
                        row.append(value)
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
        required=True,
        help="Output CSV file path for storing results",
    )
    parser.add_argument(
        "--gap-cls",
        action="store_true",
        help="Output gap-closed percentages instead of raw values for algorithm selectors",
    )
    parser.add_argument(
        "--par2",
        action="store_true",
        help="Output PAR2 scores instead of solved counts",
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
            results = process_logic(
                logic, selector_res_dir, raw_perf_dir, timeout, use_par2=args.par2
            )
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
    metric_label = "PAR2" if args.par2 else "#solved"
    if args.gap_cls:
        print("=" * 80)
        print(
            f"{'Logic':12s} | {'machsmt':>8s} | {'syn':>8s} | {'des':>8s} | {'des_exp':>8s} | {'des_syn':>8s}"
        )
        print(f"  (gap-closed % based on {metric_label})" + " " * 40)
        print("=" * 80)
    else:
        print("=" * 110)
        print(
            f"{'Logic':12s} | {'machsmt':>8s} | {'syn':>8s} | {'des':>8s} | {'des_exp':>8s} | {'des_syn':>8s} | {'sbs':>8s} | {'vbs':>8s}"
        )
        if args.par2:
            print(f"  ({metric_label})" + " " * 95)
        print("=" * 110)

    for logic in sorted(logic_results.keys()):
        results = logic_results[logic]
        sbs = results.get("sbs", "")
        vbs = results.get("vbs", "")

        if args.gap_cls:
            # Compute gap-closed percentages for selectors
            machsmt_val = results.get("machsmt", "")
            syn_val = results.get("syn", "")
            des_val = results.get("des", "")
            des_exp_val = results.get("des_exp", "")
            des_syn_val = results.get("des_syn", "")

            def format_gap(val):
                if val == "" or sbs == "" or vbs == "":
                    return ""
                gap_pct = compute_gap_closed(val, sbs, vbs, args.par2)
                if gap_pct is None:
                    return ""
                return f"{gap_pct:.1f}%"

            machsmt = format_gap(machsmt_val)
            syn = format_gap(syn_val)
            des = format_gap(des_val)
            des_exp = format_gap(des_exp_val)
            des_syn = format_gap(des_syn_val)

            print(
                f"{logic:12s} | {machsmt:>8} | {syn:>8} | {des:>8} | {des_exp:>8} | {des_syn:>8}"
            )
        else:
            machsmt = results.get("machsmt", "")
            syn = results.get("syn", "")
            des = results.get("des", "")
            des_exp = results.get("des_exp", "")
            des_syn = results.get("des_syn", "")

            if args.par2:
                # Format PAR2 values with one decimal place
                def fmt(v):
                    return f"{v:.1f}" if v != "" else ""

                print(
                    f"{logic:12s} | {fmt(machsmt):>8} | {fmt(syn):>8} | {fmt(des):>8} | {fmt(des_exp):>8} | {fmt(des_syn):>8} | {fmt(sbs):>8} | {fmt(vbs):>8}"
                )
            else:
                print(
                    f"{logic:12s} | {machsmt:>8} | {syn:>8} | {des:>8} | {des_exp:>8} | {des_syn:>8} | {sbs:>8} | {vbs:>8}"
                )

    if args.gap_cls:
        print("=" * 80)
    else:
        print("=" * 110)

    # Write results to CSV
    output_path = Path(args.output)
    write_results_csv(
        output_path, logic_results, gap_cls=args.gap_cls, use_par2=args.par2
    )


if __name__ == "__main__":
    main()
