#!/usr/bin/env python3
"""
Generate a CSV summary of cross-validation results.

Scans data/cv_results for all method folders and extracts gap_cls_par2 metrics
(mean ± std) for train and test sets. Creates a CSV with:
- Rows: logics
- Columns: {method}_train, {method}_test
- Values: mean ± std format
"""

import argparse
import json
from pathlib import Path
import csv


def load_summary_json(summary_path: Path) -> dict:
    """Load and return the summary.json file as a dictionary."""
    with open(summary_path, "r") as f:
        return json.load(f)


def format_mean_std(mean: float, std: float, precision: int = 4) -> str:
    """Format mean and std as 'mean ± std'."""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def discover_methods_and_logics(
    results_dir: Path,
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Discover all methods and their associated logics.

    Args:
        results_dir: Base directory containing CV results (e.g., data/cv_results)

    Returns:
        Tuple of (method_to_logics dict, all_logics list)
    """
    method_to_logics = {}
    all_logics = set()

    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return method_to_logics, []

    for method_dir in sorted(results_dir.iterdir()):
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name
        logics = []

        for logic_dir in sorted(method_dir.iterdir()):
            if not logic_dir.is_dir():
                continue

            summary_path = logic_dir / "summary.json"
            if summary_path.exists():
                logics.append(logic_dir.name)
                all_logics.add(logic_dir.name)

        if logics:
            method_to_logics[method_name] = sorted(logics)

    return method_to_logics, sorted(all_logics)


def extract_gap_cls_par2(summary: dict, split: str) -> tuple[float, float] | None:
    """
    Extract gap_cls_par2 mean and std from summary.

    Args:
        summary: Summary dictionary from JSON
        split: Either 'train' or 'test'

    Returns:
        Tuple of (mean, std) or None if not found
    """
    aggregated = summary.get("aggregated", {})
    mean_key = f"{split}_gap_cls_par2_mean"
    std_key = f"{split}_gap_cls_par2_std"

    if mean_key in aggregated and std_key in aggregated:
        return aggregated[mean_key], aggregated[std_key]
    return None


def generate_summary_csv(
    results_dir: Path,
    output_csv: Path,
    precision: int = 4,
    show_train: bool = False,
):
    """
    Generate CSV summary of CV results.

    Args:
        results_dir: Base directory containing CV results
        output_csv: Path to output CSV file
        precision: Number of decimal places for formatting
        show_train: Include train statistics if True
    """
    # Discover methods and logics
    method_to_logics, all_logics = discover_methods_and_logics(results_dir)

    if not method_to_logics:
        print("No methods found in results directory.")
        return

    print(
        f"Found {len(method_to_logics)} methods: {', '.join(method_to_logics.keys())}"
    )
    print(f"Found {len(all_logics)} logics: {', '.join(all_logics)}")

    # Build data structure: logic -> method -> (train_value, test_value)
    data = {logic: {} for logic in all_logics}

    for method_name, logics in method_to_logics.items():
        for logic in logics:
            summary_path = results_dir / method_name / logic / "summary.json"
            if not summary_path.exists():
                print(f"Warning: Missing summary.json for {method_name}/{logic}")
                continue

            try:
                summary = load_summary_json(summary_path)

                if show_train:
                    # Extract train metrics
                    train_metrics = extract_gap_cls_par2(summary, "train")
                    if train_metrics:
                        mean, std = train_metrics
                        # Multiply by 100 to show as percentage
                        mean *= 100
                        std *= 100
                        data[logic][f"{method_name}_train"] = format_mean_std(
                            mean, std, precision=1
                        )
                    else:
                        data[logic][f"{method_name}_train"] = "N/A"

                # Extract test metrics
                test_metrics = extract_gap_cls_par2(summary, "test")
                if test_metrics:
                    mean, std = test_metrics
                    # Multiply by 100 to show as percentage
                    mean *= 100
                    std *= 100
                    data[logic][f"{method_name}_test"] = format_mean_std(
                        mean, std, precision=1
                    )
                else:
                    data[logic][f"{method_name}_test"] = "N/A"

            except Exception as e:
                print(f"Error processing {method_name}/{logic}: {e}")
                if show_train:
                    data[logic][f"{method_name}_train"] = "ERROR"
                data[logic][f"{method_name}_test"] = "ERROR"

    # Build column names: all methods with _test suffixes, optional _train
    columns = ["logic"]
    for method_name in sorted(method_to_logics.keys()):
        if show_train:
            columns.append(f"{method_name}_train")
        columns.append(f"{method_name}_test")

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for logic in all_logics:
            row = [logic]
            for method_name in sorted(method_to_logics.keys()):
                if show_train:
                    row.append(data[logic].get(f"{method_name}_train", "N/A"))
                row.append(data[logic].get(f"{method_name}_test", "N/A"))
            writer.writerow(row)

    print(f"\nSummary CSV written to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV summary of cross-validation results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/cv_results",
        help="Base directory containing CV results (default: data/cv_results)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cv_results/summary.csv",
        help="Output CSV file path (default: data/cv_results/summary.csv)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places for formatting (default: 4)",
    )
    parser.add_argument(
        "--show-train",
        action="store_true",
        help="Include train statistics in the output CSV",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_csv = Path(args.output)

    generate_summary_csv(results_dir, output_csv, args.precision, args.show_train)


if __name__ == "__main__":
    main()
