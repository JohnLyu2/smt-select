#!/usr/bin/env python3
"""
Script to split a performance CSV file into training and testing sets.
"""

import csv
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_csv(input_path, train_path, test_path, test_size=0.2, seed=42):
    """
    Split a CSV file into training and testing sets.

    Args:
        input_path: Path to input CSV file
        train_path: Path to save training CSV file
        test_path: Path to save testing CSV file
        test_size: Proportion of data to use for testing (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    """
    input_path = Path(input_path)
    train_path = Path(train_path)
    test_path = Path(test_path)

    # Read the CSV file
    with input_path.open(mode="r", newline="") as csv_file:
        csv_reader = csv.reader(csv_file)

        # Read header rows (first two rows)
        header1 = next(csv_reader)
        header2 = next(csv_reader)

        # Read all data rows
        data_rows = list(csv_reader)

    train_rows, test_rows = train_test_split(
        data_rows, test_size=test_size, random_state=seed, shuffle=True
    )

    # Write training set
    with train_path.open(mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header1)
        csv_writer.writerow(header2)
        csv_writer.writerows(train_rows)

    # Write testing set
    with test_path.open(mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header1)
        csv_writer.writerow(header2)
        csv_writer.writerows(test_rows)

    print("Split complete!")
    print(f"  Total rows: {len(data_rows)}")
    print(
        f"  Training rows: {len(train_rows)} ({len(train_rows) / len(data_rows) * 100:.1f}%)"
    )
    print(
        f"  Testing rows: {len(test_rows)} ({len(test_rows) / len(data_rows) * 100:.1f}%)"
    )
    print(f"  Training file: {train_path}")
    print(f"  Testing file: {test_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split qflia.csv into training and testing sets"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file path",
    )
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Output training CSV file path",
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Output testing CSV file path",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    split_csv(
        args.input,
        args.train,
        args.test,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
