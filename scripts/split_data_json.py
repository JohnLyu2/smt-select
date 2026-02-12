#!/usr/bin/env python3
"""
Script to split a performance JSON file into training and testing sets.
JSON is expected to be an object: benchmark_path -> { solver -> result data }.
Keys (benchmarks) are split; structure of each value is preserved.
"""

import json
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split


def split_json(
    input_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Split a performance JSON into training and testing sets by benchmark keys.

    Args:
        input_path: Path to input JSON file
        train_path: Path to save training JSON file
        test_path: Path to save testing JSON file
        test_size: Proportion of data to use for testing (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    """
    input_path = Path(input_path)
    train_path = Path(train_path)
    test_path = Path(test_path)

    with input_path.open() as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object (dict)")

    keys = list(data.keys())
    train_keys, test_keys = train_test_split(
        keys, test_size=test_size, random_state=seed, shuffle=True
    )

    train_data = {k: data[k] for k in train_keys}
    test_data = {k: data[k] for k in test_keys}

    with train_path.open("w") as f:
        json.dump(train_data, f, indent=2)

    with test_path.open("w") as f:
        json.dump(test_data, f, indent=2)

    n = len(keys)
    print("Split complete!")
    print(f"  Total benchmarks: {n}")
    print(f"  Training: {len(train_keys)} ({len(train_keys) / n * 100:.1f}%)")
    print(f"  Testing: {len(test_keys)} ({len(test_keys) / n * 100:.1f}%)")
    print(f"  Training file: {train_path}")
    print(f"  Testing file: {test_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a performance JSON into training and testing sets"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file path (e.g. data/cp26/raw_data/smtcomp24_performance/BV.json)",
    )
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Output training JSON file path",
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Output testing JSON file path",
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

    split_json(
        args.input,
        args.train,
        args.test,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
