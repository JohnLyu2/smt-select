#!/usr/bin/env python3
"""
Script to split a performance JSON file into k folds.

Each fold is saved as a separate JSON file in the output folder (0.json, 1.json, ...).
Format is the same as input: { benchmark_path: { solver -> result data } }.
"""

import argparse
import json
from pathlib import Path

from sklearn.model_selection import KFold


def split_into_folds(
    input_path: str | Path,
    output_dir: str | Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> None:
    """
    Split a performance JSON into k folds and save each fold as a separate file.

    Args:
        input_path: Path to input performance JSON file
        output_dir: Directory to save fold JSON files (0.json, 1.json, ...)
        n_splits: Number of folds (default: 5)
        random_state: Random seed for reproducibility (default: 42)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object (dict)")

    keys = list(data.keys())
    n_instances = len(keys)
    if n_instances < n_splits:
        raise ValueError(
            f"Not enough instances ({n_instances}) for {n_splits} folds. "
            f"Need at least {n_splits} instances."
        )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_num, (_, test_idx) in enumerate(kf.split(keys)):
        fold_keys = [keys[i] for i in test_idx]
        fold_data = {k: data[k] for k in fold_keys}
        fold_path = output_dir / f"{fold_num}.json"
        with fold_path.open("w", encoding="utf-8") as f:
            json.dump(fold_data, f, indent=2)
        print(
            f"Fold {fold_num + 1}/{n_splits}: {len(fold_keys)} instances -> {fold_path}"
        )

    print("\nSplit complete!")
    print(f"  Total instances: {n_instances}")
    print(f"  Number of folds: {n_splits}")
    print(f"  Instances per fold: ~{n_instances // n_splits}")
    print(f"  Output directory: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a performance JSON file into k folds"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input performance JSON file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save fold JSON files (0.json, 1.json, ...)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=4,
        help="Number of folds (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    split_into_folds(
        args.input,
        args.output_dir,
        n_splits=args.n_splits,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
