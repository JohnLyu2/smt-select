#!/usr/bin/env python3
"""
Run evaluate_multi_splits.py for all logics that have syntactic feature CSV and splits.

Uses features from data/features/syntactic/catalog_all/<logic>.csv and
splits from data/cp26/performance_splits/smtcomp24/<logic>/.
Saves results to data/cp26/results/synt/catalog_0_pwc_svm/<logic>/.

Usage (from project root, with venv activated):
  python scripts/run_syntactic_splits.py
  python scripts/run_syntactic_splits.py --jobs 4
"""

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Project root when run as: python scripts/run_syntactic_splits.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "syntactic" / "catalog_all"
SPLITS_BASE = PROJECT_ROOT / "data" / "cp26" / "performance_splits" / "smtcomp24"
RESULTS_BASE = PROJECT_ROOT / "data" / "cp26" / "results" / "synt" / "catalog_0_pwc_svm"
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_multi_splits.py"


def _run_one(csv_path: Path, results_base: Path) -> str | None:
    """Run evaluate_multi_splits for one logic. Returns logic name on failure, None on success."""
    logic = csv_path.stem
    splits_dir = SPLITS_BASE / logic
    output_dir = results_base / logic
    if not splits_dir.is_dir():
        logging.warning("Skipping %s: no splits dir %s", logic, splits_dir)
        return None
    logging.info("Running %s -> %s", logic, output_dir)
    ret = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--splits-dir", str(splits_dir),
            "--feature-csv", str(csv_path),
            "--output-dir", str(output_dir),
        ],
        cwd=str(PROJECT_ROOT),
    )
    if ret.returncode != 0:
        return logic
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluate_multi_splits for all syntactic logics")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Max number of logics to run in parallel (default: 1)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if not FEATURE_DIR.is_dir():
        logging.error("Feature dir not found: %s", FEATURE_DIR)
        return 1
    if not SCRIPT.is_file():
        logging.error("Script not found: %s", SCRIPT)
        return 1

    csv_files = sorted(FEATURE_DIR.glob("*.csv"))
    if not csv_files:
        logging.error("No CSV files in %s", FEATURE_DIR)
        return 1

    results_base = RESULTS_BASE.resolve()
    results_base.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    jobs = max(1, args.jobs)

    if jobs == 1:
        for csv_path in csv_files:
            f = _run_one(csv_path, results_base)
            if f is not None:
                failed.append(f)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(_run_one, csv_path, results_base): csv_path for csv_path in csv_files}
            for future in as_completed(futures):
                f = future.result()
                if f is not None:
                    failed.append(f)

    if failed:
        logging.error("Failed: %s", failed)
        return 1
    logging.info("Done. Results in %s", results_base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
