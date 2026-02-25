#!/usr/bin/env python3
"""
Run evaluate_multi_splits.py for all logics that have syntactic features and splits.

Expects folder structure like data/features/syntactic with one dir per logic (ABV, ALIA, ...),
each containing features.csv (and optionally extraction_times.csv).
Uses splits from data/cp26/performance_splits/smtcomp24/<logic>/.
Saves results to data/cp26/results/synt/<logic>/.

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
FEATURES_DIR = PROJECT_ROOT / "data" / "features" / "syntactic"
SPLITS_BASE = PROJECT_ROOT / "data" / "cp26" / "performance_splits" / "smtcomp24"
RESULTS_BASE = PROJECT_ROOT / "data" / "cp26" / "results" / "synt"
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_multi_splits.py"


def _run_one(logic: str, results_base: Path, features_dir: Path) -> str | None:
    """Run evaluate_multi_splits for one logic. Returns logic name on failure, None on success."""
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
            "--features-dir", str(features_dir),
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
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=FEATURES_DIR,
        help=f"Base dir with per-logic subdirs containing features.csv (default: {FEATURES_DIR})",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_BASE,
        help=f"Base dir for results (default: {RESULTS_BASE})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    features_dir = args.features_dir.resolve()
    if not features_dir.is_dir():
        logging.error("Features dir not found: %s", features_dir)
        return 1
    if not SCRIPT.is_file():
        logging.error("Script not found: %s", SCRIPT)
        return 1

    # Find logic dirs that contain features.csv
    logics = sorted(
        d.name for d in features_dir.iterdir()
        if d.is_dir() and (d / "features.csv").is_file()
    )
    if not logics:
        logging.error("No logic dirs with features.csv in %s", features_dir)
        return 1

    results_base = args.results_dir.resolve()
    results_base.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    jobs = max(1, args.jobs)

    if jobs == 1:
        for logic in logics:
            f = _run_one(logic, results_base, features_dir)
            if f is not None:
                failed.append(f)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(_run_one, logic, results_base, features_dir): logic for logic in logics}
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
