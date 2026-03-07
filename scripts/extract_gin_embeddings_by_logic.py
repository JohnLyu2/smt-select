#!/usr/bin/env python3
"""
Extract GIN-PWC backbone embeddings for a logic across all seed models.

Specify a logic (e.g. ABV). Uses models under models/gin_pwc/<logic>/ (e.g. seed0, seed10,
seed20, seed30, seed40). For each seed, runs extraction and writes to
data/features/gin_pwc/<logic>/seedX/ using benchmarks from
data/raw_data/smtcomp24_performance/<logic>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root on path so "scripts.extract_gin_embeddings" resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.extract_gin_embeddings import run_extraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS_BASE = Path("models/gin_pwc")
DEFAULT_PERF_DIR = Path("data/raw_data/smtcomp24_performance")
DEFAULT_FEATURES_BASE = Path("data/features/gin_pwc")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract GIN-PWC embeddings for a logic across all seed models"
    )
    parser.add_argument(
        "logic",
        type=str,
        help="Logic name (e.g. ABV). Model dir: models/gin_pwc/<logic>/seedX; benchmarks: data/raw_data/smtcomp24_performance/<logic>.json",
    )
    parser.add_argument(
        "--models-base",
        type=Path,
        default=DEFAULT_MODELS_BASE,
        help=f"Base dir for models (default: {DEFAULT_MODELS_BASE})",
    )
    parser.add_argument(
        "--perf-dir",
        type=Path,
        default=DEFAULT_PERF_DIR,
        help=f"Directory containing <logic>.json performance files (default: {DEFAULT_PERF_DIR})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_FEATURES_BASE,
        help=f"Output base for features (default: {DEFAULT_FEATURES_BASE}); writes to <output-dir>/<logic>/seedX",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=None,
        help="Root for relative instance paths (default: from src.defaults)",
    )
    parser.add_argument(
        "--graph-timeout",
        type=int,
        default=None,
        help="Graph build timeout in seconds (default: from each model config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip already-completed seeds or resume partial extraction; run from scratch",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Write partial results every N instances (default: 10); 0 to disable mid-run checkpoints",
    )
    args = parser.parse_args()

    logic = args.logic
    models_base = args.models_base.resolve()
    logic_models_dir = models_base / logic
    if not logic_models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {logic_models_dir}")

    benchmarks_json = args.perf_dir.resolve() / f"{logic}.json"
    if not benchmarks_json.is_file():
        raise FileNotFoundError(f"Benchmarks JSON not found: {benchmarks_json}")

    out_base = args.output_dir.resolve()

    seed_dirs = sorted(
        d for d in logic_models_dir.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )
    if not seed_dirs:
        raise FileNotFoundError(
            f"No seed model dirs (with config.json) found under {logic_models_dir}"
        )
    logger.info("Logic %s: %d seeds -> %s", logic, len(seed_dirs), [d.name for d in seed_dirs])

    with open(benchmarks_json) as f:
        n_benchmarks = len(json.load(f))

    def seed_complete(seed_out: Path) -> bool:
        times_csv = seed_out / "extraction_times.csv"
        if not times_csv.is_file():
            return False
        with open(times_csv) as f:
            return sum(1 for _ in f) == n_benchmarks + 1  # header + one row per benchmark

    resume = not args.no_resume
    for seed_dir in seed_dirs:
        seed_out = out_base / logic / seed_dir.name
        if resume and seed_complete(seed_out):
            logger.info("Skipping %s / %s (already complete)", logic, seed_dir.name)
            continue
        logger.info("Extracting %s / %s ...", logic, seed_dir.name)
        run_extraction(
            model_dir=seed_dir,
            benchmarks_json=benchmarks_json,
            output_dir=out_base,
            benchmark_root=args.benchmark_root,
            graph_timeout=args.graph_timeout,
            device=args.device,
            resume=resume,
            checkpoint_interval=args.checkpoint_interval,
        )
    logger.info("Done. Features written under %s/%s/", out_base, logic)


if __name__ == "__main__":
    main()
