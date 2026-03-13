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
import csv
import json
import logging
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from src.defaults import DEFAULT_BENCHMARK_ROOT
from src.gin_ehm import graph_dict_to_gin_data
from src.gin_pwc import GINPwcSelector
from src.graph_rep import _suppress_z3_destructor_noise, build_smt_graph_dict_timeout

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS_BASE = Path("models/gin_pwc")
DEFAULT_PERF_DIR = Path("data/raw_data/smtcomp24_performance")
DEFAULT_FEATURES_BASE = Path("data/features/graph")


def _write_checkpoint(
    subdir: Path,
    hidden_dim: int,
    embeddings: list[tuple[str, list[float]]],
    extraction_times: list[tuple[str, float]],
    failed: list[str],
) -> None:
    """Write current state to features.csv and extraction_times.csv for resume/partial results."""
    failed_set = set(failed)
    csv_path = subdir / "features.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path"] + [f"emb_{i}" for i in range(hidden_dim)])
        for rel_path, vec in embeddings:
            writer.writerow([rel_path] + [str(x) for x in vec])
    times_path = subdir / "extraction_times.csv"
    with open(times_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "time_sec", "failed"])
        for rel_path, sec in extraction_times:
            writer.writerow([rel_path, sec, "1" if rel_path in failed_set else "0"])


def _load_existing_checkpoint(
    subdir: Path,
) -> tuple[list[tuple[str, list[float]]], list[tuple[str, float]], list[str], int] | None:
    """Load existing features.csv and extraction_times.csv if present. Returns (embeddings, times, failed, hidden_dim) or None."""
    csv_path = subdir / "features.csv"
    times_path = subdir / "extraction_times.csv"
    if not csv_path.is_file() or not times_path.is_file():
        return None
    embeddings: list[tuple[str, list[float]]] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or header[0] != "path":
            return None
        hidden_dim = len(header) - 1
        for row in reader:
            if len(row) < 2:
                continue
            path, *rest = row
            try:
                vec = [float(x) for x in rest[:hidden_dim]]
            except ValueError:
                continue
            embeddings.append((path, vec))
    extraction_times: list[tuple[str, float]] = []
    failed: list[str] = []
    with open(times_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 3:
                continue
            path, time_str, failed_val = row[0], row[1], row[2]
            try:
                sec = float(time_str)
            except ValueError:
                sec = 0.0
            extraction_times.append((path, sec))
            if failed_val.strip() == "1":
                failed.append(path)
    return embeddings, extraction_times, failed, hidden_dim


def run_extraction(
    model_dir: Path | str,
    benchmarks_json: Path | str,
    output_dir: Path | str,
    *,
    benchmark_root: Path | str | None = None,
    graph_timeout: int | None = None,
    device: str | None = None,
    resume: bool = True,
    checkpoint_interval: int = 10,
) -> None:
    """Extract GIN-PWC backbone embeddings for a single seed model.

    Writes features.csv and extraction_times.csv under output_dir/<division>/<seed>.
    If resume=True and partial results exist, skips already-done instances and appends;
    checkpoints every checkpoint_interval instances.
    """
    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir() or not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Model directory not found or missing config.json: {model_dir}")

    benchmarks_path = Path(benchmarks_json).resolve()
    if not benchmarks_path.is_file():
        raise FileNotFoundError(f"Benchmarks JSON not found: {benchmarks_path}")

    root = Path(benchmark_root or DEFAULT_BENCHMARK_ROOT).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark root is not a directory: {root}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    division = model_dir.parent.name
    seed_name = model_dir.name
    subdir = out_dir / division / seed_name
    subdir.mkdir(parents=True, exist_ok=True)

    with open(benchmarks_path) as f:
        perf = json.load(f)
    all_relative_paths = list(perf.keys())
    logger.info("Loaded %d benchmark keys from %s", len(all_relative_paths), benchmarks_path)

    selector = GINPwcSelector.load(model_dir, device=device)
    graph_timeout_val = graph_timeout if graph_timeout is not None else selector.graph_timeout
    hidden_dim = selector.model.backbone.embed.embedding_dim

    embeddings: list[tuple[str, list[float]]] = []
    extraction_times: list[tuple[str, float]] = []
    failed: list[str] = []
    relative_paths = all_relative_paths

    if resume:
        existing = _load_existing_checkpoint(subdir)
        if existing is not None:
            embeddings, extraction_times, failed, loaded_dim = existing
            if loaded_dim != hidden_dim:
                logger.warning(
                    "Existing checkpoint hidden_dim=%d != model hidden_dim=%d; ignoring checkpoint",
                    loaded_dim,
                    hidden_dim,
                )
            else:
                completed = {p for p, _ in extraction_times}
                relative_paths = [p for p in all_relative_paths if p not in completed]
                logger.info(
                    "Resuming: %d already done, %d remaining",
                    len(completed),
                    len(relative_paths),
                )

    # Every branch below appends exactly one (rel_path, time) to extraction_times so nothing is missed.
    for idx, rel_path in enumerate(
        tqdm(relative_paths, desc="Extracting embeddings", unit="instance")
    ):
        t0 = time.perf_counter()
        full_path = root / rel_path
        if not full_path.exists():
            logger.error("Benchmark file not found: %s", full_path)
            failed.append(rel_path)
            extraction_times.append((rel_path, time.perf_counter() - t0))
            continue
        graph_dict = build_smt_graph_dict_timeout(str(full_path), graph_timeout_val)
        if graph_dict is None:
            _suppress_z3_destructor_noise()
            failed.append(rel_path)
            elapsed = time.perf_counter() - t0
            extraction_times.append((rel_path, min(elapsed, graph_timeout_val)))
            continue
        data = graph_dict_to_gin_data(graph_dict, selector.vocabulary)
        if data is None:
            del graph_dict
            _suppress_z3_destructor_noise()
            failed.append(rel_path)
            extraction_times.append((rel_path, time.perf_counter() - t0))
            continue
        if data.num_nodes == 0:
            del graph_dict, data
            failed.append(rel_path)
            extraction_times.append((rel_path, time.perf_counter() - t0))
            continue
        batch = Batch.from_data_list([data]).to(selector.device)
        with torch.no_grad():
            emb = selector.model.forward_embedding(batch)
        vec = emb[0].cpu().tolist()
        embeddings.append((rel_path, vec))
        extraction_times.append((rel_path, time.perf_counter() - t0))
        del graph_dict, data, batch, emb
        dev = selector.device
        if (isinstance(dev, torch.device) and dev.type == "cuda") or (
            isinstance(dev, str) and dev.startswith("cuda")
        ):
            torch.cuda.empty_cache()

        if checkpoint_interval > 0 and (idx + 1) % checkpoint_interval == 0:
            _write_checkpoint(subdir, hidden_dim, embeddings, extraction_times, failed)
            logger.debug("Checkpoint: %d instances written", len(extraction_times))

    _write_checkpoint(subdir, hidden_dim, embeddings, extraction_times, failed)
    logger.info("Wrote %d embeddings to %s", len(embeddings), subdir / "features.csv")
    logger.info(
        "Wrote extraction times for %d instances to %s",
        len(extraction_times),
        subdir / "extraction_times.csv",
    )
    logger.info("Failed: %d of %d instances", len(failed), len(all_relative_paths))


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
