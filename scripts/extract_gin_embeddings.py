#!/usr/bin/env python3
"""
Extract GIN-PWC backbone embeddings for benchmarks listed in a performance JSON.

Loads a saved GIN-PWC model (e.g. models/gin_pwc/ABV/seed0), iterates over all
benchmark paths from a JSON file (e.g. data/cp26/raw_data/smtcomp24_performance/ABV.json),
builds the graph for each instance, runs the backbone, and saves:
  - <out_dir>/features.csv: path + columns emb_0 .. emb_{d-1}
  - <out_dir>/extraction_times.csv: path, time_sec, failed (0 or 1; graph build is capped at graph_timeout)

Instance paths in the JSON are rebased with --benchmark-root to get full .smt2 paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm
from torch_geometric.data import Batch

from src.defaults import DEFAULT_BENCHMARK_ROOT
from src.gin_ehm import graph_dict_to_gin_data
from src.gin_pwc import GINPwcSelector
from src.graph_rep import build_smt_graph_dict_timeout, _suppress_z3_destructor_noise

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract GIN-PWC backbone embeddings for benchmarks in a performance JSON"
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory with saved GIN-PWC model (config.json, model.pt, vocab.json), e.g. models/gin_pwc/ABV/seed0",
    )
    parser.add_argument(
        "benchmarks_json",
        type=Path,
        help="JSON file listing benchmarks (keys = relative paths), e.g. data/cp26/raw_data/smtcomp24_performance/ABV.json",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data/features/gin_pwc"),
        help="Output directory for features.csv and extraction_times.csv (default: data/features/gin_pwc)",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=None,
        help=f"Root for relative instance paths (default: {DEFAULT_BENCHMARK_ROOT})",
    )
    parser.add_argument(
        "--graph-timeout",
        type=int,
        default=None,
        help="Graph build timeout in seconds (default: from model config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model (default: cuda if available else cpu)",
    )
    args = parser.parse_args()
    run_extraction(
        model_dir=args.model_dir,
        benchmarks_json=args.benchmarks_json,
        output_dir=args.output_dir,
        benchmark_root=args.benchmark_root,
        graph_timeout=args.graph_timeout,
        device=args.device,
    )


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
    """Extract GIN-PWC backbone embeddings; writes features.csv and extraction_times.csv under output_dir/<division>/<seed>.
    If resume=True and partial results exist, skips already-done instances and appends; checkpoints every checkpoint_interval instances."""
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
    for idx, rel_path in enumerate(tqdm(relative_paths, desc="Extracting embeddings", unit="instance")):
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
    logger.info("Wrote extraction times for %d instances to %s", len(extraction_times), subdir / "extraction_times.csv")
    logger.info("Failed: %d of %d instances", len(failed), len(all_relative_paths))


if __name__ == "__main__":
    main()
