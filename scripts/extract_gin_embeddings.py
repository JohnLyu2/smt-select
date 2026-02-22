#!/usr/bin/env python3
"""
Extract GIN-PWC backbone embeddings for benchmarks listed in a performance JSON.

Loads a saved GIN-PWC model (e.g. models/gin_pwc/ABV/seed0), iterates over all
benchmark paths from a JSON file (e.g. data/cp26/raw_data/smtcomp24_performance/ABV.json),
builds the graph for each instance, runs the backbone, and saves:
  - <out_dir>/embeddings.csv: benchmark path + columns emb_0 .. emb_{d-1}
  - <out_dir>/extraction_times.csv: benchmark, time_sec, status (status is "ok" or "failed"; graph build is capped at graph_timeout)

Instance paths in the JSON are rebased with --benchmark-root to get full .smt2 paths.
"""

from __future__ import annotations

import argparse
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
        help="Output directory for embeddings.csv and extraction_times.csv (default: data/features/gin_pwc)",
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


def run_extraction(
    model_dir: Path | str,
    benchmarks_json: Path | str,
    output_dir: Path | str,
    *,
    benchmark_root: Path | str | None = None,
    graph_timeout: int | None = None,
    device: str | None = None,
) -> None:
    """Extract GIN-PWC backbone embeddings; writes embeddings.csv and extraction_times.csv under output_dir/<division>/<seed>."""
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

    with open(benchmarks_path) as f:
        perf = json.load(f)
    relative_paths = list(perf.keys())
    logger.info("Loaded %d benchmark keys from %s", len(relative_paths), benchmarks_path)

    selector = GINPwcSelector.load(model_dir, device=device)
    graph_timeout_val = graph_timeout if graph_timeout is not None else selector.graph_timeout
    hidden_dim = selector.model.backbone.embed.embedding_dim

    embeddings: list[tuple[str, list[float]]] = []
    extraction_times: list[tuple[str, float]] = []
    failed: list[str] = []

    for rel_path in tqdm(relative_paths, desc="Extracting embeddings", unit="instance"):
        t0 = time.perf_counter()
        full_path = root / rel_path
        if not full_path.exists():
            logger.debug("Benchmark file not found: %s", full_path)
            failed.append(rel_path)
            extraction_times.append((rel_path, time.perf_counter() - t0))
            continue
        graph_dict = build_smt_graph_dict_timeout(str(full_path), graph_timeout_val)
        if graph_dict is None:
            _suppress_z3_destructor_noise()
            failed.append(rel_path)
            # Graph build time is capped at graph_timeout
            elapsed = time.perf_counter() - t0
            extraction_times.append((rel_path, min(elapsed, graph_timeout_val)))
            continue
        data = graph_dict_to_gin_data(graph_dict, selector.vocabulary)
        if data is None:
            _suppress_z3_destructor_noise()
            failed.append(rel_path)
            extraction_times.append((rel_path, time.perf_counter() - t0))
            continue
        if data.num_nodes == 0:
            failed.append(rel_path)
            extraction_times.append((rel_path, time.perf_counter() - t0))
            continue
        batch = Batch.from_data_list([data]).to(selector.device)
        with torch.no_grad():
            emb = selector.model.forward_embedding(batch)
        vec = emb[0].cpu().tolist()
        embeddings.append((rel_path, vec))
        extraction_times.append((rel_path, time.perf_counter() - t0))

    # Output under out_dir / division / seed (e.g. data/features/gin_pwc/ABV/seed0)
    division = model_dir.parent.name
    seed_name = model_dir.name
    subdir = out_dir / division / seed_name
    subdir.mkdir(parents=True, exist_ok=True)

    csv_path = subdir / "embeddings.csv"
    with open(csv_path, "w") as f:
        header = ["benchmark"] + [f"emb_{i}" for i in range(hidden_dim)]
        f.write(",".join(header) + "\n")
        for rel_path, vec in embeddings:
            row = [rel_path] + [str(x) for x in vec]
            f.write(",".join(row) + "\n")
    logger.info("Wrote %d embeddings to %s", len(embeddings), csv_path)

    failed_set = set(failed)
    times_path = subdir / "extraction_times.csv"
    with open(times_path, "w") as f:
        f.write("benchmark,time_sec,status\n")
        for rel_path, sec in extraction_times:
            status = "failed" if rel_path in failed_set else "ok"
            f.write(f"{rel_path},{sec},{status}\n")
    logger.info("Wrote extraction times for %d instances to %s", len(extraction_times), times_path)
    logger.info("Failed: %d of %d instances", len(failed), len(relative_paths))


if __name__ == "__main__":
    main()
