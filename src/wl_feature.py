"""Save Weisfeiler-Lehman features for instance paths from a performance JSON.

Input: performance JSON (e.g. data/cp26/raw_data/smtcomp24_performance/BV.json).
Instance paths are the top-level keys; --benchmark-root is required.

Builds SMT graphs, fits WL, extracts per-level features. Saves one CSV per WL
level (level_0.csv, level_1.csv, ...) and failed_paths.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import scipy.sparse
from grakel import Graph
from grakel.kernels import WeisfeilerLehman
from tqdm import tqdm

from .graph_rep import (
    build_smt_graph_dict_timeout,
    smt_graph_to_grakel,
    _suppress_z3_destructor_noise,
)


def build_smt_graph_timeout(smt_path: str | Path, timeout_sec: int) -> Graph | None:
    """Build a GraKel graph for the SMT instance with a timeout. Returns None on timeout/error."""
    graph_dict = build_smt_graph_dict_timeout(smt_path, timeout_sec)
    if graph_dict is None:
        return None
    return smt_graph_to_grakel(graph_dict)


def _normalize_path(path: str) -> str:
    """Normalize path for matching (same as feature.py)."""
    return path.strip().replace("\\", "/")


def paths_from_performance_json(
    json_path: str | Path,
    benchmark_root: str | Path,
) -> list[str]:
    """Read instance paths (top-level keys) from a performance JSON; rebase with benchmark_root."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Performance JSON root must be an object")
    root = Path(benchmark_root).resolve()
    return [str(root / _normalize_path(p)) for p in data if isinstance(data.get(p), dict)]


def extract_wl_features_from_fitted(
    wl_kernel: WeisfeilerLehman,
) -> dict[int, np.ndarray | scipy.sparse.spmatrix]:
    """Extract per-level feature matrices from a fitted WeisfeilerLehman kernel.

    Keeps sparse matrices sparse to avoid huge allocations; callers must write
    row-by-row (densify one row at a time) when writing CSVs.
    """
    features: dict[int, np.ndarray | scipy.sparse.spmatrix] = {}
    for level, base in wl_kernel.X.items():
        logging.debug("Extracting level %d", level)
        X = base.X
        if scipy.sparse.issparse(X):
            features[level] = X
        else:
            features[level] = np.asarray(X, dtype=np.float64)
    return features


def _path_for_csv(path: str, benchmark_root: str | Path | None) -> str:
    """Return path as stored in CSV: relative to benchmark_root if set, else unchanged."""
    if benchmark_root is None:
        return path
    root = Path(benchmark_root).resolve()
    try:
        rel = Path(path).resolve().relative_to(root)
        return _normalize_path(str(rel))
    except ValueError:
        return path


def save_wl_features_to_dir(
    instance_paths: list[str],
    output_dir: str | Path,
    wl_iter: int = 2,
    graph_timeout: int = 10,
    benchmark_root: str | Path | None = None,
) -> tuple[int, list[str]]:
    """
    Build graphs for instance_paths, fit WL, save one CSV per level and a failed-paths list.

    output_dir: directory to write into (created if needed).
    Level CSVs: level_0.csv, level_1.csv, ... Failed paths: failed_paths.txt.
    Paths in CSVs and failed_paths are relative to benchmark_root when set.
    Returns (number of rows written, list of failed paths as stored in file).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_dict: dict[str, Graph] = {}
    failed_list: list[str] = []
    for p in tqdm(instance_paths, desc="Building graphs", unit="instance"):
        g = build_smt_graph_timeout(p, graph_timeout)
        if g is not None:
            graph_dict[p] = g
        else:
            failed_list.append(p)
            _suppress_z3_destructor_noise()
    logging.info(
        "Graphs: %d built, %d failed (of %d instances)",
        len(graph_dict),
        len(failed_list),
        len(instance_paths),
    )
    if not graph_dict:
        raise ValueError(
            "No graphs could be built (all failed: timeout/recursion/error?). "
            "Increase graph_timeout or check instance paths."
        )

    # Write failed paths before WL extraction so it exists even if extraction fails (e.g. OOM).
    failed_paths_file = output_dir / "failed_paths.txt"
    with open(failed_paths_file, "w", encoding="utf-8") as f:
        for p in failed_list:
            f.write(_path_for_csv(p, benchmark_root) + "\n")
    if failed_list:
        logging.info("Wrote %s: %d failed paths", failed_paths_file.name, len(failed_list))

    train_paths = list(graph_dict.keys())
    train_graphs = [graph_dict[p] for p in train_paths]

    wl_kernel = WeisfeilerLehman(n_iter=wl_iter, normalize=True)
    wl_kernel.fit(train_graphs)
    per_level = extract_wl_features_from_fitted(wl_kernel)

    for level in sorted(per_level.keys()):
        F = per_level[level]
        n_rows, n_features = F.shape
        csv_path = output_dir / f"level_{level}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["path"] + [f"wl_{j}" for j in range(n_features)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, path in enumerate(train_paths):
                row = {"path": _path_for_csv(path, benchmark_root)}
                if scipy.sparse.issparse(F):
                    row_vec = F.getrow(i).toarray().ravel()
                    for j in range(n_features):
                        row[f"wl_{j}"] = row_vec[j]
                else:
                    for j in range(n_features):
                        row[f"wl_{j}"] = F[i, j]
                writer.writerow(row)
        logging.info("Wrote %s: %d rows, %d dims", csv_path.name, n_rows, n_features)

    return len(train_paths), failed_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save WL features for instance paths from a performance JSON (e.g. data/cp26/raw_data/smtcomp24_performance/BV.json)",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Performance JSON path. Instance paths are top-level keys.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory. Saves level_0.csv, level_1.csv, ... per WL level and failed_paths.txt.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=str,
        required=True,
        help="Root directory for instance paths (paths in JSON are relative, e.g. BV/.../file.smt2).",
    )
    parser.add_argument(
        "--wl-iter",
        type=int,
        default=2,
        help="Weisfeiler-Lehman iteration count (default: 2)",
    )
    parser.add_argument(
        "--graph-timeout",
        type=int,
        default=10,
        help="Graph build timeout per instance in seconds (default: 10)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (e.g. which level is being extracted).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if input_path.suffix.lower() != ".json":
        raise ValueError("Input must be a performance JSON file (.json).")

    paths = paths_from_performance_json(args.input, args.benchmark_root)
    if not paths:
        raise ValueError(f"No paths found in {args.input}")
    logging.info("Loaded %d paths from %s", len(paths), args.input)

    save_wl_features_to_dir(
        paths,
        args.output,
        wl_iter=args.wl_iter,
        graph_timeout=args.graph_timeout,
        benchmark_root=args.benchmark_root,
    )


if __name__ == "__main__":
    main()
