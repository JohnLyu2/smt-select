"""Save Weisfeiler-Lehman features for instance paths from a CSV or performance JSON.

Input can be:
- A performance JSON (e.g. data/cp26/raw_data/smtcomp24_performance/BV.json).
  Instance paths are the top-level keys; --benchmark-root is required.
- A CSV with a path column. Use --benchmark-root if paths are relative.

Builds SMT graphs, fits WL, extracts per-level features. Saves one CSV per WL
level (level_0.csv, level_1.csv, ...) and timeout{N}_failed_paths.txt.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from grakel.kernels import WeisfeilerLehman

from .performance import parse_performance_json
from .pwc_wl import generate_graph_dict


def _normalize_path(path: str) -> str:
    """Normalize path for matching (same as feature.py)."""
    return path.strip().replace("\\", "/")


def paths_from_performance_json(
    json_path: str | Path,
    benchmark_root: str | Path,
    timeout: float = 1200.0,
) -> list[str]:
    """Read instance paths from a performance JSON via parse_performance_json; rebase with benchmark_root."""
    multi_perf = parse_performance_json(str(json_path), timeout)
    root = Path(benchmark_root).resolve()
    return [str(root / _normalize_path(p)) for p in multi_perf.keys()]


def extract_wl_features_from_fitted(
    wl_kernel: WeisfeilerLehman,
) -> dict[int, np.ndarray]:
    """Extract per-level feature matrices from a fitted WeisfeilerLehman kernel."""
    features: dict[int, np.ndarray] = {}
    for level, base in wl_kernel.X.items():
        X = base.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        features[level] = np.asarray(X, dtype=np.float64)
    return features


def paths_from_csv(
    csv_path: str | Path,
    path_column: str = "path",
    benchmark_root: str | Path | None = None,
) -> list[str]:
    """Read instance paths from a CSV (one path per row in path_column)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    paths: list[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if path_column not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV must have column '{path_column}'. Found: {reader.fieldnames}"
            )
        for row in reader:
            p = _normalize_path(row[path_column])
            if not p:
                continue
            if benchmark_root is not None:
                p = str(Path(benchmark_root) / p)
            paths.append(p)
    return paths


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
    Level CSVs: level_0.csv, level_1.csv, ... Failed paths: timeout{graph_timeout}_failed_paths.txt.
    Paths in CSVs and failed_paths are relative to benchmark_root when set.
    Returns (number of rows written, list of failed paths as stored in file).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_prefix = f"timeout{graph_timeout}"

    graph_dict, failed_list = generate_graph_dict(instance_paths, graph_timeout)
    if not graph_dict:
        raise ValueError(
            "No graphs could be built (all failed: timeout/recursion/error?). "
            "Increase graph_timeout or check instance paths."
        )

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
                for j in range(n_features):
                    row[f"wl_{j}"] = F[i, j]
                writer.writerow(row)
        logging.info("Wrote %s: %d rows, %d dims", csv_path.name, n_rows, n_features)

    failed_paths_file = output_dir / f"{failed_prefix}_failed_paths.txt"
    with open(failed_paths_file, "w", encoding="utf-8") as f:
        for p in failed_list:
            f.write(_path_for_csv(p, benchmark_root) + "\n")
    if failed_list:
        logging.info("Wrote %s: %d failed paths", failed_paths_file.name, len(failed_list))

    return len(train_paths), failed_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save WL features for instance paths from a CSV or performance JSON (e.g. data/cp26/raw_data/smtcomp24_performance/BV.json)",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input: CSV (path column) or performance JSON (.json). For JSON, paths are top-level keys; use --benchmark-root.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory. Saves level_0.csv, level_1.csv, ... per WL level and timeout{N}_failed_paths.txt (N = --graph-timeout).",
    )
    parser.add_argument(
        "--path-column",
        type=str,
        default="path",
        help="CSV column name for instance paths (default: path). Ignored for JSON input.",
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
        "--benchmark-root",
        type=str,
        default=None,
        help="Root directory for instance paths. Required for JSON input; optional for CSV (prepends to relative paths).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Timeout in seconds for parse_performance_json (default: 1200.0). Used for JSON input only.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    if input_path.suffix.lower() == ".json":
        if not args.benchmark_root:
            raise ValueError(
                "Performance JSON input requires --benchmark-root (paths in JSON are relative, e.g. BV/.../file.smt2)."
            )
        paths = paths_from_performance_json(
            args.input, args.benchmark_root, timeout=args.timeout
        )
    else:
        paths = paths_from_csv(
            args.input,
            path_column=args.path_column,
            benchmark_root=args.benchmark_root,
        )
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
