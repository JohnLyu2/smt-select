#!/usr/bin/env python3
"""
Build gin_pwc_fb results: for each logic, merge GIN-PWC eval CSVs with synt SVM,
replacing feature-fail rows (feature_fail != 0) with synt's result. Assert synt row exists.
When feature_fail != 0, output overhead = gin_overhead + synt_overhead (synt_overhead from
synt extraction_times.csv).

Output: data/cp26/results/graph/lite_fallback/<logic>/seedN/train_eval.csv, test_eval.csv,
        and summary.json (same shape as gin_pwc).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.defaults import DEFAULT_BENCHMARK_ROOT
from src.evaluate import compute_metrics
from src.evaluate import load_extraction_times_csv
from src.performance import parse_as_perf_csv
from src.utils import normalize_path
from src.performance import parse_performance_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIN_PWC_DIR = PROJECT_ROOT / "data" / "cp26" / "results" / "graph" / "before_fallback"
SYNT_DIR = PROJECT_ROOT / "data" / "cp26" / "results" / "lite"
SYNT_EXTRACTION_BASE = PROJECT_ROOT / "data" / "features" / "syntactic"
SPLITS_BASE = PROJECT_ROOT / "data" / "cp26" / "performance_splits" / "smtcomp24"
OUT_ROOT = PROJECT_ROOT / "data" / "cp26" / "results" / "graph" / "lite_fallback"
SEEDS = [0, 10, 20, 30, 40]

GIN_HEADER = ["benchmark", "selected", "solved", "runtime", "solver_runtime", "overhead", "feature_fail"]


def load_gin_csv(path: Path) -> list[dict]:
    """Load GIN eval CSV; return list of dicts with GIN_HEADER keys."""
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def load_synt_lookup(path: Path) -> dict[str, dict]:
    """Load synt CSV; return benchmark -> {selected, solved, runtime}."""
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            bench = row.get("benchmark", "").strip()
            if not bench:
                continue
            out[bench] = {
                "selected": row.get("selected", "").strip(),
                "solved": int(row["solved"]) if row.get("solved") else 0,
                "runtime": float(row["runtime"]) if row.get("runtime") else 0.0,
            }
    return out


def merge_single_split(
    gin_rows: list[dict],
    synt_lookup: dict[str, dict],
    synt_extraction_by_path: dict[str, float],
    logic: str,
    seed: int,
    split: str,
    timeout: float = 1200.0,
) -> list[dict]:
    """Replace feature_fail rows with synt; assert synt row exists for each.
    When feature_fail != 0, output overhead = gin_overhead + synt_overhead (from extraction_times).
    """
    out = []
    for row in gin_rows:
        bench = row.get("benchmark", "").strip()
        try:
            ff = int(row.get("feature_fail", 0))
        except (ValueError, TypeError):
            ff = 0
        if ff != 0:
            assert bench in synt_lookup, (
                f"Missing synt row for benchmark (logic={logic}, seed={seed}, split={split}): {bench!r}"
            )
            s = synt_lookup[bench]
            raw_overhead = row.get("overhead", "")
            try:
                gin_overhead = float(raw_overhead) if raw_overhead not in ("", None) else 0.0
            except (TypeError, ValueError):
                gin_overhead = 0.0
            synt_overhead = synt_extraction_by_path.get(normalize_path(bench), 0.0)
            new_overhead = gin_overhead + synt_overhead
            synt_runtime = s["runtime"]
            total_runtime = synt_runtime + gin_overhead
            if total_runtime > timeout:
                solved = 0
                runtime = timeout
            else:
                solved = s["solved"]
                runtime = total_runtime
            out.append({
                "benchmark": bench,
                "selected": s["selected"],
                "solved": str(solved),
                "runtime": str(runtime),
                "solver_runtime": str(synt_runtime),
                "overhead": f"{new_overhead:.6f}",
                "feature_fail": "1",
            })
        else:
            out.append(row)
    return out


def write_gin_format_csv(path: Path, rows: list[dict]) -> None:
    """Write rows in GIN eval CSV format (same column order and header)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(GIN_HEADER)
        for row in rows:
            w.writerow([row.get(k, "") for k in GIN_HEADER])


def build_fallback_for_logic(
    logic: str,
    gin_dir: Path,
    synt_dir: Path,
    synt_extraction_base: Path,
    splits_dir: Path,
    out_dir: Path,
    timeout: float = 1200.0,
) -> None:
    """Build fallback CSVs and summary.json for one logic."""
    extraction_times_csv = synt_extraction_base / logic / "extraction_times.csv"
    if not extraction_times_csv.is_file():
        raise FileNotFoundError(f"Synt extraction times not found: {extraction_times_csv}")
    synt_extraction_by_path = load_extraction_times_csv(extraction_times_csv)

    seed_results = []
    for seed in SEEDS:
        gin_seed = gin_dir / logic / f"seed{seed}"
        synt_train = synt_dir / logic / f"seed{seed}" / "train_eval.csv"
        synt_test = synt_dir / logic / f"seed{seed}" / "test_eval.csv"
        train_json = splits_dir / logic / f"seed{seed}" / "train.json"
        test_json = splits_dir / logic / f"seed{seed}" / "test.json"

        if not gin_seed.is_dir():
            raise FileNotFoundError(f"GIN seed dir not found: {gin_seed}")
        for p in [synt_train, synt_test, train_json, test_json]:
            if not p.is_file():
                raise FileNotFoundError(f"Required file not found: {p}")

        gin_train_csv = gin_seed / "train_eval.csv"
        gin_test_csv = gin_seed / "test_eval.csv"
        if not gin_train_csv.is_file() or not gin_test_csv.is_file():
            raise FileNotFoundError(f"GIN eval CSVs not found under {gin_seed}")

        # Merge train
        gin_train_rows = load_gin_csv(gin_train_csv)
        synt_train_lookup = load_synt_lookup(synt_train)
        train_merged = merge_single_split(
            gin_train_rows,
            synt_train_lookup,
            synt_extraction_by_path,
            logic,
            seed,
            "train",
            timeout=timeout,
        )
        out_train = out_dir / logic / f"seed{seed}" / "train_eval.csv"
        write_gin_format_csv(out_train, train_merged)

        # Merge test
        gin_test_rows = load_gin_csv(gin_test_csv)
        synt_test_lookup = load_synt_lookup(synt_test)
        test_merged = merge_single_split(
            gin_test_rows,
            synt_test_lookup,
            synt_extraction_by_path,
            logic,
            seed,
            "test",
            timeout=timeout,
        )
        out_test = out_dir / logic / f"seed{seed}" / "test_eval.csv"
        write_gin_format_csv(out_test, test_merged)

        # Metrics: load perf JSONs (logic-relative keys), build SingleSolverDataset from fallback CSV
        multi_train = parse_performance_json(str(train_json), timeout)
        multi_test = parse_performance_json(str(test_json), timeout)
        result_train = parse_as_perf_csv(str(out_train), multi_train.get_timeout())
        result_test = parse_as_perf_csv(str(out_test), multi_test.get_timeout())

        train_metrics = compute_metrics(result_train, multi_train)
        test_metrics = compute_metrics(result_test, multi_test)

        seed_results.append({
            "seed": seed,
            "train_size": len(train_merged),
            "test_size": len(test_merged),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        })

    # Aggregate
    train_metrics_list = [r["train_metrics"] for r in seed_results]
    test_metrics_list = [r["test_metrics"] for r in seed_results]
    aggregated = {
        "train": {
            "gap_cls_solved_mean": float(np.mean([m["gap_cls_solved"] for m in train_metrics_list])),
            "gap_cls_solved_std": float(np.std([m["gap_cls_solved"] for m in train_metrics_list])),
            "gap_cls_par2_mean": float(np.mean([m["gap_cls_par2"] for m in train_metrics_list])),
            "gap_cls_par2_std": float(np.std([m["gap_cls_par2"] for m in train_metrics_list])),
        },
        "test": {
            "gap_cls_solved_mean": float(np.mean([m["gap_cls_solved"] for m in test_metrics_list])),
            "gap_cls_solved_std": float(np.std([m["gap_cls_solved"] for m in test_metrics_list])),
            "gap_cls_par2_mean": float(np.mean([m["gap_cls_par2"] for m in test_metrics_list])),
            "gap_cls_par2_std": float(np.std([m["gap_cls_par2"] for m in test_metrics_list])),
        },
    }

    def to_python(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_python(x) for x in obj]
        return obj

    results = {
        "division": logic,
        "n_seeds": len(SEEDS),
        "seed_values": SEEDS,
        "model_type": "graph+lite_fallback",
        "splits_dir": str(splits_dir / logic),
        "benchmark_root": str(DEFAULT_BENCHMARK_ROOT),
        "seeds": seed_results,
        "aggregated": aggregated,
    }
    summary_path = out_dir / logic / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(to_python(results), f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build lite_fallback: merge graph/before_fallback + lite for feature-fail instances."
    )
    parser.add_argument("--logic", required=True, help="Logic division (e.g. QF_IDL, ABV)")
    parser.add_argument(
        "--gin-dir",
        type=Path,
        default=GIN_PWC_DIR,
        help="GIN-PWC results root (default: data/cp26/results/graph/before_fallback)",
    )
    parser.add_argument(
        "--synt-dir",
        type=Path,
        default=SYNT_DIR,
        help="Lite (synt) results root (default: data/cp26/results/lite)",
    )
    parser.add_argument(
        "--synt-extraction-base",
        type=Path,
        default=SYNT_EXTRACTION_BASE,
        help="Synt extraction_times.csv base (default: data/features/syntactic)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_BASE,
        help="Performance splits root (default: data/cp26/performance_splits/smtcomp24)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_ROOT,
        help="Output root for lite_fallback (default: data/cp26/results/graph/lite_fallback)",
    )
    parser.add_argument("--timeout", type=float, default=1200.0, help="Timeout for PAR-2 (default: 1200)")
    args = parser.parse_args()

    build_fallback_for_logic(
        logic=args.logic,
        gin_dir=Path(args.gin_dir),
        synt_dir=Path(args.synt_dir),
        synt_extraction_base=Path(args.synt_extraction_base),
        splits_dir=Path(args.splits_dir),
        out_dir=Path(args.out_dir),
        timeout=args.timeout,
    )
    print(f"Done: {args.out_dir / args.logic}")


if __name__ == "__main__":
    main()
