#!/usr/bin/env python3
"""
Build doc/logic_filter/smtcomp24_logic_info.csv from meta_info and performance data.

For each logic:
- num_benchmarks, num_families: from data/meta_info_24/<logic>.json if present,
  else num_benchmarks from performance instance count and num_families=0.
- num_solvers, sbs, vbs: from data/cp26/raw_data/smtcomp24_performance/<logic>.json.
- description_rate: fraction of benchmarks with a description. ave_size: average SMT2 file size in KB.

Output: logic,num_benchmarks,num_families,num_solvers,sbs,vbs,description_rate,ave_size
"""

import argparse
import csv
import json
from pathlib import Path

# Add project root for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.performance import parse_performance_json


def load_meta_info(meta_dir: Path, logic: str) -> tuple[int, int, int, int] | None:
    """Load meta_info JSON for a logic. Returns (num_benchmarks, num_families, num_with_description, total_size) or None."""
    path = meta_dir / f"{logic}.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return None
    families = set()
    with_desc = 0
    total_size = 0
    for item in data:
        if isinstance(item, dict):
            if "family" in item:
                families.add(item["family"])
            desc = item.get("description")
            if desc is not None and isinstance(desc, str) and desc.strip():
                with_desc += 1
            s = item.get("size")
            if isinstance(s, (int, float)) and s >= 0:
                total_size += int(s)
    return len(data), len(families), with_desc, total_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build smtcomp24_logic_info.csv from meta_info and performance JSONs."
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "meta_info_24",
        help="Directory containing <logic>.json meta-info files (default: data/meta_info_24)",
    )
    parser.add_argument(
        "--perf-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cp26" / "raw_data" / "smtcomp24_performance",
        help="Directory containing <logic>.json performance files (default: data/cp26/raw_data/smtcomp24_performance)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "doc" / "logic_filter" / "smtcomp24_logic_info.csv",
        help="Output CSV path (default: doc/logic_filter/smtcomp24_logic_info.csv)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="PAR-2 timeout in seconds (default: 1200)",
    )
    args = parser.parse_args()

    perf_dir = args.perf_dir.resolve()
    if not perf_dir.is_dir():
        raise SystemExit(f"Performance directory not found: {perf_dir}")

    meta_dir = args.meta_dir.resolve()
    rows: list[dict[str, int | str | float]] = []

    for perf_path in sorted(perf_dir.glob("*.json")):
        logic = perf_path.stem
        try:
            multi = parse_performance_json(str(perf_path), args.timeout)
        except (ValueError, OSError) as e:
            print(f"Warning: skip {perf_path.name}: {e}", file=sys.stderr)
            continue

        num_solvers = multi.num_solvers()
        sbs_ds = multi.get_best_solver_dataset()
        vbs_ds = multi.get_virtual_best_solver_dataset()
        sbs = sbs_ds.get_solved_count()
        vbs = vbs_ds.get_solved_count()
        num_benchmarks_perf = len(multi)

        meta = load_meta_info(meta_dir, logic)
        if meta is not None:
            num_benchmarks, num_families, num_with_desc, total_size = meta
            rate = num_with_desc / num_benchmarks if num_benchmarks else 0.0
            description_rate = round(rate, 3)
            ave_size = round(total_size / num_benchmarks / 1024, 1) if num_benchmarks else 0  # KB
        else:
            num_benchmarks = num_benchmarks_perf
            num_families = 0
            description_rate = ""
            ave_size = ""

        rows.append({
            "logic": logic,
            "num_benchmarks": num_benchmarks,
            "num_families": num_families,
            "num_solvers": num_solvers,
            "sbs": sbs,
            "vbs": vbs,
            "description_rate": description_rate,
            "ave_size": ave_size,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["logic", "num_benchmarks", "num_families", "num_solvers", "sbs", "vbs", "description_rate", "ave_size"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} logics to {args.output}")


if __name__ == "__main__":
    main()
