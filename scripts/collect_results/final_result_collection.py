#!/usr/bin/env python3
"""
Build a result summary CSV in doc/result_summary with one row per logic and PAR2 gap closed
(test, mean ± std over seeds) from selected result directories.

Default result dirs:
- data/cp26/results/synt
- data/cp26/results/synt+smtlib_desc/synt+mpnet

Output columns: logic, <name1>, <name1>_std, <name2>, <name2>_std, ... Missing logic/result pairs are left empty.
"""

import argparse
import csv
import json
from pathlib import Path

# Script is under scripts/collect_results/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_test_gap_cls_par2_mean_std(summary_path: Path) -> tuple[float | None, float | None]:
    """Read summary.json and return (mean, std) of test gap_cls_par2 over seeds, or (None, None) if missing/invalid."""
    if not summary_path.is_file():
        return None, None
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None

    agg = data.get("aggregated") or {}
    test_agg = agg.get("test") or {}
    if "gap_cls_par2_mean" in test_agg and "gap_cls_par2_std" in test_agg:
        return float(test_agg["gap_cls_par2_mean"]), float(test_agg["gap_cls_par2_std"])

    seeds = data.get("seeds") or []
    values = []
    for s in seeds:
        tm = (s or {}).get("test_metrics") or {}
        if "gap_cls_par2" in tm:
            values.append(float(tm["gap_cls_par2"]))
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = variance ** 0.5
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build result summary CSV: rows=logics, columns=PAR2 gap closed per result dir."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "doc" / "result_summary" / "final_res.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "result_dirs",
        nargs="*",
        type=Path,
        default=[
            PROJECT_ROOT / "data" / "cp26" / "results" / "synt",
            PROJECT_ROOT / "data" / "cp26" / "results" / "synt+smtlib_desc" / "synt+mpnet",
        ],
        help="Result directories (each contains <logic>/summary.json). Default: synt, synt+smtlib_desc/synt+mpnet",
    )
    args = parser.parse_args()

    result_dirs = [Path(p).resolve() for p in args.result_dirs]
    for d in result_dirs:
        if not d.is_dir():
            raise SystemExit(f"Result directory not found: {d}")

    # Collect all logics from any result dir
    all_logics: set[str] = set()
    for res_dir in result_dirs:
        for p in res_dir.iterdir():
            if p.is_dir() and (p / "summary.json").is_file():
                all_logics.add(p.name)
    logics = sorted(all_logics)

    # Column labels: short names for default dirs
    default_names = ["synt", "synt_mpnet"]
    if len(result_dirs) == 2 and default_names[0] in str(result_dirs[0]) and "mpnet" in str(result_dirs[1]):
        labels = default_names
    else:
        labels = [d.name for d in result_dirs]

    rows: list[dict[str, str | float]] = []
    for logic in logics:
        row: dict[str, str | float] = {"logic": logic}
        for res_dir, label in zip(result_dirs, labels):
            summary_path = res_dir / logic / "summary.json"
            mean, std = get_test_gap_cls_par2_mean_std(summary_path)
            row[label] = round(mean, 4) if mean is not None else ""
            row[f"{label}_std"] = round(std, 4) if std is not None else ""
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["logic"] + [item for label in labels for item in (label, f"{label}_std")]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} logics to {args.output}")


if __name__ == "__main__":
    main()
