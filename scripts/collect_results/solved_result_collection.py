#!/usr/bin/env python3
"""
Build a result summary CSV of test solved counts (mean ± std over seeds) per logic
from the same result directories as final_result_collection.

Output: doc/result_summary/final_solved.csv
"""

import argparse
import csv
import json
from pathlib import Path

# Script is under scripts/collect_results/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIRS = [
    PROJECT_ROOT / "data" / "cp26" / "results" / "lite",
    PROJECT_ROOT / "data" / "cp26" / "results" / "lite+text",
    PROJECT_ROOT / "data" / "cp26" / "results" / "graph",
    PROJECT_ROOT / "data" / "cp26" / "results" / "machsmt" / "ehm",
    PROJECT_ROOT / "data" / "cp26" / "results" / "graph+text",
    PROJECT_ROOT / "data" / "cp26" / "results" / "sibyl" / "evaluation",
]
LABELS = ["synt", "synt_mpnet", "gin_pwc", "mach_ehm", "fusion_pwc", "sibyl"]


def get_test_sbs_vbs_mean(summary_path: Path) -> tuple[float | None, float | None]:
    """Read summary.json and return (mean sbs_solved, mean vbs_solved) over test seeds, or (None, None) if missing."""
    if not summary_path.is_file():
        return None, None
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None
    seeds = data.get("seeds") or []
    sbs_vals, vbs_vals = [], []
    for s in seeds:
        tm = (s or {}).get("test_metrics") or {}
        if "sbs_solved" in tm:
            sbs_vals.append(float(tm["sbs_solved"]))
        if "vbs_solved" in tm:
            vbs_vals.append(float(tm["vbs_solved"]))
    if not sbs_vals or not vbs_vals:
        return None, None
    return sum(sbs_vals) / len(sbs_vals), sum(vbs_vals) / len(vbs_vals)


def get_test_solved_mean_std(summary_path: Path) -> tuple[float | None, float | None]:
    """Read summary.json and return (mean, std) of test solved count over seeds, or (None, None) if missing/invalid."""
    if not summary_path.is_file():
        return None, None
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None

    seeds = data.get("seeds") or []
    values = []
    for s in seeds:
        tm = (s or {}).get("test_metrics") or {}
        if "solved" in tm:
            values.append(float(tm["solved"]))
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = variance ** 0.5
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build result summary CSV: rows=logics, columns=test solved count per result dir."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "doc" / "result_summary" / "final_solved.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    result_dirs = [d.resolve() for d in RESULT_DIRS]
    for d in result_dirs:
        if not d.is_dir():
            raise SystemExit(f"Result directory not found: {d}")

    all_logics: set[str] = set()
    for res_dir in result_dirs:
        for p in res_dir.iterdir():
            if p.is_dir() and (p / "summary.json").is_file():
                all_logics.add(p.name)
    logics = sorted(all_logics)

    rows: list[dict[str, str | float]] = []
    for logic in logics:
        row: dict[str, str | float] = {"logic": logic}
        # SBS and VBS (same for all result dirs; use first available summary)
        sbs_mean, vbs_mean = None, None
        for res_dir in result_dirs:
            sbs_mean, vbs_mean = get_test_sbs_vbs_mean(res_dir / logic / "summary.json")
            if sbs_mean is not None:
                break
        row["sbs"] = round(sbs_mean, 1) if sbs_mean is not None else ""
        row["vbs"] = round(vbs_mean, 1) if vbs_mean is not None else ""
        for res_dir, label in zip(result_dirs, LABELS):
            summary_path = res_dir / logic / "summary.json"
            mean, std = get_test_solved_mean_std(summary_path)
            row[label] = round(mean, 1) if mean is not None else ""
            row[f"{label}_std"] = round(std, 1) if std is not None else ""
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["logic", "sbs", "vbs"] + [item for label in LABELS for item in (label, f"{label}_std")]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} logics to {args.output}")


if __name__ == "__main__":
    main()
