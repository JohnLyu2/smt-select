#!/usr/bin/env python3
"""
Collect graph build/fail counts from GIN-PWC train logs.

Reads logs from data/results/graph/gin_pwc/<logic>/train_log/seed*.log,
parses lines like: "Graphs: 1865 built, 0 failed (of 1865 instances)",
and writes doc/result_summary/feature_fail.csv with one row per logic: logic, avg_fail_rate (average fail rate over seeds).
"""

import csv
import re
from pathlib import Path

# Script is under scripts/collect_results/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GIN_PWC_ROOT = PROJECT_ROOT / "data" / "results" / "graph" / "gin_pwc"
OUTPUT_PATH = PROJECT_ROOT / "doc" / "result_summary" / "feature_fail.csv"

# "2026-02-22 23:03:30,273 - INFO - Graphs: 1865 built, 0 failed (of 1865 instances)"
GRAPHS_LINE_RE = re.compile(
    r"Graphs: (\d+) built, (\d+) failed \(of (\d+) instances\)"
)


def parse_seed_from_filename(log_path: Path) -> int | None:
    """Extract seed number from seed<N>.log. Returns None if not matched."""
    m = re.match(r"seed(\d+)\.log", log_path.name)
    return int(m.group(1)) if m else None


def collect_from_log(log_path: Path) -> tuple[int, int, int] | None:
    """
    Read log and return (built, failed, total) from the Graphs line, or None if not found.
    """
    if not log_path.is_file():
        return None
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                m = GRAPHS_LINE_RE.search(line)
                if m:
                    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except OSError:
        return None
    return None


def main() -> None:
    if not GIN_PWC_ROOT.is_dir():
        raise FileNotFoundError(f"GIN-PWC results directory not found: {GIN_PWC_ROOT}")

    rows: list[dict[str, str | float]] = []
    for logic_dir in sorted(GIN_PWC_ROOT.iterdir()):
        if not logic_dir.is_dir():
            continue
        train_log_dir = logic_dir / "train_log"
        if not train_log_dir.is_dir():
            continue
        logic = logic_dir.name
        fail_rates: list[float] = []
        for log_path in sorted(train_log_dir.glob("seed*.log")):
            if parse_seed_from_filename(log_path) is None:
                continue
            data = collect_from_log(log_path)
            if data is None:
                continue
            _built, failed, total = data
            rate = failed / total if total else 0.0
            fail_rates.append(rate)
        if not fail_rates:
            continue
        avg_fail_rate = round(sum(fail_rates) / len(fail_rates), 4)
        rows.append({"logic": logic, "avg_fail_rate": avg_fail_rate})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["logic", "avg_fail_rate"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
