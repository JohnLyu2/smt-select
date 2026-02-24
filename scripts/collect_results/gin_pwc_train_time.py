#!/usr/bin/env python3
"""
Collect GIN-PWC training time from train logs (e.g. data/cp26/results/gnn/gin_pwc/<logic>/train_log/seed*.log).

For each logic, parses each seed log: first and last timestamp determine duration in seconds.
Outputs CSV with logic, train_time_sec (mean over seeds). Default output: doc/result_summary/gin_pwc_train_time.csv.
"""

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

# Script is under scripts/collect_results/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Log lines start with: 2026-02-20 10:29:44,327 - INFO - ...
LOG_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) ")


def parse_log_timestamp(line: str) -> float | None:
    """Parse a log line timestamp to seconds since epoch. Returns None if line doesn't match."""
    m = LOG_TIMESTAMP_RE.match(line.strip())
    if not m:
        return None
    datepart, msec = m.group(1), m.group(2)
    try:
        dt = datetime.strptime(datepart, "%Y-%m-%d %H:%M:%S")
        return dt.timestamp() + int(msec) / 1000.0
    except ValueError:
        return None


def train_duration_sec(log_path: Path) -> float | None:
    """Return training duration in seconds from first and last timestamp in the log, or None."""
    first_ts: float | None = None
    last_ts: float | None = None
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                ts = parse_log_timestamp(line)
                if ts is not None:
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts
    except OSError:
        return None
    if first_ts is not None and last_ts is not None:
        return last_ts - first_ts
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect GIN-PWC training time from train_log/seed*.log, average over seeds."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cp26" / "results" / "gnn" / "gin_pwc",
        help="Root dir containing <logic>/train_log/ (default: data/cp26/results/gnn/gin_pwc)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "doc" / "result_summary" / "gin_pwc_train_time.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    root = args.input_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Input directory not found: {root}")

    # Find all logic dirs that have a train_log with at least one seed*.log
    logics: list[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        train_log_dir = p / "train_log"
        if not train_log_dir.is_dir():
            continue
        seed_logs = sorted(train_log_dir.glob("seed*.log"))
        if seed_logs:
            logics.append(p.name)

    rows: list[dict[str, str | float]] = []
    for logic in logics:
        train_log_dir = root / logic / "train_log"
        seed_logs = sorted(train_log_dir.glob("seed*.log"))
        durations: list[float] = []
        for log_path in seed_logs:
            d = train_duration_sec(log_path)
            if d is not None:
                durations.append(d)
        if durations:
            mean_sec = sum(durations) / len(durations)
            rows.append({"logic": logic, "train_time_sec": round(mean_sec, 1)})
        else:
            rows.append({"logic": logic, "train_time_sec": ""})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["logic", "train_time_sec"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} logics to {args.output}")


if __name__ == "__main__":
    main()
