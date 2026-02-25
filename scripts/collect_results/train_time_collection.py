#!/usr/bin/env python3
"""
Collect training time from train logs in two locations:
- GIN-PWC: data/cp26/results/gnn/gin_pwc/<logic>/train_log/seed*.log
- Syntactic: data/cp26/results/synt/<logic>/train_log/seed*.log

For each logic, parses each seed log: first and last timestamp determine duration in seconds.
Adds total feature extraction time from data/features/syntactic/<logic>/extraction_times.csv to both gin_pwc and synt (capped at 5s per instance).
Writes one CSV: doc/result_summary/train_time.csv with columns logic, gin_pwc, synt.
"""

import csv
import re
from datetime import datetime
from pathlib import Path

# Script is under scripts/collect_results/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GIN_PWC_ROOT = PROJECT_ROOT / "data" / "cp26" / "results" / "gnn" / "gin_pwc"
SYNT_ROOT = PROJECT_ROOT / "data" / "cp26" / "results" / "synt"
FEATURES_SYNTACTIC_ROOT = PROJECT_ROOT / "data" / "features" / "syntactic"
OUTPUT_PATH = PROJECT_ROOT / "doc" / "result_summary" / "train_time.csv"

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


def collect_from_root(root: Path) -> dict[str, float | str]:
    """Collect {logic: train_time_sec} from a root dir with <logic>/train_log/seed*.log. Empty string if no duration."""
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

    result: dict[str, float | str] = {}
    for logic in logics:
        train_log_dir = root / logic / "train_log"
        seed_logs = sorted(train_log_dir.glob("seed*.log"))
        durations: list[float] = []
        for log_path in seed_logs:
            d = train_duration_sec(log_path)
            if d is not None:
                durations.append(d)
        result[logic] = round(sum(durations) / len(durations), 1) if durations else ""
    return result


EXTRACTION_TIME_CAP_SEC = 5.0


def total_extraction_time_sec(logic: str) -> float:
    """Sum time_sec from data/features/syntactic/<logic>/extraction_times.csv, capping each instance at 5s. Return 0 if missing."""
    path = FEATURES_SYNTACTIC_ROOT / logic / "extraction_times.csv"
    if not path.is_file():
        return 0.0
    total = 0.0
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    total += min(float(row["time_sec"]), EXTRACTION_TIME_CAP_SEC)
                except (ValueError, KeyError):
                    pass
    except OSError:
        return 0.0
    return total


def main() -> None:
    gin_pwc_times: dict[str, float | str] = {}
    synt_times: dict[str, float | str] = {}

    if GIN_PWC_ROOT.is_dir():
        gin_pwc_times = collect_from_root(GIN_PWC_ROOT)
        # Add feature extraction time to each gin_pwc training time
        for logic in gin_pwc_times:
            val = gin_pwc_times[logic]
            if val != "":
                extraction_sec = total_extraction_time_sec(logic)
                gin_pwc_times[logic] = round(float(val) + extraction_sec, 1)
    else:
        print(f"Skipping GIN-PWC: directory not found: {GIN_PWC_ROOT}")

    if SYNT_ROOT.is_dir():
        synt_times = collect_from_root(SYNT_ROOT)
        # Add feature extraction time to each synt training time
        for logic in synt_times:
            val = synt_times[logic]
            if val != "":
                extraction_sec = total_extraction_time_sec(logic)
                synt_times[logic] = round(float(val) + extraction_sec, 1)
    else:
        print(f"Skipping synt: directory not found: {SYNT_ROOT}")

    all_logics = sorted(set(gin_pwc_times) | set(synt_times))
    if all_logics:
        rows = [
            {
                "logic": logic,
                "gin_pwc": gin_pwc_times.get(logic, ""),
                "synt": synt_times.get(logic, ""),
            }
            for logic in all_logics
        ]
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["logic", "gin_pwc", "synt"]
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
