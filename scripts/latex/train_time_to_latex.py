#!/usr/bin/env python3
"""
Build doc/cp26/train_time.tex from train logs in result directories (no intermediate CSV).

Reads train logs from:
- Lite: data/cp26/results/lite/<logic>/train_log/seed*.log
- Lite+Text: data/cp26/results/lite+text/all-mpnet-base-v2/<logic>/train_log/seed*.log
- Text: data/cp26/results/text/all-mpnet-base-v2/<logic>/train_log/seed*.log
- Graph: data/cp26/results/graph/<logic>/train_log/seed*.log
- Graph+Text: data/cp26/results/graph+text/all-mpnet-base-v2/<logic>/train_log/seed*.log

For each logic, first/last timestamp per seed log gives duration. Lite/Lite+Text include
syntactic (and desc) extraction time; Text = log duration + desc extraction; Graph/Graph+Text
include GNN + fallback times. Writes a LaTeX table with times in minutes.
"""

import csv
import re
from datetime import datetime
from pathlib import Path

# Script is under scripts/latex/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LITE_ROOT = PROJECT_ROOT / "data" / "cp26" / "results" / "lite"
LITE_TEXT_ROOT = PROJECT_ROOT / "data" / "cp26" / "results" / "lite+text" / "all-mpnet-base-v2"
TEXT_ROOT = PROJECT_ROOT / "data" / "cp26" / "results" / "text" / "all-mpnet-base-v2"
GRAPH_ROOT = PROJECT_ROOT / "data" / "cp26" / "results" / "graph"
GRAPH_TEXT_ROOT = PROJECT_ROOT / "data" / "cp26" / "results" / "graph+text" / "all-mpnet-base-v2"
FEATURES_SYNTACTIC_ROOT = PROJECT_ROOT / "data" / "features" / "syntactic"
FEATURES_DESC_ROOT = PROJECT_ROOT / "data" / "features" / "desc" / "all-mpnet-base-v2"
OUTPUT_TEX = PROJECT_ROOT / "doc" / "cp26" / "train_time.tex"

LOG_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) ")
EXTRACTION_TIME_CAP_SEC = 5.0
NUM_PARALLEL = 8


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


def _sum_extraction_csv(csv_path: Path) -> float:
    """Sum time_sec from an extraction_times.csv, capping each instance at EXTRACTION_TIME_CAP_SEC. Return 0 if missing."""
    if not csv_path.is_file():
        return 0.0
    total = 0.0
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    total += min(float(row["time_sec"]), EXTRACTION_TIME_CAP_SEC)
                except (ValueError, KeyError):
                    pass
    except OSError:
        return 0.0
    return total / NUM_PARALLEL


def total_syntactic_extraction_sec(logic: str) -> float:
    """Syntactic feature extraction time for a logic."""
    return _sum_extraction_csv(FEATURES_SYNTACTIC_ROOT / logic / "extraction_times.csv")


def total_desc_extraction_sec(logic: str) -> float:
    """Description embedding extraction time for a logic."""
    return _sum_extraction_csv(FEATURES_DESC_ROOT / logic / "extraction_times.csv")


def latex_escape(s: str) -> str:
    """Escape characters that are special in LaTeX."""
    s = str(s)
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("$", "\\$")
    s = s.replace("#", "\\#")
    s = s.replace("_", "\\_")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s


def format_time_min(val: str | float) -> str:
    """Format train time (seconds) as minutes for LaTeX; empty -> ---; <= 0.1 min -> $\\leq 0.1$."""
    if val == "" or val is None:
        return "---"
    try:
        sec = float(val)
        min_val = sec / 60
        if round(min_val, 1) <= 0.1:
            return "$\\leq 0.1$"
        return f"{min_val:.1f}"
    except (TypeError, ValueError):
        return str(val)


def main() -> None:
    lite_times: dict[str, float | str] = {}
    lite_text_times: dict[str, float | str] = {}
    text_times: dict[str, float | str] = {}
    graph_raw: dict[str, float | str] = {}
    graph_times: dict[str, float | str] = {}
    graph_text_times: dict[str, float | str] = {}

    if LITE_ROOT.is_dir():
        lite_times = collect_from_root(LITE_ROOT)
        for logic in lite_times:
            val = lite_times[logic]
            if val != "":
                lite_times[logic] = round(float(val) + total_syntactic_extraction_sec(logic), 1)
    else:
        print(f"Skipping Lite: directory not found: {LITE_ROOT}")

    if LITE_TEXT_ROOT.is_dir():
        lite_text_times = collect_from_root(LITE_TEXT_ROOT)
        for logic in lite_text_times:
            val = lite_text_times[logic]
            if val != "":
                lite_text_times[logic] = round(
                    float(val) + total_syntactic_extraction_sec(logic) + total_desc_extraction_sec(logic), 1
                )
    else:
        print(f"Skipping Lite+Text: directory not found: {LITE_TEXT_ROOT}")

    if TEXT_ROOT.is_dir():
        text_times = collect_from_root(TEXT_ROOT)
        for logic in text_times:
            val = text_times[logic]
            if val != "":
                text_times[logic] = round(float(val) + total_desc_extraction_sec(logic), 1)
    else:
        print(f"Skipping Text: directory not found: {TEXT_ROOT}")

    if GRAPH_ROOT.is_dir():
        graph_raw = collect_from_root(GRAPH_ROOT)
        for logic in graph_raw:
            val = graph_raw[logic]
            if val != "":
                lite_val = lite_times.get(logic, "")
                lite_total = float(lite_val) if lite_val != "" else 0.0
                graph_times[logic] = round(float(val) + lite_total, 1)
    else:
        print(f"Skipping Graph: directory not found: {GRAPH_ROOT}")

    if GRAPH_TEXT_ROOT.is_dir():
        graph_text_raw = collect_from_root(GRAPH_TEXT_ROOT)
        for logic in graph_text_raw:
            val = graph_text_raw[logic]
            if val != "":
                gnn_raw = graph_raw.get(logic, "")
                gnn_dur = float(gnn_raw) if gnn_raw != "" else 0.0
                lite_text_val = lite_text_times.get(logic, "")
                lite_text_total = float(lite_text_val) if lite_text_val != "" else 0.0
                desc_ext = total_desc_extraction_sec(logic)
                graph_text_times[logic] = round(
                    desc_ext + gnn_dur + float(val) + lite_text_total, 1
                )
    else:
        print(f"Skipping Graph+Text: directory not found: {GRAPH_TEXT_ROOT}")

    all_logics = sorted(
        set(lite_times) | set(lite_text_times) | set(text_times) | set(graph_times) | set(graph_text_times)
    )
    if not all_logics:
        raise SystemExit("No logics found in any result directory.")

    rows = [
        {
            "logic": logic,
            "lite": lite_times.get(logic, ""),
            "lite_text": lite_text_times.get(logic, ""),
            "text": text_times.get(logic, ""),
            "graph": graph_times.get(logic, ""),
            "graph_text": graph_text_times.get(logic, ""),
        }
        for logic in all_logics
    ]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        " & \\multicolumn{5}{c}{SMT-Select} \\\\",
        "\\cmidrule(lr){2-6}",
        " & Lite & Text & Lite+Text & Graph & Graph+Text \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [
            latex_escape(row["logic"]),
            format_time_min(row["lite"]),
            format_time_min(row["text"]),
            format_time_min(row["lite_text"]),
            format_time_min(row["graph"]),
            format_time_min(row["graph_text"]),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Average training time in minutes.}",
        "\\label{tab:train_time}",
        "\\end{table}",
    ])

    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
