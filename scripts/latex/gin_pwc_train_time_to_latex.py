#!/usr/bin/env python3
"""
Convert doc/result_summary/gin_pwc_train_time.csv into a LaTeX table (doc/cp26/train_time.tex).

Reads the CSV with logic and train_time_sec (mean over seeds), writes a tabular .tex with
escaped logic names and training time in seconds.
"""

import argparse
import csv
from pathlib import Path

# Script is under scripts/latex/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    """Format train time (CSV in seconds) as minutes for LaTeX; empty -> ---."""
    if val == "" or val is None:
        return "---"
    try:
        sec = float(val)
        return f"{sec / 60:.1f}"
    except (TypeError, ValueError):
        return str(val)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GIN-PWC train time CSV into a LaTeX table."
    )
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "doc" / "result_summary" / "gin_pwc_train_time.csv",
        help="Input CSV path (default: doc/result_summary/gin_pwc_train_time.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "doc" / "cp26" / "train_time.tex",
        help="Output .tex path (default: doc/cp26/train_time.tex)",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Omit the tabular header row",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"File not found: {csv_path}")

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    columns = [c for c in fieldnames if c]
    header_map = {"logic": "", "train_time_sec": "SMT-Select"}
    headers = [header_map.get(k, k.replace("_", " ").title()) for k in columns]

    ncols = len(columns)
    col_spec = "l" + "c" * (ncols - 1)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
    ]

    if not args.no_header:
        lines.append(" & ".join(headers) + " \\\\")
        lines.append("\\midrule")

    for row in rows:
        cells = []
        for key in columns:
            val = row.get(key, "")
            if key == "logic":
                cells.append(latex_escape(val))
            else:
                cells.append(format_time_min(val))
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Average training time in minutes.}",
        "\\label{tab:train_time}",
        "\\end{table}",
    ])

    out = "\n".join(lines) + "\n"
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
