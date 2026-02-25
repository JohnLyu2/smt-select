#!/usr/bin/env python3
"""
Convert doc/result_summary/train_time.csv into a LaTeX table (doc/cp26/train_time.tex).

Reads the CSV with columns logic, gin_pwc, synt (train time in seconds per source).
Writes a tabular .tex with escaped logic names and training time in minutes.
"""

import csv
from pathlib import Path

# Script is under scripts/latex/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "doc" / "result_summary" / "train_time.csv"
OUTPUT_TEX = PROJECT_ROOT / "doc" / "cp26" / "train_time.tex"


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
    """Format train time (CSV in seconds) as minutes for LaTeX; empty -> ---; <= 0.1 min -> $\\leq 0.1$."""
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
    if not INPUT_CSV.is_file():
        raise SystemExit(f"File not found: {INPUT_CSV}")

    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    columns = ["logic", "synt", "gin_pwc"]
    headers = ["", "SMT-Select-Lite", "SMT-Select-Graph"]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    for row in rows:
        cells = [
            latex_escape(row.get("logic", "")),
            format_time_min(row.get("synt", "")),
            format_time_min(row.get("gin_pwc", "")),
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
