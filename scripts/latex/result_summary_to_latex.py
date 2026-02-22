#!/usr/bin/env python3
"""
Convert doc/result_summary/final_res.csv into a LaTeX table (doc/cp26/final_res.tex).

Reads the result summary CSV (logics × PAR2 gap closed columns) and writes
a tabular .tex file with escaped logic names and formatted numeric cells.
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


def format_par2_gap(val: str | float, std_val: str | float | None = None) -> str:
    """Format PAR2 gap closed (0–1) as percentage; if std_val given, format as mean ± std (one decimal)."""
    if val == "" or val is None:
        return "---"
    try:
        x = float(val)
        mean_str = f"{x * 100:.1f}"
        if std_val not in ("", None):
            try:
                s = float(std_val)
                return f"{mean_str} $\\pm$ {s * 100:.1f}"
            except (TypeError, ValueError):
                pass
        return mean_str
    except (TypeError, ValueError):
        return str(val)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert result summary CSV (PAR2 gap closed) into a LaTeX table."
    )
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "doc" / "result_summary" / "final_res.csv",
        help="Input CSV path (default: doc/result_summary/final_res.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "doc" / "cp26" / "final_res.tex",
        help="Output .tex path (default: doc/cp26/final_res.tex)",
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

    # Display columns: logic plus value columns (exclude *_std; those are paired with value columns)
    value_columns = [k for k in fieldnames if k != "logic" and not k.endswith("_std")]
    columns = ["logic"] + value_columns

    header_map = {
        "logic": "Logic",
        "synt": "SMT-Select-Lite",
        "synt_mpnet": "SMT-Select-Lite-Text",
    }
    headers = [header_map.get(k, k.replace("_", "+").title()) for k in columns]

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
                std_val = row.get(f"{key}_std", "")
                cells.append(format_par2_gap(val, std_val))
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Experimental results on the held-out test set, measured by the PAR-2 SBS–VBS gap closed (\\%). Results are averaged over five random train–test splits and reported as mean $\\pm$ standard deviation.}",
        "\\label{tab:final_res}",
        "\\end{table}",
    ])

    out = "\n".join(lines) + "\n"
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
