#!/usr/bin/env python3
"""
Convert doc/logic_filter/smtcomp24_logic_info.csv into a LaTeX table.

Reads the CSV and prints a tabular environment with escaped logic names
and formatted numeric columns.
"""

import argparse
import csv
from pathlib import Path

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


def format_rate(val: str | float) -> str:
    """Format description_rate for LaTeX (e.g. 0.986 -> 98.6). Unit in column header."""
    if val == "" or val is None:
        return "---"
    try:
        x = float(val)
        return f"{x * 100:.1f}"
    except (TypeError, ValueError):
        return str(val)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert smtcomp24_logic_info.csv into a LaTeX table."
    )
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "doc" / "logic_filter" / "smtcomp24_logic_info.csv",
        help="Input CSV path (default: doc/logic_filter/smtcomp24_logic_info.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "doc" / "cp26" / "smtcomp24_logic_info.tex",
        help="Output .tex path (default: doc/cp26/smtcomp24_logic_info.tex)",
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

    # Exclude columns we don't want in the table
    exclude = {"num_families"}
    columns = [k for k in fieldnames if k not in exclude]

    # Column headers for LaTeX (short, escaped)
    header_map = {
        "logic": "Logic",
        "num_benchmarks": "Benchmarks",
        "num_solvers": "Solvers",
        "sbs": "SBS solved",
        "vbs": "VBS solved",
        "description_rate": "Desc.\\%",
        "ave_size": "Avg. file size (KB)",
    }
    headers = [header_map.get(k, k.replace("_", " ").title()) for k in columns]

    # Build tabular spec: l for logic, c for the rest
    ncols = len(columns)
    col_spec = "l" + "c" * (ncols - 1)

    lines = [
        "\\begin{table}[htbp]",
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
            elif key == "description_rate":
                cells.append(format_rate(val))
            elif key == "ave_size":
                cells.append(str(val) if val != "" else "---")
            else:
                # integers
                cells.append(str(val) if val != "" else "---")
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Meta-information for the selected SMT-COMP 2024 logics: benchmark count, number of competitionsolvers, SBS and VBS solved counts, description rate, and average file size.}",
        "\\label{tab:logic-info}",
        "\\end{table}",
    ])

    out = "\n".join(lines) + "\n"
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
