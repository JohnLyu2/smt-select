#!/usr/bin/env python3
"""
Convert doc/result_summary/final_solved.csv into a LaTeX table (doc/cp26/final_solved.tex).

Reads the solved-count summary CSV and writes a tabular .tex file with the same
layout as the PAR2 gap table (logics × methods, mean ± std of test solved count).
"""

import argparse
import csv
from pathlib import Path

# Script is under scripts/latex/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "doc" / "result_summary" / "final_solved.csv"
TEX_PATH = PROJECT_ROOT / "doc" / "cp26" / "final_solved.tex"


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


def format_solved(val: str | float) -> str:
    """Format solved count as mean with one decimal."""
    if val == "" or val is None:
        return "---"
    try:
        x = float(val)
        return f"{x:.1f}"
    except (TypeError, ValueError):
        return str(val)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert solved-count summary CSV into a LaTeX table."
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Omit the tabular header row",
    )
    args = parser.parse_args()

    if not CSV_PATH.is_file():
        raise SystemExit(f"File not found: {CSV_PATH}")

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    display_order = ["sbs", "vbs", "mach_ehm", "sibyl", "synt", "gin_pwc", "synt_mpnet", "fusion_pwc"]
    value_columns = [c for c in display_order if c in fieldnames]
    columns = ["logic"] + value_columns

    variant_map = {
        "sbs": "",
        "vbs": "",
        "synt": "Lite",
        "gin_pwc": "Graph",
        "synt_mpnet": "Lite+Text",
        "fusion_pwc": "Graph+Text",
        "sibyl": "",
    }
    has_sibyl = "sibyl" in value_columns
    n_baseline = 2 if ("sbs" in value_columns and "vbs" in value_columns) else 0
    n_without = 4 if has_sibyl else 3
    n_with = 2
    # Bold best in row (all columns except VBS)
    columns_to_bold = [c for c in value_columns if c != "vbs"]

    ncols = len(columns)
    col_spec = "l" + "c" * (ncols - 1)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
    ]

    if not args.no_header:
        if n_baseline == 2:
            lines.append(
                " & \\multicolumn{2}{c}{\\textbf{Reference}} & \\multicolumn{" + str(n_without) + "}{c}{\\textbf{Without Description}} & \\multicolumn{" + str(n_with) + "}{c}{\\textbf{With Description}} \\\\"
            )
            lines.append("\\cmidrule(lr){2-3}")
            lines.append("\\cmidrule(lr){4-" + str(3 + n_without) + "}")
            lines.append("\\cmidrule(lr){" + str(4 + n_without) + "-" + str(ncols) + "}")
        else:
            lines.append(" & \\multicolumn{" + str(n_without) + "}{c}{\\textbf{Without Description}} & \\multicolumn{" + str(n_with) + "}{c}{\\textbf{With Description}} \\\\")
            lines.append("\\cmidrule(lr){2-" + str(1 + n_without) + "}")
            lines.append("\\cmidrule(lr){" + str(2 + n_without) + "-" + str(ncols) + "}")
        if n_baseline == 2 and has_sibyl:
            lines.append(
                " & \\multirow{2}{*}{SBS} & \\multirow{2}{*}{VBS} & "
                "\\multirow{2}{*}{MachSMT} & \\multirow{2}{*}{Sibyl} & "
                "\\multicolumn{2}{c}{SMT-Select} & \\multicolumn{" + str(n_with) + "}{c}{SMT-Select} \\\\"
            )
            lines.append("\\cmidrule(lr){6-7}")
            lines.append("\\cmidrule(lr){8-" + str(ncols) + "}")
        elif n_baseline == 2:
            lines.append(
                " & \\multirow{2}{*}{SBS} & \\multirow{2}{*}{VBS} & \\multirow{2}{*}{MachSMT} & "
                "\\multicolumn{2}{c}{SMT-Select} & \\multicolumn{" + str(n_with) + "}{c}{SMT-Select} \\\\"
            )
            lines.append("\\cmidrule(lr){5-6}")
            lines.append("\\cmidrule(lr){7-" + str(ncols) + "}")
        elif has_sibyl:
            lines.append(
                " & \\multirow{2}{*}{MachSMT} & \\multirow{2}{*}{Sibyl} & "
                "\\multicolumn{2}{c}{SMT-Select} & \\multicolumn{" + str(n_with) + "}{c}{SMT-Select} \\\\"
            )
            lines.append("\\cmidrule(lr){4-5}")
            lines.append("\\cmidrule(lr){6-" + str(ncols) + "}")
        else:
            lines.append(
                " & \\multirow{2}{*}{MachSMT} & \\multicolumn{2}{c}{SMT-Select} & "
                "\\multicolumn{" + str(n_with) + "}{c}{SMT-Select} \\\\"
            )
            lines.append("\\cmidrule(lr){3-4}")
            lines.append("\\cmidrule(lr){5-" + str(ncols) + "}")
        row3_cells = [""] + [variant_map.get(k, "") for k in value_columns]
        lines.append(" & ".join(row3_cells) + " \\\\")
        lines.append("\\midrule")

    for row in rows:
        # Bold best in row (all columns except VBS)
        row_vals: dict[str, float] = {}
        for key in columns_to_bold:
            val = row.get(key, "")
            if val != "" and val is not None:
                try:
                    row_vals[key] = float(val)
                except (TypeError, ValueError):
                    pass
        max_val = max(row_vals.values()) if row_vals else None
        max_keys = {k for k, v in row_vals.items() if v == max_val} if max_val is not None else set()

        cells = []
        for key in columns:
            val = row.get(key, "")
            if key == "logic":
                cells.append(latex_escape(val))
            elif key == "vbs":
                cell = format_solved(val)
                cells.append(cell)
            else:
                cell = format_solved(val)
                if key in max_keys:
                    cell = "\\textbf{" + cell + "}"
                cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Number of instances solved on the held-out test set. Results are averaged over five random train–test splits.}",
        "\\label{tab:final_solved}",
        "\\end{table}",
    ])

    out = "\n".join(lines) + "\n"
    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text(out, encoding="utf-8")
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
