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
CSV_PATH = PROJECT_ROOT / "doc" / "result_summary" / "final_res.csv"
TEX_PATH = PROJECT_ROOT / "doc" / "cp26" / "final_res.tex"


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

    # Display order: Without Description (MachSMT, Sibyl, Lite, Graph), then With Description (Lite+Text, Graph+Text)
    display_order = ["mach_ehm", "sibyl", "synt", "gin_pwc", "synt_mpnet", "fusion_pwc"]
    value_columns = [c for c in display_order if c in fieldnames]
    columns = ["logic"] + value_columns

    # Header blocks: Without Description (first 3 or 4 cols), With Description (rest)
    variant_map = {
        "synt": "Lite",
        "gin_pwc": "Graph",
        "synt_mpnet": "Lite+Text",
        "fusion_pwc": "Graph+Text",
        "sibyl": "",
    }
    has_sibyl = "sibyl" in value_columns
    n_without = 4 if has_sibyl else 3  # MachSMT, (optional Sibyl), Lite, Graph
    n_with = len(value_columns) - n_without
    first_block_columns = value_columns[:n_without] if len(value_columns) >= n_without else value_columns

    ncols = len(columns)
    col_spec = "l" + "c" * (ncols - 1)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
    ]

    if not args.no_header:
        # Row 1: Without Description (first n_without cols), With Description (n_with cols)
        lines.append(" & \\multicolumn{" + str(n_without) + "}{c}{\\textbf{Without Description}} & \\multicolumn{" + str(n_with) + "}{c}{\\textbf{With Description}} \\\\")
        lines.append("\\cmidrule(lr){2-" + str(1 + n_without) + "}")
        lines.append("\\cmidrule(lr){" + str(2 + n_without) + "-" + str(ncols) + "}")
        # Row 2: MachSMT (multirow 2), optional Sibyl (own column), then SMT-Select blocks
        if has_sibyl:
            # Columns: logic | MachSMT | Sibyl | Lite | Graph | (With-description SMT-Select...)
            lines.append(
                " & \\multirow{2}{*}{MachSMT} & \\multirow{2}{*}{Sibyl} & "
                "\\multicolumn{2}{c}{SMT-Select} & \\multicolumn{" + str(n_with) + "}{c}{SMT-Select} \\\\"
            )
            lines.append("\\cmidrule(lr){4-5}")
            lines.append("\\cmidrule(lr){" + str(2 + n_without) + "-" + str(ncols) + "}")
        else:
            # Original layout: MachSMT, then SMT-Select blocks
            lines.append(
                " & \\multirow{2}{*}{MachSMT} & \\multicolumn{2}{c}{SMT-Select} & "
                "\\multicolumn{" + str(n_with) + "}{c}{SMT-Select} \\\\"
            )
            lines.append("\\cmidrule(lr){3-4}")
            lines.append("\\cmidrule(lr){" + str(2 + n_without) + "-" + str(ncols) + "}")
        # Row 3: variant names (empty for logic, empty under MachSMT, then Lite, Graph, Graph+Text, Lite+Text)
        row3_cells = [""] + [variant_map.get(k, "") for k in value_columns]
        lines.append(" & ".join(row3_cells) + " \\\\")
        lines.append("\\midrule")

    for row in rows:
        # Find highest mean in first block for bolding
        first_block_vals: dict[str, float] = {}
        for key in first_block_columns:
            val = row.get(key, "")
            if val != "" and val is not None:
                try:
                    first_block_vals[key] = float(val)
                except (TypeError, ValueError):
                    pass
        max_mean = max(first_block_vals.values()) if first_block_vals else None
        max_keys = {k for k, v in first_block_vals.items() if v == max_mean} if max_mean is not None else set()

        cells = []
        for key in columns:
            val = row.get(key, "")
            if key == "logic":
                cells.append(latex_escape(val))
            else:
                std_val = row.get(f"{key}_std", "")
                cell = format_par2_gap(val, std_val)
                if key in max_keys:
                    # Bold only the mean, not the std
                    if " $\\pm$ " in cell:
                        mean_part, _, std_part = cell.partition(" $\\pm$ ")
                        cell = "\\textbf{" + mean_part + "} $\\pm$ " + std_part
                    else:
                        cell = "\\textbf{" + cell + "}"
                cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Experimental results on the held-out test set, measured by the PAR-2 SBS–VBS gap closed (\\%). Results are averaged over five random train–test splits and reported as mean $\\pm$ standard deviation. Bold values indicate the best result within each feature setting.}",
        "\\label{tab:final_res}",
        "\\end{table}",
    ])

    out = "\n".join(lines) + "\n"
    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text(out, encoding="utf-8")
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
