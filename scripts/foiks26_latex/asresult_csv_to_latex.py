#!/usr/bin/env python3
"""
Convert the foiks26 CV summary CSV to a LaTeX table.
"""

import argparse
import csv
import sys
from pathlib import Path


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    text = str(text)
    text = text.replace("\\", "\\textbackslash{}")
    text = text.replace("&", "\\&")
    text = text.replace("%", "\\%")
    text = text.replace("$", "\\$")
    text = text.replace("#", "\\#")
    text = text.replace("^", "\\textasciicircum{}")
    text = text.replace("_", "\\_")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    text = text.replace("~", "\\textasciitilde{}")
    return text


def format_header(header_text: str) -> str:
    """Map CSV headers to display names."""
    header_map = {
        "logic": "Logic",
        "desc_test": "Desc.",
        "machsmt_test": "MachSMT",
        "syn+desc_test": "Synt.+Desc.",
        "synt_test": "Synt.",
    }
    return header_map.get(header_text.lower(), escape_latex(header_text))


def parse_mean_std(cell: str) -> tuple[float, float] | None:
    """Parse 'mean ± std' and return numeric values."""
    cell = cell.strip()
    if not cell or cell.upper() == "N/A":
        return None
    if "±" in cell:
        left, right = cell.split("±", maxsplit=1)
        return float(left.strip()), float(right.strip())
    if "+/-" in cell:
        left, right = cell.split("+/-", maxsplit=1)
        return float(left.strip()), float(right.strip())
    return None


def format_cell(mean_std: tuple[float, float]) -> tuple[str, str]:
    """Format mean/std strings for math mode (no $ delimiters)."""
    mean, std = mean_std
    return f"{mean:.1f}", f"{std:.1f}"


def csv_to_latex(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    tabular_only: bool = True,
) -> str:
    """
    Convert the foiks26 summary CSV to a LaTeX table.

    Args:
        csv_path: Path to input CSV file
        output_path: Output LaTeX file path
        caption: Optional table caption
        label: Optional table label
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Error: File {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    if output_path is None:
        csv_name = csv_path.stem
        default_dir = Path("data/foiks26/result_summary")
        default_dir.mkdir(parents=True, exist_ok=True)
        output_path = default_dir / f"{csv_name}.tex"
    else:
        output_path = Path(output_path)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("Error: CSV file is empty.", file=sys.stderr)
        sys.exit(1)

    header = rows[0]
    data_rows = [row for row in rows[1:] if row and any(cell.strip() for cell in row)]

    header_index = {h.lower(): i for i, h in enumerate(header)}
    desired_order = [
        "logic",
        "machsmt_test",
        "synt_test",
        "desc_test",
        "syn+desc_test",
    ]
    if all(key in header_index for key in desired_order):
        order_indices = [header_index[key] for key in desired_order]
        header = [header[i] for i in order_indices]
        data_rows = [
            [row[i] if i < len(row) else "" for i in order_indices] for row in data_rows
        ]

    num_cols = len(header)
    latex_lines: list[str] = []
    if not tabular_only:
        latex_lines.append("\\begin{table}[t]")
        latex_lines.append("\\centering")
    latex_lines.append("\\setlength{\\tabcolsep}{12pt}")
    col_spec = "l" + "c" * (num_cols - 1)
    latex_lines.append(
        f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{col_spec}@{{}}}}"
    )
    latex_lines.append("\\toprule")

    smt_select_cols = []
    for i, h in enumerate(header):
        if h.lower() in ["synt_test", "desc_test", "syn+desc_test"]:
            smt_select_cols.append(i)

    if smt_select_cols:
        first_smt_col = min(smt_select_cols)
        num_smt_cols = len(smt_select_cols)

        first_row_cells = []
        for i, cell in enumerate(header):
            if i < first_smt_col:
                formatted_cell = format_header(cell)
                if i == 0:
                    first_row_cells.append(f"\\multirow{{2}}{{*}}{{{formatted_cell}}}")
                else:
                    first_row_cells.append(
                        f"\\multirow{{2}}{{*}}{{\\makecell{{{formatted_cell} \\\\ (Synt.)}}}}"
                    )
            elif i == first_smt_col:
                first_row_cells.append(
                    f"\\multicolumn{{{num_smt_cols}}}{{c}}{{SMT-Select}}"
                )
        latex_lines.append(" & ".join(first_row_cells) + " \\\\")
        latex_lines.append(
            "\\cmidrule(lr){"
            + str(first_smt_col + 1)
            + "-"
            + str(first_smt_col + num_smt_cols)
            + "}"
        )

        second_row_cells = []
        for i, cell in enumerate(header):
            formatted_cell = format_header(cell)
            if i == 0 or i < first_smt_col:
                second_row_cells.append("")
            else:
                second_row_cells.append(f"{formatted_cell}")
        latex_lines.append(" & ".join(second_row_cells) + " \\\\")
    else:
        header_cells = []
        for i, cell in enumerate(header):
            formatted_cell = format_header(cell)
            align = "l" if i == 0 else "c"
            header_cells.append(f"\\multicolumn{{1}}{{{align}}}{{{formatted_cell}}}")
        latex_lines.append(" & ".join(header_cells) + " \\\\")
    latex_lines.append("\\midrule")

    for row in data_rows:
        while len(row) < num_cols:
            row.append("")

        numeric_means: list[float | None] = []
        for i, cell in enumerate(row):
            if i == 0:
                numeric_means.append(None)
                continue
            parsed = parse_mean_std(cell)
            numeric_means.append(parsed[0] if parsed is not None else None)

        valid_values = [(i, v) for i, v in enumerate(numeric_means) if v is not None]
        sorted_values = sorted(valid_values, key=lambda x: x[1], reverse=True)
        highest_idx = sorted_values[0][0] if len(sorted_values) > 0 else None

        cells: list[str] = []
        for i, cell in enumerate(row):
            cell = cell.strip()
            if i == 0:
                cells.append(escape_latex(cell.upper()))
                continue

            parsed = parse_mean_std(cell)
            if parsed is None:
                cells.append("--")
                continue

            mean_str, std_str = format_cell(parsed)
            if i == highest_idx:
                cells.append(f"$\\boldsymbol{{{mean_str}}} \\pm {std_str}$")
            else:
                cells.append(f"${mean_str} \\pm {std_str}$")

        latex_lines.append(" & ".join(cells) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular*}")
    if not tabular_only:
        latex_lines.append("\\end{table}")

    latex_output = "\n".join(latex_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_output, encoding="utf-8")
    print(f"LaTeX table written to {output_path}")
    return latex_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert foiks26 summary CSV file to LaTeX table format"
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="data/foiks26/cv_results/summary.csv",
        help="Path to input CSV file (default: data/foiks26/cv_results/summary.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output LaTeX file (default: data/foiks26/result_summary/{csv_name}.tex)",
    )
    parser.add_argument(
        "--with-table",
        action="store_true",
        help="Include the table environment (caption/label allowed).",
    )

    args = parser.parse_args()
    csv_to_latex(
        args.csv_file,
        output_path=args.output,
        tabular_only=not args.with_table,
    )


if __name__ == "__main__":
    main()
