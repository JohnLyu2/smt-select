#!/usr/bin/env python3
"""
Convert result summary CSV file to LaTeX table format.
"""

import csv
import sys
import argparse
from pathlib import Path


def escape_latex(text):
    """Escape special LaTeX characters."""
    if isinstance(text, (int, float)):
        text = str(text)
    text = str(text)
    # Escape special characters
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


def format_header(header_text):
    """
    Format header text, converting to uppercase for display.
    """
    # Map column names to display names
    header_map = {
        "logic": "Logic",
        "machsmt": "MachSMT",
        "syn": "Syntactic",
        "des": "Desc.",
        "des_exp": "Des\\_Exp",
        "des_syn": "Syntactic+Desc.",
    }
    return header_map.get(header_text.lower(), escape_latex(header_text))


def csv_to_latex(csv_path, output_path=None, caption=None, label=None):
    """
    Convert CSV file to LaTeX table.

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output LaTeX file (if None, defaults to data/folks26/result_summary/{csv_name}.tex)
        caption: Optional table caption
        label: Optional table label for referencing
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Error: File {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    # Set default output path if not provided
    if output_path is None:
        # Get the CSV filename and change extension to .tex
        csv_name = csv_path.stem
        # Default to data/folks26/result_summary directory
        default_dir = Path("data/folks26/result_summary")
        default_dir.mkdir(parents=True, exist_ok=True)
        output_path = default_dir / f"{csv_name}.tex"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("Error: CSV file is empty.", file=sys.stderr)
        sys.exit(1)

    # Separate header and data
    header = rows[0]
    data_rows = [row for row in rows[1:] if row and any(cell.strip() for cell in row)]

    # Find and exclude des_exp column
    exclude_cols = set()
    for i, h in enumerate(header):
        if h.lower() == "des_exp":
            exclude_cols.add(i)

    # Filter out excluded columns from header and data
    header = [h for i, h in enumerate(header) if i not in exclude_cols]
    data_rows = [
        [cell for i, cell in enumerate(row) if i not in exclude_cols]
        for row in data_rows
    ]

    # Determine number of columns
    num_cols = len(header)

    # Start building LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering")
    # Increase column separation to make columns wider
    latex_lines.append("\\setlength{\\tabcolsep}{12pt}")

    # Create column specification (left-aligned for first column, centered for others)
    col_spec = "l" + "c" * (num_cols - 1)
    # Use tabular* with \textwidth
    tabular_star = f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{col_spec}@{{}}}}"
    latex_lines.append(tabular_star)
    latex_lines.append("\\toprule")

    # Find SMT-Select columns (syn, des, des_syn)
    smt_select_cols = []
    for i, h in enumerate(header):
        if h.lower() in ["syn", "des", "des_syn"]:
            smt_select_cols.append(i)

    # Add header row with SMT-Select spanning multiple columns
    # First row: Logic, MachSMT, and SMT-Select spanning syn/des/des_syn
    if smt_select_cols:
        first_smt_col = min(smt_select_cols)
        num_smt_cols = len(smt_select_cols)

        # Build first header row with multicolumn for SMT-Select
        # Use \multirow with [c] option for vertical centering
        first_row_cells = []
        for i, cell in enumerate(header):
            if i < first_smt_col:
                formatted_cell = format_header(cell)
                if i == 0:
                    # Logic spans 2 rows vertically centered
                    first_row_cells.append(f"\\multirow{{2}}{{*}}{{{formatted_cell}}}")
                else:
                    # MachSMT with (Syntactic) below, spans 2 rows
                    first_row_cells.append(
                        f"\\multirow{{2}}{{*}}{{\\makecell{{{formatted_cell} \\\\ (Syntactic)}}}}"
                    )
            elif i == first_smt_col:
                # Add SMT-Select spanning all SMT-Select columns
                first_row_cells.append(
                    f"\\multicolumn{{{num_smt_cols}}}{{c}}{{SMT-Select}}"
                )
            # Skip other SMT-Select columns (they're covered by multicolumn)
        latex_lines.append(" & ".join(first_row_cells) + " \\\\")
        latex_lines.append(
            "\\cmidrule(lr){"
            + str(first_smt_col + 1)
            + "-"
            + str(first_smt_col + num_smt_cols)
            + "}"
        )

        # Build second header row with individual column names
        second_row_cells = []
        for i, cell in enumerate(header):
            formatted_cell = format_header(cell)
            if i == 0:
                second_row_cells.append("")  # Empty for Logic (multirow from above)
            elif i < first_smt_col:
                second_row_cells.append("")  # Empty for MachSMT (multirow from above)
            else:
                second_row_cells.append(f"{formatted_cell}")
        latex_lines.append(" & ".join(second_row_cells) + " \\\\")
    else:
        # Fallback: no SMT-Select columns, use simple header
        header_cells = []
        for i, cell in enumerate(header):
            formatted_cell = format_header(cell)
            if i == 0:
                header_cells.append(f"\\multicolumn{{1}}{{l}}{{{formatted_cell}}}")
            else:
                header_cells.append(f"\\multicolumn{{1}}{{c}}{{{formatted_cell}}}")
        latex_lines.append(" & ".join(header_cells) + " \\\\")
    latex_lines.append("\\midrule")

    # Add data rows
    for row in data_rows:
        # Pad row if necessary
        while len(row) < num_cols:
            row.append("")

        # Parse numeric values for finding highest and second highest
        numeric_values = []
        for i, cell in enumerate(row):
            if i == 0:
                numeric_values.append(None)  # Skip first column
            else:
                cell = cell.strip()
                if cell == "":
                    numeric_values.append(None)
                else:
                    try:
                        numeric_values.append(float(cell))
                    except ValueError:
                        numeric_values.append(None)

        # Find highest and second highest values (excluding None)
        valid_values = [(i, v) for i, v in enumerate(numeric_values) if v is not None]
        sorted_values = sorted(valid_values, key=lambda x: x[1], reverse=True)
        highest_idx = sorted_values[0][0] if len(sorted_values) > 0 else None
        second_highest_idx = sorted_values[1][0] if len(sorted_values) > 1 else None

        # Format cells
        cells = []
        for i, cell in enumerate(row):
            cell = cell.strip()
            if i == 0:
                # First column (Logic) - convert to uppercase
                cells.append(escape_latex(cell.upper()))
            else:
                # Numeric columns - format numbers with one decimal place
                if cell == "":
                    cells.append("--")  # Empty cell placeholder
                else:
                    try:
                        num = float(cell)
                        formatted = f"{num:.1f}"
                        if i == highest_idx:
                            # Bold the highest
                            cells.append(f"\\textbf{{{formatted}}}")
                        elif i == second_highest_idx:
                            # Underline the second highest
                            cells.append(f"\\underline{{{formatted}}}")
                        else:
                            cells.append(formatted)
                    except ValueError:
                        cells.append(escape_latex(cell))

        latex_lines.append(" & ".join(cells) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular*}")

    # Add caption and label if provided
    if caption:
        latex_lines.append(f"\\caption{{{escape_latex(caption)}}}")
    if label:
        latex_lines.append(f"\\label{{{label}}}")

    latex_lines.append("\\end{table}")

    # Join all lines
    latex_output = "\n".join(latex_lines)

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_output)
    print(f"LaTeX table written to {output_path}")

    return latex_output


def main():
    parser = argparse.ArgumentParser(
        description="Convert result summary CSV file to LaTeX table format"
    )
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output LaTeX file (default: data/folks26/result_summary/{csv_name}.tex)",
    )
    parser.add_argument("-c", "--caption", help="Table caption")
    parser.add_argument("-l", "--label", help="Table label for LaTeX referencing")

    args = parser.parse_args()

    csv_to_latex(
        args.csv_file, output_path=args.output, caption=args.caption, label=args.label
    )


if __name__ == "__main__":
    main()
