#!/usr/bin/env python3
"""
Convert CSV file to LaTeX table format.
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


def format_number(value):
    """Format numbers appropriately for LaTeX."""
    if isinstance(value, float):
        # Format floats to remove unnecessary decimals
        if value == int(value):
            return str(int(value))
        # Keep one decimal place for readability
        return f"{value:.1f}"
    return str(value)


def format_header(header_text, max_length=10):
    """
    Format header text, using makecell for line breaks if too long.

    Args:
        header_text: The header text
        max_length: Maximum length before wrapping (default: 10, more aggressive)

    Returns:
        Formatted header with makecell if needed
    """
    escaped = escape_latex(header_text)

    # Special handling for common patterns
    # "#SBS Solved" or "#VBS Solved" -> "#SBS \\ Solved" or "#VBS \\ Solved"
    if header_text.startswith("#") and "Solved" in header_text:
        parts = header_text.split(" ", 1)
        if len(parts) == 2:
            return (
                f"\\makecell{{{escape_latex(parts[0])} \\\\ {escape_latex(parts[1])}}}"
            )

    # "Desc. Rate (%)" -> "Desc. Rate \\ (%)"
    if "Rate" in header_text and "(" in header_text:
        # Break before the parentheses for better readability
        idx = header_text.find("(")
        if idx > 0:
            return f"\\makecell{{{escape_latex(header_text[:idx].strip())} \\\\ {escape_latex(header_text[idx:])}}}"

    # "Ave. Desc. Length" -> "Ave. Desc. \\ Length"
    if "Ave." in header_text and "Length" in header_text:
        # Break after "Desc."
        if "Desc." in header_text:
            idx = header_text.find("Desc.") + len("Desc.")
            if idx < len(header_text):
                return f"\\makecell{{{escape_latex(header_text[:idx].strip())} \\\\ {escape_latex(header_text[idx:].strip())}}}"

    # If header is short, return as is
    if len(header_text) <= max_length:
        return escaped

    # For longer headers, try to break at natural points
    # Look for common break points: spaces, periods, parentheses
    if " " in header_text:
        # Try to break at space
        parts = header_text.split(" ", 1)
        if len(parts[0]) <= max_length:
            return (
                f"\\makecell{{{escape_latex(parts[0])} \\\\ {escape_latex(parts[1])}}}"
            )

    # If no good break point, use \makecell with manual break
    # Break roughly in the middle
    mid = len(header_text) // 2
    # Find nearest space
    for i in range(mid - 3, mid + 3):
        if i > 0 and i < len(header_text) and header_text[i] == " ":
            return f"\\makecell{{{escape_latex(header_text[:i])} \\\\ {escape_latex(header_text[i + 1 :])}}}"

    # If no space found, just return as is (LaTeX will handle it)
    return escaped


def csv_to_latex(csv_path, output_path=None, caption=None, label=None):
    """
    Convert CSV file to LaTeX table.

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output LaTeX file (if None, defaults to data/folks26/logic_filtering/{csv_name}.tex)
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
        # Default to data/folks26/logic_filtering directory
        default_dir = Path("data/folks26/logic_filtering")
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
    data_rows = rows[1:]

    # Determine number of columns
    num_cols = len(header)

    # Start building LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    # Add makecell package comment (user needs to add \usepackage{makecell} in preamble)
    latex_lines.append(
        "% Note: Add \\usepackage{makecell} to your preamble for line breaks in headers"
    )
    # Increase column separation to make columns wider
    latex_lines.append("\\setlength{\\tabcolsep}{12pt}")

    # Create column specification (left-aligned for first column, centered for others)
    # Use tabular* with \textwidth to make table wider and distribute space
    col_spec = "l" + "c" * (num_cols - 1)
    # Build tabular* command: @{\extracolsep{\fill}} adds space between columns
    tabular_star = f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{col_spec}@{{}}}}"
    latex_lines.append(tabular_star)
    latex_lines.append("\\toprule")

    # Add header row with centered headers using \multicolumn
    # This ensures headers align well with their column contents
    # Use \makecell for long headers to allow line breaks
    header_cells = []
    for i, cell in enumerate(header):
        formatted_cell = format_header(cell)
        if i == 0:
            # First column: left-aligned
            header_cells.append(f"\\multicolumn{{1}}{{l}}{{{formatted_cell}}}")
        else:
            # Other columns: centered headers over right-aligned data
            header_cells.append(f"\\multicolumn{{1}}{{c}}{{{formatted_cell}}}")
    latex_lines.append(" & ".join(header_cells) + " \\\\")
    latex_lines.append("\\midrule")

    # Add data rows
    for row in data_rows:
        # Pad row if necessary
        while len(row) < num_cols:
            row.append("")

        # Format cells
        cells = []
        for i, cell in enumerate(row):
            cell = cell.strip()
            if i == 0:
                # First column (Logic) - convert to uppercase
                cells.append(escape_latex(cell.upper()))
            else:
                # Numeric columns - format numbers
                try:
                    # Check if this is "Desc. Rate (%)" or "Ave. Desc. Length" column
                    header_name = header[i] if i < len(header) else ""
                    is_decimal_column = (
                        "Desc. Rate" in header_name
                        or "Ave. Desc. Length" in header_name
                    )

                    if "." in cell:
                        num = float(cell)
                        if is_decimal_column:
                            # Always show one decimal place for Desc. Rate and Ave. Desc. Length
                            cells.append(f"{num:.1f}")
                        else:
                            cells.append(format_number(num))
                    else:
                        num = int(cell)
                        if is_decimal_column:
                            # Always show one decimal place for Desc. Rate and Ave. Desc. Length
                            cells.append(f"{num:.1f}")
                        else:
                            cells.append(str(num))
                except ValueError:
                    # Not a number, just escape
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

    # Write to file (output_path is always set now, either by user or default)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_output)
    print(f"LaTeX table written to {output_path}")

    return latex_output


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV file to LaTeX table format"
    )
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output LaTeX file (default: data/folks26/logic_filtering/{csv_name}.tex)",
    )
    parser.add_argument("-c", "--caption", help="Table caption")
    parser.add_argument("-l", "--label", help="Table label for LaTeX referencing")

    args = parser.parse_args()

    csv_to_latex(
        args.csv_file, output_path=args.output, caption=args.caption, label=args.label
    )


if __name__ == "__main__":
    main()
