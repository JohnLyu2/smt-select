#!/usr/bin/env python3
"""
Generate doc/cp26/solved_variability.tex from result summary.json files.

Reads summary.json from each SMT-Select result directory, extracts test solved count
(mean ± std over seeds), and writes a LaTeX table. No MachSMT or Sibyl columns.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SMT-Select variants only (no MachSMT, Sibyl)
RESULT_DIRS = [
    PROJECT_ROOT / "data" / "results" / "lite",
    PROJECT_ROOT / "data" / "results" / "graph",
    PROJECT_ROOT / "data" / "results" / "text" / "all-mpnet-base-v2",
    PROJECT_ROOT / "data" / "results" / "lite+text" / "all-mpnet-base-v2",
    PROJECT_ROOT / "data" / "results" / "graph+text" / "all-mpnet-base-v2",
]
LABELS = ["synt", "gin_pwc", "text_mpnet", "synt_mpnet", "fusion_pwc"]

TEX_PATH = PROJECT_ROOT / "doc" / "cp26" / "solved_variability.tex"

DISPLAY_ORDER = ["synt", "text_mpnet", "synt_mpnet", "gin_pwc", "fusion_pwc"]
VARIANT_MAP = {
    "synt": "Lite",
    "gin_pwc": "Graph",
    "text_mpnet": "Text",
    "synt_mpnet": "Lite+Text",
    "fusion_pwc": "Graph+Text",
}


def latex_escape(s: str) -> str:
    for old, new in [
        ("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
        ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"),
        ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}"),
    ]:
        s = s.replace(old, new)
    return s


def get_test_solved(summary_path: Path) -> tuple[float | None, float | None]:
    """Return (mean, std) of test solved count from summary.json, or (None, None)."""
    if not summary_path.is_file():
        return None, None
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None
    seeds = data.get("seeds") or []
    values = [
        float(s["test_metrics"]["solved"])
        for s in seeds
        if s and "test_metrics" in s and "solved" in s.get("test_metrics", {})
    ]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = variance ** 0.5
    return mean, std


def format_cell(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "---"
    return f"{mean:.1f} $\\pm$ {std:.1f}"


def main() -> None:
    result_dirs = [d.resolve() for d in RESULT_DIRS]
    for d in result_dirs:
        if not d.is_dir():
            raise SystemExit(f"Result directory not found: {d}")

    all_logics: set[str] = set()
    for res_dir in result_dirs:
        for p in res_dir.iterdir():
            if p.is_dir() and (p / "summary.json").is_file():
                all_logics.add(p.name)
    logics = sorted(all_logics)

    # Collect data: {(logic, label): (mean, std)} for test solved count
    data: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for res_dir, label in zip(result_dirs, LABELS):
        for logic in logics:
            data[(logic, label)] = get_test_solved(res_dir / logic / "summary.json")

    # Columns that have at least one value
    value_columns = [c for c in DISPLAY_ORDER if any(data.get((l, c), (None, None))[0] is not None for l in logics)]
    col_spec = "@{}l" + "c" * len(value_columns) + "@{}"

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular*}}{{\\columnwidth}}{{@{{\\extracolsep{{\\fill}}}}{col_spec}}}",
        "\\toprule",
    ]
    header_cells = [""] + [VARIANT_MAP.get(k, "") for k in value_columns]
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\midrule")

    for logic in logics:
        row_vals: dict[str, float] = {}
        for label in value_columns:
            mean, _ = data.get((logic, label), (None, None))
            if mean is not None:
                row_vals[label] = mean
        # Higher solved count is better
        max_val = max(row_vals.values()) if row_vals else None
        if max_val is not None:
            threshold = max_val - 0.01
            best_keys = {k for k, v in row_vals.items() if v >= threshold}
        else:
            best_keys = set()

        cells = [latex_escape(logic)]
        for label in value_columns:
            mean, std = data.get((logic, label), (None, None))
            cell = format_cell(mean, std)
            if label in best_keys:
                cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular*}",
        "\\caption{Test set number of instances solved by SMT-Select variant. Mean $\\pm$ std over five random train--test splits.}",
        "\\label{tab:solved_variability}",
        "\\end{table}",
    ])

    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
