#!/usr/bin/env python3
"""
Generate doc/cp26/final_par2.tex directly from result summary.json files.

Reads summary.json from each result directory, extracts test PAR-2 gap closed
(mean ± std over seeds), and writes a LaTeX table.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIRS = [
    PROJECT_ROOT / "data" / "cp26" / "results" / "lite",
    PROJECT_ROOT / "data" / "cp26" / "results" / "lite+text",
    PROJECT_ROOT / "data" / "cp26" / "results" / "graph",
    PROJECT_ROOT / "data" / "cp26" / "results" / "machsmt" / "ehm",
    PROJECT_ROOT / "data" / "cp26" / "results" / "graph+text",
    PROJECT_ROOT / "data" / "cp26" / "results" / "sibyl" / "evaluation",
]
LABELS = ["synt", "synt_mpnet", "gin_pwc", "mach_ehm", "fusion_pwc", "sibyl"]

TEX_PATH = PROJECT_ROOT / "doc" / "cp26" / "final_par2.tex"

DISPLAY_ORDER = ["mach_ehm", "sibyl", "synt", "gin_pwc", "synt_mpnet", "fusion_pwc"]
VARIANT_MAP = {
    "synt": "Lite",
    "gin_pwc": "Graph",
    "synt_mpnet": "Lite+Text",
    "fusion_pwc": "Graph+Text",
    "sibyl": "",
}


def latex_escape(s: str) -> str:
    for old, new in [
        ("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
        ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"),
        ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}"),
    ]:
        s = s.replace(old, new)
    return s


def get_test_gap_par2(summary_path: Path) -> tuple[float | None, float | None]:
    """Return (mean, std) of test gap_cls_par2 from summary.json, or (None, None)."""
    if not summary_path.is_file():
        return None, None
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None
    agg = data.get("aggregated", {}).get("test", {})
    if "gap_cls_par2_mean" in agg and "gap_cls_par2_std" in agg:
        return float(agg["gap_cls_par2_mean"]), float(agg["gap_cls_par2_std"])
    seeds = data.get("seeds") or []
    values = [
        float(s["test_metrics"]["gap_cls_par2"])
        for s in seeds
        if s and "test_metrics" in s and "gap_cls_par2" in s.get("test_metrics", {})
    ]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
    return mean, std


def format_cell(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "---"
    mean_str = f"{mean * 100:.1f}"
    if std is not None:
        return f"{mean_str} $\\pm$ {std * 100:.1f}"
    return mean_str


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

    # Collect data: {(logic, label): (mean, std)}
    data: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for res_dir, label in zip(result_dirs, LABELS):
        for logic in logics:
            data[(logic, label)] = get_test_gap_par2(res_dir / logic / "summary.json")

    # Determine which labels have data
    value_columns = [c for c in DISPLAY_ORDER if any(data.get((l, c), (None, None))[0] is not None for l in logics)]

    has_sibyl = "sibyl" in value_columns
    n_without = 4 if has_sibyl else 3
    n_with = len(value_columns) - n_without
    ncols = 1 + len(value_columns)
    col_spec = "l" + "c" * len(value_columns)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f" & \\multicolumn{{{n_without}}}{{c}}{{\\textbf{{Without Description}}}} & \\multicolumn{{{n_with}}}{{c}}{{\\textbf{{With Description}}}} \\\\",
        f"\\cmidrule(lr){{2-{1 + n_without}}}",
        f"\\cmidrule(lr){{{2 + n_without}-{ncols}}}",
    ]

    if has_sibyl:
        lines.append(
            " & \\multirow{2}{*}{MachSMT} & \\multirow{2}{*}{Sibyl} & "
            f"\\multicolumn{{2}}{{c}}{{SMT-Select}} & \\multicolumn{{{n_with}}}{{c}}{{SMT-Select}} \\\\"
        )
        lines.append("\\cmidrule(lr){4-5}")
        lines.append(f"\\cmidrule(lr){{{2 + n_without}-{ncols}}}")
    else:
        lines.append(
            " & \\multirow{2}{*}{MachSMT} & \\multicolumn{2}{c}{SMT-Select} & "
            f"\\multicolumn{{{n_with}}}{{c}}{{SMT-Select}} \\\\"
        )
        lines.append("\\cmidrule(lr){3-4}")
        lines.append(f"\\cmidrule(lr){{{2 + n_without}-{ncols}}}")

    row3_cells = [""] + [VARIANT_MAP.get(k, "") for k in value_columns]
    lines.append(" & ".join(row3_cells) + " \\\\")
    lines.append("\\midrule")

    for logic in logics:
        row_vals: dict[str, float] = {}
        for label in value_columns:
            mean, _ = data.get((logic, label), (None, None))
            if mean is not None:
                row_vals[label] = mean
        max_val = max(row_vals.values()) if row_vals else None
        if max_val is not None:
            threshold = max_val - 0.005
            best_keys = {k for k, v in row_vals.items() if v >= threshold}
        else:
            best_keys = set()

        cells = [latex_escape(logic)]
        for label in value_columns:
            mean, std = data.get((logic, label), (None, None))
            cell = format_cell(mean, std)
            if label in best_keys:
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
        "\\caption{Experimental results on the held-out test set, measured by the PAR-2 SBS--VBS gap closed (\\%). Results are averaged over five random train--test splits and reported as mean $\\pm$ standard deviation.}",
        "\\label{tab:final_par2}",
        "\\end{table}",
    ])

    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
