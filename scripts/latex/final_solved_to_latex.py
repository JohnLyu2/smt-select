#!/usr/bin/env python3
"""
Generate doc/cp26/final_solved.tex directly from result summary.json files.

Reads summary.json from each result directory, extracts test solved counts
(mean over seeds) plus SBS/VBS reference, and writes a LaTeX table.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIRS = [
    PROJECT_ROOT / "data" / "cp26" / "results" / "lite",
    PROJECT_ROOT / "data" / "cp26" / "results" / "lite+text" / "all-mpnet-base-v2",
    PROJECT_ROOT / "data" / "cp26" / "results" / "graph",
    PROJECT_ROOT / "data" / "cp26" / "results" / "machsmt" / "ehm",
    PROJECT_ROOT / "data" / "cp26" / "results" / "graph+text",
    PROJECT_ROOT / "data" / "cp26" / "results" / "sibyl" / "evaluation",
    PROJECT_ROOT / "data" / "cp26" / "results" / "text" / "all-mpnet-base-v2",
]
LABELS = ["synt", "synt_mpnet", "gin_pwc", "mach_ehm", "fusion_pwc", "sibyl", "text_mpnet"]

TEX_PATH = PROJECT_ROOT / "doc" / "cp26" / "final_solved.tex"

DISPLAY_ORDER = ["vbs", "mach_ehm", "sibyl", "synt", "gin_pwc", "text_mpnet", "synt_mpnet", "fusion_pwc"]
VARIANT_MAP = {
    "vbs": "",
    "synt": "Lite",
    "gin_pwc": "Graph",
    "text_mpnet": "Text",
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


def get_test_solved_mean(summary_path: Path) -> float | None:
    """Return mean test solved count from summary.json, or None."""
    if not summary_path.is_file():
        return None
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    seeds = data.get("seeds") or []
    values = [
        float(s["test_metrics"]["solved"])
        for s in seeds
        if s and "test_metrics" in s and "solved" in s.get("test_metrics", {})
    ]
    return sum(values) / len(values) if values else None


def get_test_sbs_vbs_mean(summary_path: Path) -> tuple[float | None, float | None]:
    """Return (mean sbs_solved, mean vbs_solved) from summary.json, or (None, None)."""
    if not summary_path.is_file():
        return None, None
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None
    seeds = data.get("seeds") or []
    sbs_vals = [float(s["test_metrics"]["sbs_solved"]) for s in seeds if s and "sbs_solved" in s.get("test_metrics", {})]
    vbs_vals = [float(s["test_metrics"]["vbs_solved"]) for s in seeds if s and "vbs_solved" in s.get("test_metrics", {})]
    if not sbs_vals or not vbs_vals:
        return None, None
    return sum(sbs_vals) / len(sbs_vals), sum(vbs_vals) / len(vbs_vals)


def format_solved(val: float | None) -> str:
    if val is None:
        return "---"
    return f"{val:.1f}"


def format_diff(val: float | None, sbs: float | None) -> str:
    if val is None or sbs is None:
        return "---"
    diff = val - sbs
    return f"{diff:+.1f}"


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

    # Collect solved means: {(logic, label): float}
    solved: dict[tuple[str, str], float] = {}
    for res_dir, label in zip(result_dirs, LABELS):
        for logic in logics:
            val = get_test_solved_mean(res_dir / logic / "summary.json")
            if val is not None:
                solved[(logic, label)] = val

    # Collect SBS/VBS (use first available summary per logic)
    sbs_vbs: dict[str, tuple[float, float]] = {}
    for logic in logics:
        for res_dir in result_dirs:
            sbs, vbs = get_test_sbs_vbs_mean(res_dir / logic / "summary.json")
            if sbs is not None and vbs is not None:
                sbs_vbs[logic] = (sbs, vbs)
                break

    # Determine which labels have data
    method_columns = [c for c in DISPLAY_ORDER if c not in ("vbs",) and any((l, c) in solved for l in logics)]
    has_vbs = bool(sbs_vbs)
    value_columns = (["vbs"] if has_vbs else []) + method_columns

    has_sibyl = "sibyl" in method_columns
    n_ref = 1 if has_vbs else 0
    n_without = 4 if has_sibyl else 3
    n_with = len(method_columns) - n_without
    columns_to_bold = [c for c in value_columns if c != "vbs"]

    ncols = 1 + len(value_columns)
    col_spec = "@{}l" + "c" * len(value_columns) + "@{}"

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Header rows
    if n_ref == 1:
        lines.append(
            f" & \\multicolumn{{1}}{{c}}{{\\textbf{{Ref.}}}} & \\multicolumn{{{n_without}}}{{c}}{{\\textbf{{Without Description}}}} & \\multicolumn{{{n_with}}}{{c}}{{\\textbf{{With Description}}}} \\\\"
        )
        lines.append("\\cmidrule(lr){2-2}")
        lines.append(f"\\cmidrule(lr){{3-{2 + n_without}}}")
        lines.append(f"\\cmidrule(lr){{{3 + n_without}-{ncols}}}")
    else:
        lines.append(
            f" & \\multicolumn{{{n_without}}}{{c}}{{\\textbf{{Without Description}}}} & \\multicolumn{{{n_with}}}{{c}}{{\\textbf{{With Description}}}} \\\\"
        )
        lines.append(f"\\cmidrule(lr){{2-{1 + n_without}}}")
        lines.append(f"\\cmidrule(lr){{{2 + n_without}-{ncols}}}")

    if n_ref == 1 and has_sibyl:
        lines.append(
            " & \\multirow{2}{*}{VBS} & "
            "\\multirow{2}{*}{MachSMT} & \\multirow{2}{*}{Sibyl} & "
            f"\\multicolumn{{2}}{{c}}{{SMT-Select}} & \\multicolumn{{{n_with}}}{{c}}{{SMT-Select}} \\\\"
        )
        lines.append("\\cmidrule(lr){5-6}")
        lines.append(f"\\cmidrule(lr){{7-{ncols}}}")
    elif n_ref == 1:
        lines.append(
            " & \\multirow{2}{*}{VBS} & \\multirow{2}{*}{MachSMT} & "
            f"\\multicolumn{{2}}{{c}}{{SMT-Select}} & \\multicolumn{{{n_with}}}{{c}}{{SMT-Select}} \\\\"
        )
        lines.append("\\cmidrule(lr){4-5}")
        lines.append(f"\\cmidrule(lr){{6-{ncols}}}")
    elif has_sibyl:
        lines.append(
            " & \\multirow{2}{*}{MachSMT} & \\multirow{2}{*}{Sibyl} & "
            f"\\multicolumn{{2}}{{c}}{{SMT-Select}} & \\multicolumn{{{n_with}}}{{c}}{{SMT-Select}} \\\\"
        )
        lines.append("\\cmidrule(lr){4-5}")
        lines.append(f"\\cmidrule(lr){{6-{ncols}}}")
    else:
        lines.append(
            " & \\multirow{2}{*}{MachSMT} & \\multicolumn{2}{c}{SMT-Select} & "
            f"\\multicolumn{{{n_with}}}{{c}}{{SMT-Select}} \\\\"
        )
        lines.append("\\cmidrule(lr){3-4}")
        lines.append(f"\\cmidrule(lr){{5-{ncols}}}")

    row3_cells = [""] + [VARIANT_MAP.get(k, "") for k in value_columns]
    lines.append(" & ".join(row3_cells) + " \\\\")
    lines.append("\\midrule")

    # Data rows — values shown as difference from SBS
    for logic in logics:
        sbs_val = sbs_vbs[logic][0] if logic in sbs_vbs else None

        row_diffs: dict[str, float] = {}
        for key in columns_to_bold:
            val = solved.get((logic, key))
            if val is not None and sbs_val is not None:
                row_diffs[key] = val - sbs_val
        max_diff = max(row_diffs.values()) if row_diffs else None
        best_keys = {k for k, v in row_diffs.items() if v == max_diff} if max_diff is not None else set()

        cells = [latex_escape(logic)]
        for key in value_columns:
            if key == "vbs":
                cell = format_diff(sbs_vbs[logic][1] if logic in sbs_vbs else None, sbs_val)
            else:
                val = solved.get((logic, key))
                cell = format_diff(val, sbs_val)
            if key != "vbs" and key in best_keys:
                cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\caption{Difference in instances solved vs.\\ competition winner (SBS) on the held-out test set. Results are averaged over five random train--test splits.}",
        "\\label{tab:final_solved}",
        "\\end{table}",
    ])

    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
