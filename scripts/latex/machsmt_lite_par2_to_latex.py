#!/usr/bin/env python3
"""
Generate doc/cp26/machsmt_lite_par2.tex comparing PAR-2 gap closed (%)
for MachSMT EHM, MachSMT PWC, and SMT-Select Lite.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIRS = [
    PROJECT_ROOT / "data" / "results" / "machsmt" / "ehm",
    PROJECT_ROOT / "data" / "results" / "machsmt" / "pwc",
    PROJECT_ROOT / "data" / "results" / "lite",
]
LABELS = ["mach_ehm", "mach_pwc", "synt"]

TEX_PATH = PROJECT_ROOT / "doc" / "cp26" / "machsmt_lite_par2.tex"

COLUMN_HEADERS = ["MachSMT-EHM", "MachSMT-PWC", "SMT-Select-Lite"]


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


def format_cell(mean: float | None) -> str:
    if mean is None:
        return "---"
    return f"{mean * 100:.2f}"


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

    ncols = 1 + len(LABELS)
    col_spec = "@{}l" + "c" * len(LABELS) + "@{}"

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular*}}{{\\columnwidth}}{{@{{\\extracolsep{{\\fill}}}}{col_spec}}}",
        "\\toprule",
        " & " + " & ".join(COLUMN_HEADERS) + " \\\\",
        "\\midrule",
    ]

    for logic in logics:
        row_vals: dict[str, float] = {}
        for label in LABELS:
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
        for label in LABELS:
            mean, _ = data.get((logic, label), (None, None))
            cell = format_cell(mean)
            if label in best_keys:
                cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular*}",
        "\\caption{PAR-2 SBS--VBS gap closed (\\%) on the test set: MachSMT-EHM, MachSMT-PWC vs.\\ SMT-Select-Lite. Mean over five random train--test splits.}",
        "\\label{tab:machsmt_lite_par2}",
        "\\end{table}",
    ])

    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
